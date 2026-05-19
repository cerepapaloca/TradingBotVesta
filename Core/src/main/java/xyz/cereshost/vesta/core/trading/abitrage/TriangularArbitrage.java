package xyz.cereshost.vesta.core.trading.abitrage;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.SneakyThrows;
import org.jetbrains.annotations.Blocking;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.Main;
import xyz.cereshost.vesta.core.market.MarketStatus;
import xyz.cereshost.vesta.core.market.SymbolConfigurable;
import xyz.cereshost.vesta.core.trading.real.api.BinanceWebSocketFull;
import xyz.cereshost.vesta.core.trading.real.api.model.BookTicker;
import xyz.cereshost.vesta.core.trading.real.api.model.ExchangeInfo;
import xyz.cereshost.vesta.core.trading.real.api.model.Ticker24H;

import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Consumer;

@RequiredArgsConstructor
public class TriangularArbitrage {

    private static final double PROFIT_EPSILON = 1e-12;
    private static final double DEFAULT_FEE_RATE = 0.00075; // 0.075% aprox
    private static final int MIN_CYCLE_LENGTH = 3;
    private static final int MAX_CYCLE_LENGTH = 3;

    private final BinanceWebSocketFull binanceApi;
    private final Consumer<List<TriangularArbitrageOpportunity>> onOpportunity;

    private volatile boolean started = false;
    @Nullable private volatile ExchangeInfo exchangeInfoSpot = null;
    @Nullable private volatile Consumer<BookTicker> streamListener = null;
    @NotNull private final ConcurrentMap<String, BookTicker> liveTickers = new ConcurrentHashMap<>();
    @NotNull private final AtomicBoolean calculationInProgress = new AtomicBoolean(false);
    @NotNull private final AtomicBoolean calculationRequested = new AtomicBoolean(false);
    @NotNull private final BlockingDeque<AtomicReference<BookTicker>> pendingUpdatedTicker = new LinkedBlockingDeque<>(256);
    @Nullable private volatile Executor calculationExecutor = null;

    @Blocking
    public void startSearch(Executor executor) {
        if (started) {
            return;
        }
        started = true;
        calculationExecutor = executor;

        CompletableFuture<ExchangeInfo> exchangeInfoFuture = CompletableFuture.supplyAsync(
                () -> binanceApi.getRequest().getExchangeInfo(false),
                Main.EXECUTOR
        );
        CompletableFuture<Map<String, BookTicker>> tickersFuture = CompletableFuture.supplyAsync(
                () -> binanceApi.getRequest().getBookTickers(null, false),
                Main.EXECUTOR
        );

        executor.execute(() -> {
            try {
                exchangeInfoSpot = exchangeInfoFuture.get();
                liveTickers.clear();
                liveTickers.putAll(tickersFuture.get());

                Set<Ticker24H> ticker24H = binanceApi.getRequest().getTicker24H(null);
                HashMap<String, Ticker24H> bookTicker24H = new HashMap<>();
                for (Ticker24H ticker : ticker24H) {
                    bookTicker24H.put(ticker.symbol(), ticker);
                }

                Set<String> symbolsToSubscribe = getSpotTradingSymbols(Objects.requireNonNull(exchangeInfoSpot), bookTicker24H);
                liveTickers.keySet().retainAll(symbolsToSubscribe);
                Consumer<BookTicker> listener = this::onBookTickerUpdate;
                streamListener = listener;
                binanceApi.getStream().subscribeIndividualSymbolBookTickerStreams(
                        symbolsToSubscribe,
                        listener
                );
                requestCalculation(null);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                stopSearch();
                Vesta.sendWaringException("Error iniciando stream de arbitraje", e);
            } catch (ExecutionException e) {
                stopSearch();
                Vesta.sendWaringException("Error al hacer solicitud a binance", e);
            } catch (Exception e) {
                stopSearch();
                Vesta.sendWaringException("Error suscribiendo streams de bookTicker", e);
            }
        });
    }

    public void stopSearch() {
        started = false;
        Consumer<BookTicker> listener = streamListener;
        if (listener != null) {
            binanceApi.getStream().removeBookTickerListener(listener);
        }
        streamListener = null;
        exchangeInfoSpot = null;
        liveTickers.clear();
        pendingUpdatedTicker.clear();
        calculationRequested.set(false);
        lastTriangular.clear();
    }

    private void onBookTickerUpdate(@NotNull BookTicker bookTicker) {
        if (!started) {
            return;
        }
        requestCalculation(bookTicker);
    }

    private void requestCalculation(@Nullable BookTicker updatedTicker) {
        if (updatedTicker != null) {
            liveTickers.put(updatedTicker.symbol(), updatedTicker);
            enqueueUpdatedTicker(new AtomicReference<>(updatedTicker));
        } else {
            enqueueUpdatedTicker(new AtomicReference<>(null));
        }

        calculationRequested.set(true);
        tryStartCalculationLoop();
    }

    private void enqueueUpdatedTicker(@NotNull AtomicReference<BookTicker> updatedTickerRef) {
        if (pendingUpdatedTicker.offerLast(updatedTickerRef)) {
            return;
        }

        // Cola llena: descartamos el mÃ¡s antiguo para conservar el estado mÃ¡s reciente.
        pendingUpdatedTicker.pollFirst();
        if (pendingUpdatedTicker.offerLast(updatedTickerRef)) {
            return;
        }

        // Fallback defensivo frente a carreras extremas.
        pendingUpdatedTicker.clear();
        pendingUpdatedTicker.offerLast(updatedTickerRef);
    }

    private void tryStartCalculationLoop() {
        Executor executor = calculationExecutor;
        if (executor == null) {
            return;
        }
        if (!calculationInProgress.compareAndSet(false, true)) {
            return;
        }
        executor.execute(this::runCalculationLoop);
    }

    private void runCalculationLoop() {
        try {
            while (started && calculationRequested.getAndSet(false)) {
                ExchangeInfo exchangeInfo = exchangeInfoSpot;
                if (exchangeInfo == null) {
                    return;
                }
                // Coalescemos ráfagas: procesamos el evento más reciente disponible.
                AtomicReference<BookTicker> updatedTickerRef = pendingUpdatedTicker.pollLast();
                if (updatedTickerRef == null) {
                    updatedTickerRef = pendingUpdatedTicker.pollFirst();
                }
                pendingUpdatedTicker.clear();
                BookTicker updatedTicker = updatedTickerRef == null ? null : updatedTickerRef.get();
                List<TriangularArbitrageOpportunity> list = findTriangularArbitrageOpportunities(
                        exchangeInfo,
                        new HashMap<>(liveTickers),
                        updatedTicker
                );
                onOpportunity.accept(list);
            }
        } catch (Exception e) {
            Vesta.sendWaringException("Error calculando arbitrajes triangulares", e);
        } finally {
            calculationInProgress.set(false);
            if (started && calculationRequested.get()) {
                tryStartCalculationLoop();
            }
        }
    }

    private @NotNull Set<String> getSpotTradingSymbols(@NotNull ExchangeInfo exchangeInfo, @NotNull HashMap<String, Ticker24H> bookTicker24H) {
        Map<String, List<AssetRate>> conversionGraph = buildAssetConversionGraph(exchangeInfo);
        List<SymbolVolume> candidates = new ArrayList<>();

        for (SymbolConfigurable symbolConfigurable : exchangeInfo.symbols()) {
            if (!symbolConfigurable.getIsSpot()) continue;
            if (!MarketStatus.TRADING.equals(symbolConfigurable.getMarketStatus())) continue;
            if (!symbolConfigurable.getIsAllowTrading()) continue;

            Ticker24H ticker24H = bookTicker24H.get(symbolConfigurable.name());
            if (ticker24H == null) continue;

            double quoteVolume = ticker24H.quoteVolumen() == null ? 0.0 : ticker24H.quoteVolumen();
            double baseVolume = ticker24H.baseVolumen() == null ? 0.0 : ticker24H.baseVolumen();
            double volumeUsdt = 0.0;

            if (quoteVolume > 0.0) {
                volumeUsdt = convertAssetAmountToUsdt(symbolConfigurable.getQuoteAsset(), quoteVolume, conversionGraph);
            }
            if (volumeUsdt <= 0.0 && baseVolume > 0.0) {
                volumeUsdt = convertAssetAmountToUsdt(symbolConfigurable.getBaseAsset(), baseVolume, conversionGraph);
            }

            candidates.add(new SymbolVolume(symbolConfigurable.name(), volumeUsdt));
        }

        candidates.sort((a, b) -> Double.compare(b.volumeUsdt(), a.volumeUsdt()));
        int limit = Math.min(1000, candidates.size());
        Set<String> result = new HashSet<>(limit);
        for (int i = 0; i < limit; i++) {
            result.add(candidates.get(i).symbol());
        }

        return result;
    }

    private @NotNull Map<String, List<AssetRate>> buildAssetConversionGraph(@NotNull ExchangeInfo exchangeInfo) {
        Map<String, List<AssetRate>> graph = new HashMap<>();
        for (SymbolConfigurable symbolConfigurable : exchangeInfo.symbols()) {
            if (!symbolConfigurable.getIsSpot()) continue;
            if (!MarketStatus.TRADING.equals(symbolConfigurable.getMarketStatus())) continue;

            BookTicker ticker = liveTickers.get(symbolConfigurable.name());
            if (ticker == null || ticker.bidPrice() == null || ticker.askPrice() == null) continue;

            double bid = ticker.bidPrice();
            double ask = ticker.askPrice();
            if (bid <= 0.0 || ask <= 0.0) continue;

            double midPrice = (bid + ask) / 2.0;
            if (midPrice <= 0.0) continue;

            String base = symbolConfigurable.getBaseAsset();
            String quote = symbolConfigurable.getQuoteAsset();
            graph.computeIfAbsent(base, k -> new ArrayList<>()).add(new AssetRate(quote, midPrice));
            graph.computeIfAbsent(quote, k -> new ArrayList<>()).add(new AssetRate(base, 1.0 / midPrice));
        }
        return graph;
    }

    private double convertAssetAmountToUsdt(
            @NotNull String asset,
            double amount,
            @NotNull Map<String, List<AssetRate>> conversionGraph
    ) {
        if (amount <= 0.0) return 0.0;
        if ("USDT".equalsIgnoreCase(asset)) return amount;

        record Node(String asset, double amount) {}

        Deque<Node> queue = new ArrayDeque<>();
        Set<String> visited = new HashSet<>();
        queue.add(new Node(asset, amount));
        visited.add(asset);

        while (!queue.isEmpty()) {
            Node current = queue.poll();
            List<AssetRate> rates = conversionGraph.get(current.asset());
            if (rates == null) continue;

            for (AssetRate rate : rates) {
                double convertedAmount = current.amount() * rate.rate();
                if (convertedAmount <= 0.0) continue;

                if ("USDT".equalsIgnoreCase(rate.toAsset())) {
                    return convertedAmount;
                }
                if (visited.add(rate.toAsset())) {
                    queue.add(new Node(rate.toAsset(), convertedAmount));
                }
            }
        }

        return 0.0;
    }

    private record AssetRate(
            String toAsset,
            double rate
    ) {}

    private record SymbolVolume(
            String symbol,
            double volumeUsdt
    ) {}

    @SneakyThrows
    public synchronized @NotNull List<TriangularArbitrageOpportunity> findTriangularArbitrageOpportunities(
            @NotNull ExchangeInfo exchangeInfo,
            @NotNull Map<String, BookTicker> tickers
    ) {
        return findTriangularArbitrageOpportunities(exchangeInfo, tickers, null);
    }

    private final ConcurrentMap<String, LifeTime> lastTriangular = new ConcurrentHashMap<>();

    @SneakyThrows
    public synchronized @NotNull List<TriangularArbitrageOpportunity> findTriangularArbitrageOpportunities(
            @NotNull ExchangeInfo exchangeInfo,
            @NotNull Map<String, BookTicker> tickers,
            @Nullable BookTicker updatedTicker
    ) {

        Map<String, ArrayList<ArbitrageEdge>> outgoingByFromAsset = new HashMap<>();
        String updatedSymbol = updatedTicker == null ? null : updatedTicker.symbol();
        Set<String> trackedAssets = trackedAssetsFromLastTriangular();
        Set<String> startAssetsToAnalyze = null;

        for (SymbolConfigurable symbolConfigurable : exchangeInfo.symbols()) {
            if (!MarketStatus.TRADING.equals(symbolConfigurable.getMarketStatus())) {
                continue;
            }

            // Solo spot para arbitraje triangular clásico
            if (!symbolConfigurable.getIsSpot()) {
                continue;
            }

            String symbolName = symbolConfigurable.name();
            BookTicker ticker = tickers.get(symbolName);
            if (ticker == null) {
                continue;
            }

            Double bidObj = ticker.bidPrice();
            Double askObj = ticker.askPrice();
            if (bidObj == null || askObj == null) {
                continue;
            }

            double bid = bidObj;
            double ask = askObj;
            if (bid <= 0.0 || ask <= 0.0) {
                continue;
            }

            String baseAsset = symbolConfigurable.getBaseAsset();
            String quoteAsset = symbolConfigurable.getQuoteAsset();
            if (baseAsset.equals("?") || quoteAsset.equals("?")) {
                continue;
            }

            double sellRate = bid * (1.0 - DEFAULT_FEE_RATE);
            double buyRate = (1.0 / ask) * (1.0 - DEFAULT_FEE_RATE);

            if (sellRate > 0.0) {
                addEdge(outgoingByFromAsset, new ArbitrageEdge(
                        symbolName,
                        new NameAsset(baseAsset),
                        new NameAsset(quoteAsset),
                        sellRate,
                        -Math.log(sellRate),
                        "SELL",
                        bid
                ));
            }

            if (buyRate > 0.0) {
                addEdge(outgoingByFromAsset, new ArbitrageEdge(
                        symbolName,
                        new NameAsset(quoteAsset),
                        new NameAsset(baseAsset),
                        buyRate,
                        -Math.log(buyRate),
                        "BUY",
                        ask
                ));
            }

            if ((updatedSymbol != null && updatedSymbol.equals(symbolName))
                    || trackedAssets.contains(baseAsset)
                    || trackedAssets.contains(quoteAsset)) {
                if (startAssetsToAnalyze == null) {
                    startAssetsToAnalyze = new LinkedHashSet<>(2);
                }
                startAssetsToAnalyze.add(baseAsset);
                startAssetsToAnalyze.add(quoteAsset);
            }
        }

        if (outgoingByFromAsset.size() < MIN_CYCLE_LENGTH) {
            return List.of();
        }

        if (updatedSymbol != null) {
            if (startAssetsToAnalyze == null || startAssetsToAnalyze.isEmpty()) {
                return List.of();
            }
            startAssetsToAnalyze.retainAll(outgoingByFromAsset.keySet());
            if (startAssetsToAnalyze.isEmpty()) {
                return List.of();
            }
        }

        Set<String> seenCycles = new HashSet<>();
        List<TriangularArbitrageOpportunity> opportunities = new ArrayList<>();
        Set<String> startAssets = new LinkedHashSet<>();
        startAssets.addAll(Objects.requireNonNullElseGet(startAssetsToAnalyze, outgoingByFromAsset::keySet));
        startAssets.addAll(trackedAssets);
        startAssets.retainAll(outgoingByFromAsset.keySet());
        if (startAssets.isEmpty()) {
            return List.of();
        }

        for (String startAsset : startAssets) {
            Deque<ArbitrageEdge> path = new ArrayDeque<>(MAX_CYCLE_LENGTH);
            Set<String> visitedAssets = new HashSet<>();
            visitedAssets.add(startAsset);
            searchCyclesFrom(
                    new NameAsset(startAsset),
                    new NameAsset(startAsset),
                    outgoingByFromAsset,
                    path,
                    new IntegerAtomic(0),
                    visitedAssets,
                    seenCycles,
                    opportunities
            );
        }
        Set<String> activeCycleKeys = new HashSet<>();
        for (TriangularArbitrageOpportunity opportunity : opportunities) {
            activeCycleKeys.add(canonicalCycleKey(opportunity.assetsCycle()));
        }
        lastTriangular.keySet().removeIf(key -> !activeCycleKeys.contains(key));

        opportunities.sort(Comparator.comparingDouble(TriangularArbitrageOpportunity::profitPercent).reversed());

        return opportunities;/*.stream()
                .filter(opportunity -> opportunity.edges().stream()
                        .anyMatch(edge -> updatedSymbol.equals(edge.symbol())))
                .toList();*/
    }


    private void addEdge(@NotNull Map<String, ArrayList<ArbitrageEdge>> outgoingByFromAsset,
                         @NotNull ArbitrageEdge edge) {
        outgoingByFromAsset
                .computeIfAbsent(edge.fromAsset().asset, key -> new ArrayList<>())
                .add(edge);
    }


    private void searchCyclesFrom(
            @NotNull TriangularArbitrage.NameAsset startAsset,
            @NotNull TriangularArbitrage.NameAsset currentAsset,
            @NotNull Map<String, ArrayList<ArbitrageEdge>> outgoingByFromAsset,
            @NotNull Deque<ArbitrageEdge> path,
            @NotNull IntegerAtomic sizePath,
            @NotNull Set<String> visitedAssets,
            @NotNull Set<String> seenCycles,
            @NotNull List<TriangularArbitrageOpportunity> opportunities
    ) {
        ArrayList<ArbitrageEdge> outgoing = outgoingByFromAsset.get(currentAsset.asset);
        if (outgoing == null || outgoing.isEmpty()) {
            return;
        }

        for (ArbitrageEdge edge : outgoing) {
            int nextLength = sizePath.get() + 1;


            if ((startAsset.hash == edge.toAsset.hash)) {
                if (nextLength < MIN_CYCLE_LENGTH || nextLength > MAX_CYCLE_LENGTH) {
                    continue;
                }

                path.addLast(edge);
                TriangularArbitrageOpportunity opportunity = buildOpportunityFromEdges(new ArrayList<>(path));
                path.removeLast();

                if (opportunity == null) {
                    continue;
                }

                String canonicalKey = canonicalCycleKey(opportunity.assetsCycle());
                if (seenCycles.add(canonicalKey)) {
                    opportunities.add(opportunity);
                }
                continue;
            }

            if (nextLength >= MAX_CYCLE_LENGTH) {
                continue;
            }
            if (visitedAssets.contains(edge.toAsset().asset)) {
                continue;
            }

            path.addLast(edge);
            sizePath.increment();
            visitedAssets.add(edge.toAsset().asset);
            searchCyclesFrom(
                    startAsset,
                    edge.toAsset(),
                    outgoingByFromAsset,
                    path,
                    sizePath,
                    visitedAssets,
                    seenCycles,
                    opportunities
            );
            visitedAssets.remove(edge.toAsset().asset);
            sizePath.decrement();
            path.removeLast();
        }
    }

    private @Nullable TriangularArbitrageOpportunity buildOpportunityFromEdges(@NotNull List<ArbitrageEdge> cycleEdges) {
        int cycleLength = cycleEdges.size();
        if (cycleLength < MIN_CYCLE_LENGTH || cycleLength > MAX_CYCLE_LENGTH) {
            return null;
        }

        ArbitrageEdge first = cycleEdges.getFirst();
        String startAsset = first.fromAsset().asset;
        String currentAsset = startAsset;

        List<String> cycleAssets = new ArrayList<>(cycleLength + 1);
        cycleAssets.add(startAsset);

        Set<String> distinctAssets = new HashSet<>();
        distinctAssets.add(startAsset);

        double rateProduct = 1.0;
        double totalWeight = 0.0;

        for (int i = 0; i < cycleLength; i++) {
            ArbitrageEdge edge = cycleEdges.get(i);
            if (!currentAsset.equals(edge.fromAsset().asset)) {
                return null;
            }

            currentAsset = edge.toAsset().asset;
            cycleAssets.add(currentAsset);

            rateProduct *= edge.rate();
            totalWeight += edge.weight();

            if (i < cycleLength - 1 && !distinctAssets.add(currentAsset)) {
                return null;
            }
        }

        if (!startAsset.equals(currentAsset)) {
            return null;
        }
        if (distinctAssets.size() != cycleLength) {
            return null;
        }
        if (rateProduct <= 1.0 + PROFIT_EPSILON) {
            return null;
        }
        if (totalWeight >= -PROFIT_EPSILON) {
            return null;
        }

        String cycleKey = canonicalCycleKey(cycleAssets);
        LifeTime lifeTime = lastTriangular.computeIfAbsent(cycleKey, s -> new LifeTime()).nextTicks();

        return new TriangularArbitrageOpportunity(
                cycleAssets,
                List.copyOf(cycleEdges),
                lifeTime,
                rateProduct,
                (rateProduct - 1.0) * 100.0,
                totalWeight
        );
    }

    private @NotNull Set<String> trackedAssetsFromLastTriangular() {
        Set<String> trackedAssets = new HashSet<>();
        for (String cycleKey : lastTriangular.keySet()) {
            String[] assets = cycleKey.split("->");
            int limit = Math.max(0, assets.length - 1); // el último repite el inicio
            trackedAssets.addAll(Arrays.asList(assets).subList(0, limit));
        }
        return trackedAssets;
    }

    private @NotNull String canonicalCycleKey(@NotNull List<String> cycleAssets) {
        List<String> raw = new ArrayList<>(cycleAssets.subList(0, cycleAssets.size() - 1));
        int size = raw.size();

        List<String> best = null;
        for (int shift = 0; shift < size; shift++) {
            List<String> rotated = new ArrayList<>(size);
            for (int i = 0; i < size; i++) {
                rotated.add(raw.get((shift + i) % size));
            }

            if (best == null || compareLex(rotated, best) < 0) {
                best = rotated;
            }
        }

        return String.join("->", best) + "->" + best.getFirst();
    }

    private int compareLex(@NotNull List<String> a, @NotNull List<String> b) {
        for (int i = 0; i < a.size(); i++) {
            int cmp = a.get(i).compareTo(b.get(i));
            if (cmp != 0) {
                return cmp;
            }
        }
        return 0;
    }

    public record ArbitrageEdge(
            String symbol,
            NameAsset fromAsset,
            NameAsset toAsset,
            double rate,
            double weight,
            String action,
            double referencePrice
    ) {}

    public record TriangularArbitrageOpportunity(
            List<String> assetsCycle,
            List<ArbitrageEdge> edges,
            LifeTime lifeTime,
            double rateProduct,
            double profitPercent,
            double totalWeight
    ) {}

    @Getter
    public static class LifeTime {
        private final long dateOpen = System.currentTimeMillis();
        private int ticks;


        public LifeTime nextTicks() {
            this.ticks++;
            return this;
        }
    }

    public static class NameAsset {
        public final String asset;
        public final long hash;

        private static final HashMap<String, Byte[]> cacheBytes = new HashMap<>();

        public NameAsset(String asset) {
            this.asset = asset;
            long h = 0;
            for (byte b : cacheBytes.computeIfAbsent(asset, a -> {
                byte[] bytes = a.getBytes(StandardCharsets.UTF_8);
                Byte[] byteArray = new Byte[bytes.length];
                for (int i = 0; i < bytes.length; i++) {
                    byteArray[i] = bytes[i];
                }
                return byteArray;
            })) {
                h = 7 * ((h + 37) << b);
            }
            this.hash = h;
        }
    }

    private static class IntegerAtomic {
        public int value;
        public IntegerAtomic(int value) {
            this.value = value;
        }

        public void increment() {
            value++;
        }

        public void decrement() {
            value--;
        }

        public int get() {
            return value;
        }
    }
}
