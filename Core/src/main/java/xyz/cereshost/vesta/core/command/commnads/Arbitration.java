package xyz.cereshost.vesta.core.command.commnads;

import lombok.RequiredArgsConstructor;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.Main;
import xyz.cereshost.vesta.core.command.Arguments;
import xyz.cereshost.vesta.core.command.BaseCommand;
import xyz.cereshost.vesta.core.market.DireccionOperation;
import xyz.cereshost.vesta.core.market.SymbolConfigurable;
import xyz.cereshost.vesta.core.message.DiscordNotification;
import xyz.cereshost.vesta.core.message.MediaNotification;
import xyz.cereshost.vesta.core.trading.TypeOrder;
import xyz.cereshost.vesta.core.trading.abitrage.TriangularArbitrage;
import xyz.cereshost.vesta.core.trading.real.api.BinanceApiRest;
import xyz.cereshost.vesta.core.trading.real.api.BinanceWebSocketFull;
import xyz.cereshost.vesta.core.utils.LoaderIndicator;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

public class Arbitration extends BaseCommand {

    public Arbitration() {
        super("Ejecuta una estrategia de arbitraje triangular");
    }

    private final Set<TriangularArbitrage.TriangularArbitrageOpportunity> opportunityWindows = new HashSet<>();

    @Override
    public void execute(Arguments arguments) throws Exception {
        BinanceWebSocketFull apiWebSocket = new BinanceWebSocketFull(false);

        LoaderIndicator loaderIndicator = new LoaderIndicator(10);
        loaderIndicator.setLabel("Buscado Arbitrajes...");

        BinanceApiRest apiRest = new BinanceApiRest(true, true);

        MediaNotification mediaNotification = new DiscordNotification();

        mediaNotification.updateStatusType(MediaNotification.StatusType.WORKING);
        mediaNotification.updateStatus("Analizado todos los mercados");

        AtomicLong windowStart = new AtomicLong(0L);
        ExecutorArbitrage executorArbitrage = new ExecutorArbitrage(Main.EXECUTOR, apiRest, mediaNotification);
        AtomicLong counterDecent = new AtomicLong(0L);

        TriangularArbitrage triangularArbitrage = new TriangularArbitrage(apiWebSocket, opportunities -> {
            updateLoader(loaderIndicator, counterDecent);
            updateStatus(mediaNotification, counterDecent);
            HashSet<TriangularArbitrage.TriangularArbitrageOpportunity> current = new HashSet<>(opportunities);
            boolean changed = !current.equals(opportunityWindows);
            long currentTime = System.currentTimeMillis();

            executorArbitrage.onTick(opportunities);
            executorArbitrage.tryRunLoop();
            if (!changed) {
                return;
            }

//            long duration = currentTime - windowStart.get();
            if (current.isEmpty() && !opportunityWindows.isEmpty()) {
                loaderIndicator.clearLine();
                ArrayList<TriangularArbitrage.TriangularArbitrageOpportunity> arrayOpportunity = new ArrayList<>(opportunityWindows);

                for (int i = 0; i < opportunityWindows.size(); i++) {
                    TriangularArbitrage.TriangularArbitrageOpportunity opportunity = arrayOpportunity.get(i);
                    if (opportunity.lifeTime().getTicks() > 10){
                        long delta = currentTime - opportunity.lifeTime().getDateOpen();
                        Vesta.info(
                                "[%d] %s | retorno %.6f | profit %.4f%% | peso %.8f | %,dms (%d Tick)",
                                i + 1,
                                String.join(" -> ", opportunity.assetsCycle()),
                                opportunity.rateProduct(),
                                opportunity.profitPercent(),
                                opportunity.totalWeight(),
                                delta,
                                opportunity.lifeTime().getTicks()
                        );
                        for (TriangularArbitrage.ArbitrageEdge edge : opportunity.edges()) {
                            Vesta.info(
                                    "    %s %s via %s @ %.10f -> rate %.10f ",
                                    edge.action(),
                                    edge.fromAsset().asset + "/" + edge.toAsset().asset,
                                    edge.symbol(),
                                    edge.referencePrice(),
                                    edge.rate(),
                                    edge.weight()
                            );
                        }
                    }
                }
                executorArbitrage.onClose();
//                Vesta.info("Fin de ventana de arbitraje (%d ms)", duration);
                windowStart.set(-1);
            }

            if (!current.isEmpty()) {
                loaderIndicator.clearLine();
                if (windowStart.get() == -1) {
                    windowStart.set(currentTime);
//                    Vesta.info("Inicio ventana de arbitraje");
                }
                executorArbitrage.onOpportunity(opportunities);
                counterDecent.incrementAndGet();
//                Vesta.info("Arbitrajes detectados: %d", opportunities.size());
            }

            opportunityWindows.clear();
            opportunityWindows.addAll(current);

        });
        triangularArbitrage.startSearch(Main.EXECUTOR);
        Main.EXECUTOR.scheduleAtFixedRate(() -> {
            executorArbitrage.onTick(List.of());
            triangularArbitrage.stopSearch();
            triangularArbitrage.startSearch(Main.EXECUTOR);
        }, 2, 2, TimeUnit.HOURS);
    }

    private final Queue<Long> deltasProcessing = new LinkedList<>();
    private final AtomicLong startProcessing = new AtomicLong(System.currentTimeMillis());

    private void updateLoader(LoaderIndicator loaderIndicator, AtomicLong counter) {
        long time = System.currentTimeMillis();
        if (deltasProcessing.size() > 100) deltasProcessing.poll();

        deltasProcessing.offer(time - startProcessing.get());
        startProcessing.set(time);
        double avgProcessing = deltasProcessing.stream().mapToLong(Long::longValue).average().getAsDouble();
        loaderIndicator.setLabel("%dms (%.2fu/s) %d Buscando posibles arbitrajes...".formatted((int) avgProcessing, 1000 / avgProcessing, counter.get()));
        loaderIndicator.printAndNexStep();
    }

    private long coolDown = System.currentTimeMillis();

    public void updateStatus(MediaNotification media, AtomicLong counter) {
        long time = System.currentTimeMillis();
        if (coolDown < time) {
            media.updateStatus("Posibles Arbitrajes detectados %d (%.2fu/s)", counter.get(), 1000 / deltasProcessing.stream().mapToLong(Long::longValue).average().getAsDouble());
            coolDown = time + TimeUnit.SECONDS.toMillis(15);
        }
    }

    @RequiredArgsConstructor
    public static class ExecutorArbitrage {
        private final @NotNull ScheduledExecutorService executor;
        private final @NotNull BinanceApiRest binanceApi;
        private final @NotNull HashMap<String, SymbolConfigurable> symbolsByName;
        private final @NotNull MediaNotification mediaNotification;

        public ExecutorArbitrage(@NotNull ScheduledExecutorService executor,
                                 @NotNull BinanceApiRest binanceApi,
                                 @NotNull MediaNotification mediaNotification
        ) {
            this.executor = executor;
            this.binanceApi = binanceApi;
            HashMap<String, SymbolConfigurable> symbolsByName = new HashMap<>();
            for (SymbolConfigurable symbolConfigurable : binanceApi.getExchangeInfo(false).symbols()) {
                symbolsByName.put(symbolConfigurable.name(), symbolConfigurable);
            }
            this.symbolsByName = symbolsByName;
            this.mediaNotification = mediaNotification;
        }

        private volatile TriangularArbitrage.TriangularArbitrageOpportunity opportunity;
        private volatile List<TriangularArbitrage.TriangularArbitrageOpportunity> lastOpportunities = List.of();

        public void onTick(@NotNull List<TriangularArbitrage.TriangularArbitrageOpportunity> opportunities){
            lastOpportunities = opportunities;
        }

        public synchronized void onOpportunity(@NotNull List<TriangularArbitrage.TriangularArbitrageOpportunity> opportunities) {
            TriangularArbitrage.TriangularArbitrageOpportunity best = null;
            for (TriangularArbitrage.TriangularArbitrageOpportunity opportunity : opportunities) {
                if (checkOpportunity(opportunity)) {
                    if (best == null || opportunity.profitPercent() > best.profitPercent()) {
                        best = opportunity;
                    }
                }
            }

            if (best != null) {
                final TriangularArbitrage.TriangularArbitrageOpportunity b = best;
                executor.schedule(() -> {
                    for (TriangularArbitrage.TriangularArbitrageOpportunity opportunity : lastOpportunities) {
                        if (b.edges().size() == opportunity.edges().size() && new HashSet<>(b.assetsCycle()).containsAll(opportunity.assetsCycle())) {

                            TriangularArbitrage.ArbitrageEdge USDT = null;

                            for (TriangularArbitrage.ArbitrageEdge edge : opportunity.edges()) {
                                if (Objects.equals(edge.fromAsset().asset, "USDT")){
                                    USDT = edge;
                                    break;
                                }
                            }
                            // No debería ser nulo
                            if (USDT == null) {
                                Vesta.info("No hay USDT ignorando el arbitraje");
                                return;
                            }
                            int index = opportunity.edges().indexOf(USDT);
                            if (index != -1) {
                                Collections.rotate(opportunity.edges(), -index);
                            }
                            Vesta.info(b.assetsCycle() +" == " + opportunity.assetsCycle());
                            this.opportunity = b;
                            break;
                        }else {
                            Vesta.info(b.assetsCycle() +" != " + opportunity.assetsCycle());
                        }
                    }
                }, 200, TimeUnit.MILLISECONDS);
            }

        }

        public synchronized void onClose() {
            this.opportunity = null;
        }

        private volatile CompletableFuture<Object> runLoop = CompletableFuture.completedFuture(null);

        public synchronized void tryRunLoop() {
//            if (opportunity != null) {
//                Vesta.info("OPP: " +true);
//            }
            if (runLoop.isDone()) {
                runLoop = CompletableFuture.supplyAsync(() -> {
                    while (opportunity != null) {
                        Vesta.clearLine();
                        List<TriangularArbitrage.ArbitrageEdge> edges = new ArrayList<>(opportunity.edges());

                        Vesta.info("Iniciando bucle con: " + edges.stream().map(TriangularArbitrage.ArbitrageEdge::symbol).toList().toString());
                        boolean isFirst = true;
                        for (TriangularArbitrage.ArbitrageEdge edge : edges) {
                            Double balance;
                            if (isFirst) {
                                isFirst = false;
                                balance = 5d;
                            }else {
                                System.out.println(edge.fromAsset().asset);
                                balance = binanceApi.getBalance(false).get(edge.fromAsset().asset);
                            }
                            binanceApi.placeOrder(
                                    symbolsByName.get(edge.symbol()),
                                    DireccionOperation.parse(edge.action()),
                                    TypeOrder.MARKET,
                                    null,
                                    balance,
                                    null,
                                    false,
                                    false
                            );
//                            LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(250));
                            Vesta.clearLine();
                            Vesta.info("Ejecutando: %s %s", edge.symbol(), edge.action());
                        }
//                        mediaNotification.info("Arbitrage %s (Teórico) en **%s**",
//                                opportunity != null ? "Ganado" : "Perdido",
//                                String.join(" -> ", opportunity.assetsCycle())
//                        );
                        Vesta.clearLine();
                        Vesta.info("Ciclo Completado: " + (opportunity != null ? "Ganado" : "Perdido"));
                    }
                    return new Object();
                });
            }
        }

        private boolean checkOpportunity(TriangularArbitrage.TriangularArbitrageOpportunity opportunity) {
            for (TriangularArbitrage.ArbitrageEdge edge : opportunity.edges()) {
                if (edge.fromAsset().asset.equals("USDT")) {
                    return true;
                }
            }
            return false;
        }
    }
}
