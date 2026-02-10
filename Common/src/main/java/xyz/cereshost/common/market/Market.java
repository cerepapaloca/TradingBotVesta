package xyz.cereshost.common.market;

import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;

import java.util.*;
import java.util.concurrent.*;

public class Market {

    public Market(@NotNull String symbol) {
        this.symbol = symbol;
        this.trades = new LinkedHashSet<>(100_000);
        this.depths = new LinkedHashSet<>();
        this.candleSimples = new LinkedHashSet<>(1_000);
    }

    @NotNull
    @Getter
    private final String symbol;
    @Getter
    private LinkedHashSet<Trade> trades;
    @Getter
    private LinkedHashSet<CandleSimple> candleSimples;
    @Getter
    private LinkedHashSet<Depth> depths;


    public void concat(@NotNull Market market) {
        if (!this.symbol.equals(market.symbol)) {
            throw new IllegalArgumentException("Symbols don't match");
        }
        this.trades.addAll(market.trades);
        this.depths.addAll(market.depths);
        this.candleSimples.addAll(market.candleSimples);
    }

    public synchronized void addTrade(Collection<Trade> trade) {
        Iterator<Trade> iterator = trade.iterator();
        while (iterator.hasNext()) {
            this.trades.add(iterator.next());
            iterator.remove();
        }
    }

    public synchronized void setTrade(LinkedHashSet<Trade> trades) {
        this.trades = trades;
    }

    public synchronized void addDepth(Depth tickMarker) {
        this.depths.add(tickMarker);
    }

    public synchronized void addCandles(Collection<CandleSimple> candleSimple) {
        Iterator<CandleSimple> iterator = candleSimple.iterator();
        while (iterator.hasNext()) {
            this.candleSimples.add(iterator.next());
            iterator.remove();
        }
    }

    public synchronized void setCandles(LinkedHashSet<CandleSimple> candleSimple) {
        this.candleSimples = candleSimple;
    }

    public synchronized void sortd(){
        int chunkSize = 10_000;
        trades = sortInChunks(trades, chunkSize, Trade::time);
        depths = sortInChunks(depths, chunkSize, Depth::getDate);
        candleSimples = sortInChunks(candleSimples, chunkSize, CandleSimple::openTime);
    }

    public interface TimeAccessor<T> {
        long time(T item);
    }

    @Contract(pure = true)
    public static <T> LinkedHashSet<T> sortInChunks(Collection<T> source, int chunkSize, TimeAccessor<T> accessor) {
        if (source == null || source.isEmpty()) {
            return new LinkedHashSet<>();
        }

        int safeChunkSize = Math.max(1, chunkSize);
        List<T> all = new ArrayList<>(source.size());
        Deque<T> batch = new ArrayDeque<>();

        for (Iterator<T> it = source.iterator(); it.hasNext(); ) {
            batch.add(it.next());
            it.remove();
            if (batch.size() >= safeChunkSize) {
                all.addAll(batch);
                batch.clear();
            }
        }
        // No debería quedar nada
        source.clear();
        if (!batch.isEmpty()) {
            all.addAll(batch);
            batch.clear();
        }

        all.sort(Comparator.comparingLong(accessor::time));

        LinkedHashSet<T> sorted = new LinkedHashSet<>(Math.max(16, (int) (all.size() / 0.75f) + 1));
        for (int i = 0; i < all.size(); i++) {
            sorted.add(all.get(i));
            all.set(i, null);
        }
        all.clear();
        return sorted;
    }

    @Getter
    private transient NavigableMap<Long, List<Trade>> tradesByMinuteCache;

    public List<Trade> getTradesInWindow(long startTime, long endTime) {
        if (tradesByMinuteCache == null) {
            throw new IllegalStateException("TradesByMinuteCache has not been initialized");
        }
        // Devuelve todos los trades que ocurrieron en ese minuto
        // subMap devuelve una vista, values() la colección, y flatMap las une
        return tradesByMinuteCache.subMap(startTime, true, endTime, false)
                .values().stream()
                .flatMap(List::stream)
                .sorted(Comparator.comparingLong(Trade::time)) // Asegurar orden cronológico
                .toList();
    }

    public void buildTradeCache() {
        if ((tradesByMinuteCache == null || tradesByMinuteCache.isEmpty()) && !trades.isEmpty()) {
            final LinkedHashMap<Long, List<Trade>> map = new LinkedHashMap<>();
            Iterator<Trade> it = trades.iterator();
            while (it.hasNext()) {
                Trade t = it.next();

                long minute = (t.time() / 60_000) * 60_000;
                map.computeIfAbsent(minute, k -> new ArrayList<>(20)).add(t);
                it.remove();
            }
            tradesByMinuteCache = new TreeMap<>();
            tradesByMinuteCache.putAll(map);
        }

    }

    //    public synchronized void buildTradeCache(Executor executor) {
//        if (trades.isEmpty()) {
//            tradesByMinuteCache = new TreeMap<>();
//            return;
//        }
//
//        ConcurrentSkipListMap<Long, ConcurrentLinkedQueue<Trade>> cache = new ConcurrentSkipListMap<>();
//        ConcurrentLinkedQueue<Trade> queue = new ConcurrentLinkedQueue<>();
//
//        Iterator<Trade> it = trades.iterator();
//        while (it.hasNext()) {
//            queue.add(it.next());
//            it.remove();
//        }
//
//        int workers = 8;
//        List<CompletableFuture<Void>> futures = new ArrayList<>(workers);
//
//
//        for (int i = 0; i < workers; i++) {
//            futures.add(CompletableFuture.runAsync(() -> {
//
//                Trade t;
//                final HashMap<Long, List<Trade>> local = new HashMap<>();
//                final int BATCH = 100_000; // ajustar
//                int j = 0;
//                while ((t = queue.poll()) != null) {
//
//                    long minute = (t.time() / 60_000) * 60_000;
//
//                    List<Trade> trades = local.get(minute);
//                    if (trades == null) {
//                        List<Trade> list = new ArrayList<>(20);
//                        list.add(t);
//                        local.put(minute, list);
//                    }else {
//                        trades.add(t);
//                    }
//                    //local.computeIfAbsent(minute, k -> new ArrayList<>(20)).add(t);
//
//                    if (local.size() >= BATCH) {
//                        // merge parcial
//                        mergeLocalIntoCache(local, cache);
//                        local.clear();
//                    }
//                    if ((j % 10_000) == 0) Vesta.info("J:" + j);
//                    j++;
//                }
//
//                // merge final
//                if (!local.isEmpty()) {
//                    mergeLocalIntoCache(local, cache);
//                    local.clear();
//                }
//
//            }, executor));
//        }
//
//        futures.forEach(CompletableFuture::join);
//        tradesByMinuteCache = cache;
//    }
//
//    private static void mergeLocalIntoCache(Map<Long, List<Trade>> local, ConcurrentMap<Long, ConcurrentLinkedQueue<Trade>> cache) {
//        for (Map.Entry<Long, List<Trade>> e : local.entrySet()) {
//            Long minuteKey = e.getKey(); // boxing happens aquí una vez por key
//            List<Trade> list = e.getValue();
//
//            // get/create the concurrent queue once, then addAll (fewer CAS calls)
//            cache.computeIfAbsent(minuteKey, k -> new ConcurrentLinkedQueue<>()).addAll(list);
//        }
//    }

    public synchronized Market limit(int days) {
        if (days <= 0) return this;

        // 1. Encontrar el tiempo de inicio absoluto entre todas las colecciones
        long firstTime = Long.MAX_VALUE;

        if (!candleSimples.isEmpty()) {
            firstTime = Math.min(firstTime, candleSimples.iterator().next().openTime());
        }
        if (!trades.isEmpty()) {
            firstTime = Math.min(firstTime, trades.iterator().next().time());
        }
        if (!depths.isEmpty()) {
            firstTime = Math.min(firstTime, depths.iterator().next().getDate());
        }

        // Si no hay datos, no hay nada que limitar
        if (firstTime == Long.MAX_VALUE) return this;

        // 2. Calcular el tiempo de corte (milisegundos en X días)
        long durationMs = (long) days * 24 * 60 * 60 * 1000;
        long cutoffTime = firstTime + durationMs;

        Vesta.info("Limitando datos de " + symbol + " a " + days + " días. Corte en: " + cutoffTime);

        // 3. Filtrar colecciones manteniendo solo lo que sea menor al cutoffTime
        // Usamos removeIf porque es eficiente en LinkedHashSet

        int candlesBefore = candleSimples.size();
        candleSimples.removeIf(c -> c.openTime() >= cutoffTime);

        int tradesBefore = trades.size();
        trades.removeIf(t -> t.time() >= cutoffTime);

        int depthsBefore = depths.size();
        depths.removeIf(d -> d.getDate() >= cutoffTime);

        // 4. Invalidar el cache de trades si existía, ya que los datos cambiaron
        this.tradesByMinuteCache = null;

        Vesta.info(String.format("Limpieza completada [%s]: Velas: %d->%d | Trades: %d->%d | Depth: %d->%d",
                symbol, candlesBefore, candleSimples.size(),
                tradesBefore, trades.size(),
                depthsBefore, depths.size()));
        return this;
    }

    public void clear(){
        trades.clear();
        depths.clear();
        candleSimples.clear();
        tradesByMinuteCache.clear();
    }

    public double getFeedTaker(){
        if (symbol.endsWith("USDT")) {
            return 0.0005;
        } else {
            return 0.0004;
        }
    }

    public double getFeedMaker(){
        if (symbol.endsWith("USDT")) {
            return 0.0002;
        }else {
            return 0;
        }
    }
}
