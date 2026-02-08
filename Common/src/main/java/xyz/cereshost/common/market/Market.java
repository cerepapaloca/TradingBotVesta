package xyz.cereshost.common.market;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;

import java.util.*;

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
        this.trades.addAll(trade);
    }

    public synchronized void addDepth(Depth tickMarker) {
        this.depths.add(tickMarker);
    }

    public synchronized void addCandles(Collection<CandleSimple> candleSimple) {
        this.candleSimples.addAll(candleSimple);
    }

    public synchronized void sortd(){
        int chunkSize = 10_000;
        trades = sortInChunks(trades, chunkSize, Trade::time);
        depths = sortInChunks(depths, chunkSize, Depth::getDate);
        candleSimples = sortInChunks(candleSimples, chunkSize, CandleSimple::openTime);
    }

    private interface TimeAccessor<T> {
        long time(T item);
    }

    private static <T> LinkedHashSet<T> sortInChunks(LinkedHashSet<T> source, int chunkSize, TimeAccessor<T> accessor) {
        if (source == null || source.isEmpty()) {
            return new LinkedHashSet<>();
        }

        int safeChunkSize = Math.max(1, chunkSize);
        List<T> all = new ArrayList<>(source.size());
        List<T> batch = new ArrayList<>(Math.min(safeChunkSize, source.size()));

        for (Iterator<T> it = source.iterator(); it.hasNext(); ) {
            batch.add(it.next());
            it.remove();
            if (batch.size() >= safeChunkSize) {
                all.addAll(batch);
                batch.clear();
            }
        }
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
            buildTradeCache();
        }
        // Devuelve todos los trades que ocurrieron en ese minuto
        // subMap devuelve una vista, values() la colección, y flatMap las une
        return tradesByMinuteCache.subMap(startTime, true, endTime, false)
                .values().stream()
                .flatMap(List::stream)
                .sorted(Comparator.comparingLong(Trade::time)) // Asegurar orden cronológico
                .toList();
    }

    public synchronized void buildTradeCache() {
        tradesByMinuteCache = new TreeMap<>();
        for (Trade t : trades) {
            long minute = (t.time() / 60_000) * 60_000;
            tradesByMinuteCache.computeIfAbsent(minute, k -> new ArrayList<>()).add(t);
        }
    }

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
