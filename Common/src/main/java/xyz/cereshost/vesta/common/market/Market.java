package xyz.cereshost.vesta.common.market;

import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

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

    public List<CandleSimple> candleSimpleList = null;

    public List<CandleSimple> cacheCandlesToArray() {
        if (candleSimpleList == null) {
            candleSimpleList = new ArrayList<>(candleSimples);
        }
        return candleSimpleList;
    }

    public void resetCandleSimpleList() {
        candleSimpleList = null;
    }

    /**
     * Devuelve una copia del mercado equivalente a un subList por indice de minuto.
     * El indice se aplica sobre las velas 1m ordenadas por openTime.
     */
    @Contract(pure = true, value = "_, _ -> new")
    public synchronized @NotNull Market subList(int fromMinuteIndex, int toMinuteIndex) {
        List<CandleSimple> orderedCandles = candleSimples.stream()
                .sorted(Comparator.comparingLong(CandleSimple::openTime))
                .toList();

        int size = orderedCandles.size();
        int from = Math.max(0, fromMinuteIndex);
        int to = Math.min(size, toMinuteIndex);

        Market copy = new Market(symbol);
        if (from >= to) {
            return copy;
        }

        List<CandleSimple> selectedCandles = orderedCandles.subList(from, to);
        long startTime = selectedCandles.getFirst().openTime();
        long endTimeExclusive = selectedCandles.getLast().openTime() + 60_000L;

        copy.candleSimples = new LinkedHashSet<>(selectedCandles);
        copy.trades = new LinkedHashSet<>(
                trades.stream()
                        .filter(t -> t.time() >= startTime && t.time() < endTimeExclusive)
                        .sorted(Comparator.comparingLong(Trade::time))
                        .toList()
        );
        copy.depths = new LinkedHashSet<>(
                depths.stream()
                        .filter(d -> d.getDate() >= startTime && d.getDate() < endTimeExclusive)
                        .sorted(Comparator.comparingLong(Depth::getDate))
                        .toList()
        );

        return copy;
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

    public double getFeed(){
        return getFeedTaker() + getFeedMaker();
    }

    public double getFeedPercent(){
        return getFeed() *100;
    }
}
