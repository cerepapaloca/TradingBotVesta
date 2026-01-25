package xyz.cereshost.common.market;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;

import java.util.*;

public class Market {

    public Market(@NotNull String symbol) {
        this.symbol = symbol;
        this.trades = new LinkedHashSet<>();
        this.depths = new LinkedHashSet<>();
        this.candleSimples = new LinkedHashSet<>();
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
        trades = trades.stream().sorted(Comparator.comparingLong(Trade::time)).collect(LinkedHashSet::new, LinkedHashSet::add, LinkedHashSet::addAll);
        depths = depths.stream().sorted(Comparator.comparingLong(Depth::getDate)).collect(LinkedHashSet::new, LinkedHashSet::add, LinkedHashSet::addAll);
        candleSimples = candleSimples.stream().sorted(Comparator.comparingLong(CandleSimple::openTime)).collect(LinkedHashSet::new, LinkedHashSet::add, LinkedHashSet::addAll);
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
