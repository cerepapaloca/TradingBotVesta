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

    public void addTrade(Collection<Trade> trade) {
        this.trades.addAll(trade);
    }

    public void addDepth(Depth tickMarker) {
        this.depths.add(tickMarker);
    }

    public void addCandles(Collection<CandleSimple> candleSimple) {
        this.candleSimples.addAll(candleSimple);
    }

    public synchronized void sortd(){
        trades = trades.stream().sorted(Comparator.comparingLong(Trade::time)).collect(LinkedHashSet::new, LinkedHashSet::add, LinkedHashSet::addAll);
        depths = depths.stream().sorted(Comparator.comparingLong(Depth::getDate)).collect(LinkedHashSet::new, LinkedHashSet::add, LinkedHashSet::addAll);
        candleSimples = candleSimples.stream().sorted(Comparator.comparingLong(CandleSimple::openTime)).collect(LinkedHashSet::new, LinkedHashSet::add, LinkedHashSet::addAll);
    }
}
