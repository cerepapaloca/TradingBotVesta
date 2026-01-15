package xyz.cereshost.common.market;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;

public class Market {

    public Market(@NotNull String symbol) {
        this.symbol = symbol;
        this.trades = new ArrayDeque<>();
        this.tickMarkers = new ArrayDeque<>();
    }

    @NotNull
    @Getter
    private final String symbol;
    @Getter
    private final Deque<Trade> trades;
    @Getter
    private final Deque<TickMarket> tickMarkers;


    public void concat(@NotNull Market market) {
        if (!this.symbol.equals(market.symbol)) {
            throw new IllegalArgumentException("Symbols don't match");
        }
        this.trades.addAll(market.trades);
        this.tickMarkers.addAll(market.tickMarkers);
    }

    public void add(Collection<Trade> trade) {
        this.trades.addAll(trade);
    }

    public void add(TickMarket tickMarker) {
        this.tickMarkers.addLast(tickMarker);
    }

}
