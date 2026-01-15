package xyz.cereshost.market;

import lombok.Getter;

import java.util.ArrayDeque;
import java.util.Collection;
import java.util.Deque;

@Getter
public class Depth {

    public Depth(long date, Collection<OrderLevel> bids, Collection<OrderLevel> asks) {
        this.date = date;
        this.bids = new ArrayDeque<>(bids);
        this.asks = new ArrayDeque<>(asks);
    }

    private final long date;

    private final Deque<OrderLevel> bids;
    private final Deque<OrderLevel> asks;

    public record OrderLevel(double price, double qty) {}
}
