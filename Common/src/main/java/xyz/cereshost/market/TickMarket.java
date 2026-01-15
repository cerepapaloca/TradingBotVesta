package xyz.cereshost.market;

import lombok.Getter;


@Getter
public class TickMarket {

    public TickMarket(Volumen volume, Depth depth) {
        this.volumen  = volume;
        this.depth = depth;
    }

    private final Volumen volumen;

    private final Depth depth;
}
