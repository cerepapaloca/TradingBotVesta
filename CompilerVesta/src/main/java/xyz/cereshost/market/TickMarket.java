package xyz.cereshost.market;

import lombok.Getter;

import java.util.Collection;
import java.util.HashSet;


@Getter
public class TickMarket {

    public TickMarket(Volumen volume, Depth depth) {
        this.volumen  = volume;
        this.depth = depth;
    }

    private final Volumen volumen;

    private final Depth depth;
}
