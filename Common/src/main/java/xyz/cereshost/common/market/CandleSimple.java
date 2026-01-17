package xyz.cereshost.common.market;

import java.util.Objects;

public record CandleSimple(long openTime, double open, double high, double low, double close, Volumen volumen) {
    @Override
    public int hashCode() {
        return Objects.hash(openTime);
    }
}
