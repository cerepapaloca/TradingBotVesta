package xyz.cereshost.vesta.core.utils;

import lombok.experimental.UtilityClass;
import xyz.cereshost.vesta.common.market.Candle;

import java.util.ArrayList;
import java.util.List;

@UtilityClass
public class StrategyUtils {

    public record BosChochSignal(
            boolean valid,
            boolean bos,
            boolean choch,
            int direction,          // 1 long, -1 short, 0 neutral
            int previousStructure,  // 1 bullish, -1 bearish, 0 neutral
            int pivotIndex,
            double pivotPrice
    ) {
        public static BosChochSignal none() {
            return new BosChochSignal(false, false, false, 0, 0, -1, Double.NaN);
        }
    }

    public static boolean isPeek(List<Candle> visibleCandles, int lookback) {
        return isHigh(visibleCandles, lookback) || isLow(visibleCandles, lookback);
    }

    /**
     * Indica si la vela actual está en el máximo de la última hora (60 velas de 1m).
     */
    public static boolean isHigh(List<Candle> candles, int lookback) {
        if (candles == null || candles.isEmpty()) {
            return false;
        }
        double lastPrice = candles.getLast().highBody();
        for (int i = candles.size() - 2; i >= lookback; i--) {
            if (lastPrice < candles.get(i).highBody()) return false;
        }
        return true;
    }

    /**
     * Indica si la vela actual está en el mínimo de la última hora (60 velas de 1m).
     */
    public static boolean isLow(List<Candle> candles, int lookback) {
        if (candles == null || candles.isEmpty()) {
            return false;
        }
        double lastPrice = candles.getLast().lowBody();
        for (int i = candles.size() - 2; i >= candles.size() - lookback; i--) {
            if (lastPrice > candles.get(i).lowBody()) return false;
        }
        return true;
    }

    public static BosChochSignal detectBosChoch(
            List<Candle> candles,
            int pivotLeft,
            int pivotRight,
            int lookback,
            double minBreakPercent
    ) {
        if (candles == null || candles.size() < (pivotLeft + pivotRight + 6)) {
            return BosChochSignal.none();
        }

        int last = candles.size() - 1;
        int from = Math.max(pivotLeft, last - lookback - pivotRight);
        int to = last - pivotRight;
        if (to <= from) {
            return BosChochSignal.none();
        }

        List<Integer> highs = new ArrayList<>();
        List<Integer> lows = new ArrayList<>();
        for (int i = from; i <= to; i++) {
            if (isPivotHigh(candles, i, pivotLeft, pivotRight)) highs.add(i);
            if (isPivotLow(candles, i, pivotLeft, pivotRight)) lows.add(i);
        }

        if (highs.isEmpty() || lows.isEmpty()) {
            return BosChochSignal.none();
        }

        int lastHighIdx = highs.getLast();
        int lastLowIdx = lows.getLast();
        int prevHighIdx = highs.size() > 1 ? highs.get(highs.size() - 2) : -1;
        int prevLowIdx = lows.size() > 1 ? lows.get(lows.size() - 2) : -1;

        int previousStructure = resolveStructure(candles, prevHighIdx, lastHighIdx, prevLowIdx, lastLowIdx);

        double close = candles.getLast().close();
        if (!Double.isFinite(close) || close <= 0.0) {
            return BosChochSignal.none();
        }

        double breakFactor = Math.max(0.0, minBreakPercent) / 100.0;
        double upLevel = candles.get(lastHighIdx).high() * (1.0 + breakFactor);
        double downLevel = candles.get(lastLowIdx).low() * (1.0 - breakFactor);

        boolean breakUp = close > upLevel;
        boolean breakDown = close < downLevel;
        if (breakUp == breakDown) {
            System.out.println(close + " " + upLevel + "  " + downLevel + " | " + breakUp + " " + breakDown);
            return BosChochSignal.none();
        }

        int direction = breakUp ? 1 : -1;
        int pivotIndex = breakUp ? lastHighIdx : lastLowIdx;
        double pivotPrice = breakUp ? candles.get(lastHighIdx).high() : candles.get(lastLowIdx).low();
        boolean choch = previousStructure != 0 && direction != previousStructure;
        boolean bos = !choch;

        return new BosChochSignal(true, bos, choch, direction, previousStructure, pivotIndex, pivotPrice);
    }

    public static boolean isPivotHigh(List<Candle> candles, int idx, int pivotLeft, int pivotRight) {
        if (candles == null || idx < 0 || idx >= candles.size()) return false;
        if (idx - pivotLeft < 0 || idx + pivotRight >= candles.size()) return false;

        double pivot = candles.get(idx).high();
        if (!Double.isFinite(pivot)) return false;

        for (int i = idx - pivotLeft; i <= idx + pivotRight; i++) {
            if (i == idx) continue;
            double other = candles.get(i).high();
            if (!Double.isFinite(other) || other >= pivot) return false;
        }
        return true;
    }

    public static boolean isPivotLow(List<Candle> candles, int idx, int pivotLeft, int pivotRight) {
        if (candles == null || idx < 0 || idx >= candles.size()) return false;
        if (idx - pivotLeft < 0 || idx + pivotRight >= candles.size()) return false;

        double pivot = candles.get(idx).low();
        if (!Double.isFinite(pivot)) return false;

        for (int i = idx - pivotLeft; i <= idx + pivotRight; i++) {
            if (i == idx) continue;
            double other = candles.get(i).low();
            if (!Double.isFinite(other) || other <= pivot) return false;
        }
        return true;
    }

    private static int resolveStructure(List<Candle> candles, int prevHighIdx, int lastHighIdx, int prevLowIdx, int lastLowIdx) {
        if (prevHighIdx < 0 || prevLowIdx < 0 || lastHighIdx < 0 || lastLowIdx < 0) return 0;

        double prevHigh = candles.get(prevHighIdx).high();
        double lastHigh = candles.get(lastHighIdx).high();
        double prevLow = candles.get(prevLowIdx).low();
        double lastLow = candles.get(lastLowIdx).low();

        if (!Double.isFinite(prevHigh) || !Double.isFinite(lastHigh) || !Double.isFinite(prevLow) || !Double.isFinite(lastLow)) {
            return 0;
        }

        if (lastHigh > prevHigh && lastLow > prevLow) return 1;
        if (lastHigh < prevHigh && lastLow < prevLow) return -1;
        return 0;
    }

}
