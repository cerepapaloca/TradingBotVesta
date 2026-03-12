package xyz.cereshost.vesta.core.strategys;

import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.util.Arrays;
import java.util.List;

public class BetaStrategy implements TradingStrategy {

    private static final MaType MA_TYPE = MaType.EMA;
    private static final int MA_LEN = 100;
    private static final Source MA_SOURCE = Source.CLOSE;

    private static final MaType ATR_TYPE = MaType.RMA;
    private static final double ATR_MULT = 3;
    private static final int ATR_LEN = 7;

    private static final int CONFIRM_BARS = 2;
    private static final ConfirmSource CONFIRM_SOURCE = ConfirmSource.CLOSE;

    private static final EntryMethod ENTRY_METHOD = EntryMethod.MA_CROSS;
    private static final double TP_PERCENT = 3.5;
    private static final double SL_PERCENT = 1.25;

    private static final int LEVERAGE = 3;
    private static final double ORDER_BALANCE_FRACTION = 0.2;
    private static final double MIN_ORDER_NOTIONAL = 5.2;

    private int trend = 0;
    private int breakPoint = 0;
    private boolean crossUsed = false;
    private int uptrendBars = 0;
    private int downtrendBars = 0;

    @Override
    public void executeStrategy(PredictionEngine.PredictionResult pred, List<Candle> visibleCandles, TradingManager operations) {
        if (visibleCandles == null || visibleCandles.size() < Math.max(MA_LEN, ATR_LEN) + 3) {
            return;
        }

        int lastIndex = visibleCandles.size() - 1;
        int prevIndex = lastIndex - 1;

        Candle curr = visibleCandles.get(lastIndex);
        Candle prev = visibleCandles.get(prevIndex);

        double maVal = computeMa(visibleCandles, prevIndex, MA_LEN, MA_TYPE, MA_SOURCE);
        double atrVal = computeAtr(visibleCandles, prevIndex, ATR_LEN, ATR_TYPE);
        if (!isFinite(maVal) || !isFinite(atrVal)) {
            return;
        }

        double atrUpper = maVal + (atrVal * ATR_MULT);
        double atrLower = maVal - (atrVal * ATR_MULT);

        double uptrendPrice = CONFIRM_SOURCE == ConfirmSource.CLOSE ? curr.close() : curr.high();
        double downtrendPrice = CONFIRM_SOURCE == ConfirmSource.CLOSE ? curr.close() : curr.low();

        boolean isUptrendRaw = uptrendPrice > atrUpper;
        boolean isDowntrendRaw = downtrendPrice < atrLower;

        uptrendBars = isUptrendRaw ? uptrendBars + 1 : 0;
        downtrendBars = isDowntrendRaw ? downtrendBars + 1 : 0;

        boolean isUptrendConfirmed = uptrendBars >= CONFIRM_BARS;
        boolean isDowntrendConfirmed = downtrendBars >= CONFIRM_BARS;

        if (isUptrendConfirmed && trend != 1) {
            trend = 1;
            breakPoint = 1;
            crossUsed = false;
        }
        if (isDowntrendConfirmed && trend != -1) {
            trend = -1;
            breakPoint = -1;
            crossUsed = false;
        }

        if (operations.hasOpenOperation()) {
            return;
        }

        if (ENTRY_METHOD == EntryMethod.BREAKOUT) {
            if (breakPoint == 1) {
                open(operations, DireccionOperation.LONG);
            } else if (breakPoint == -1) {
                open(operations, DireccionOperation.SHORT);
            }
        } else {
            if (crossUsed || prevIndex - 1 < 0) {
                return;
            }

            double maPrev = computeMa(visibleCandles, prevIndex - 1, MA_LEN, MA_TYPE, MA_SOURCE);
            if (!isFinite(maPrev)) {
                return;
            }

            boolean crossUnder = prev.close() >= maPrev && curr.close() < maVal;
            boolean crossOver = prev.close() <= maPrev && curr.close() > maVal;

            if (trend == 1 && crossUnder) {
                open(operations, DireccionOperation.LONG);
                crossUsed = true;
            } else if (trend == -1 && crossOver) {
                open(operations, DireccionOperation.SHORT);
                crossUsed = true;
            }
        }

    }

    @Override
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {

    }

    private void open(TradingManager operations, DireccionOperation direction) {
        double availableBalance = operations.getAvailableBalance();
        if (!isFinite(availableBalance) || availableBalance <= 0.0) {
            return;
        }

        double minMarginUsdt = MIN_ORDER_NOTIONAL / Math.max(LEVERAGE, 1);
        double amountUsdt = Math.max(availableBalance * ORDER_BALANCE_FRACTION, minMarginUsdt);
        amountUsdt = Math.min(amountUsdt, availableBalance);
        if (amountUsdt * LEVERAGE < MIN_ORDER_NOTIONAL) {
            operations.log("Balance insuficiente para abrir BetaStrategy");
            return;
        }

        operations.open(TP_PERCENT, SL_PERCENT, direction, amountUsdt, LEVERAGE);
    }

    private static double computeAtr(List<Candle> candles, int endIndex, int length, MaType atrType) {
        if (candles == null || endIndex <= 0 || length <= 0 || endIndex >= candles.size()) {
            return Double.NaN;
        }

        double[] tr = new double[endIndex + 1];
        for (int i = 0; i <= endIndex; i++) {
            Candle curr = candles.get(i);
            Candle prev = i > 0 ? candles.get(i - 1) : curr;
            double highLow = curr.high() - curr.low();
            double highClose = Math.abs(curr.high() - prev.close());
            double lowClose = Math.abs(curr.low() - prev.close());
            tr[i] = Math.max(highLow, Math.max(highClose, lowClose));
        }

        MaType normalized = switch (atrType) {
            case EMA, SMA, WMA, RMA -> atrType;
            default -> MaType.RMA;
        };
        return computeMaFromSeries(tr, endIndex, length, normalized);
    }

    private static double computeMa(List<Candle> candles, int endIndex, int length, MaType type, Source source) {
        if (candles == null || endIndex < 0 || endIndex >= candles.size() || length <= 0) {
            return Double.NaN;
        }
        if (endIndex < length - 1) {
            return Double.NaN;
        }

        return switch (type) {
            case SMA -> computeSma(candles, endIndex, length, source);
            case EMA -> computeEma(candles, endIndex, length, 2.0 / (length + 1.0), source);
            case WMA -> computeWma(candles, endIndex, length, source);
            case RMA -> computeEma(candles, endIndex, length, 1.0 / length, source);
            case VWMA -> computeVwma(candles, endIndex, length, source);
            case HMA -> computeHma(candles, endIndex, length, source);
            case TEMA -> computeTema(candles, endIndex, length, source);
        };
    }

    private static double computeSma(List<Candle> candles, int endIndex, int length, Source source) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        double sum = 0.0;
        for (int i = endIndex - length + 1; i <= endIndex; i++) {
            double v = getSourceValue(candles.get(i), source);
            if (!isFinite(v)) {
                return Double.NaN;
            }
            sum += v;
        }
        return sum / length;
    }

    private static double computeEma(List<Candle> candles, int endIndex, int length, double alpha, Source source) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        double[] values = buildSourceArray(candles, endIndex, source);
        return computeEmaFromSeries(values, endIndex, length, alpha);
    }

    private static double computeWma(List<Candle> candles, int endIndex, int length, Source source) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        double weightedSum = 0.0;
        double weightTotal = 0.0;
        int weight = 1;
        for (int i = endIndex - length + 1; i <= endIndex; i++) {
            double v = getSourceValue(candles.get(i), source);
            if (!isFinite(v)) {
                return Double.NaN;
            }
            weightedSum += v * weight;
            weightTotal += weight;
            weight++;
        }
        return weightTotal == 0.0 ? Double.NaN : (weightedSum / weightTotal);
    }

    private static double computeVwma(List<Candle> candles, int endIndex, int length, Source source) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        double numerator = 0.0;
        double denominator = 0.0;
        for (int i = endIndex - length + 1; i <= endIndex; i++) {
            Candle candle = candles.get(i);
            double v = getSourceValue(candle, source);
            double vol = candle.volumeBase();
            if (!isFinite(v) || !isFinite(vol)) {
                return Double.NaN;
            }
            numerator += v * vol;
            denominator += vol;
        }
        return denominator == 0.0 ? Double.NaN : (numerator / denominator);
    }

    private static double computeHma(List<Candle> candles, int endIndex, int length, Source source) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        int half = Math.max(1, length / 2);
        int sqrtLen = Math.max(1, (int) Math.round(Math.sqrt(length)));
        if (endIndex < sqrtLen - 1) {
            return Double.NaN;
        }

        double weightedSum = 0.0;
        double weightTotal = 0.0;
        int weight = 1;
        for (int i = endIndex - sqrtLen + 1; i <= endIndex; i++) {
            double wmaShort = computeWma(candles, i, half, source);
            double wmaLong = computeWma(candles, i, length, source);
            if (!isFinite(wmaShort) || !isFinite(wmaLong)) {
                return Double.NaN;
            }
            double diff = (2.0 * wmaShort) - wmaLong;
            weightedSum += diff * weight;
            weightTotal += weight;
            weight++;
        }
        return weightTotal == 0.0 ? Double.NaN : (weightedSum / weightTotal);
    }

    private static double computeTema(List<Candle> candles, int endIndex, int length, Source source) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        double[] values = buildSourceArray(candles, endIndex, source);
        double[] ema1 = computeEmaSeries(values, length, 2.0 / (length + 1.0));
        double[] ema2 = computeEmaSeries(ema1, length, 2.0 / (length + 1.0));
        double[] ema3 = computeEmaSeries(ema2, length, 2.0 / (length + 1.0));

        if (!isFinite(ema1[endIndex]) || !isFinite(ema2[endIndex]) || !isFinite(ema3[endIndex])) {
            return Double.NaN;
        }
        return (3.0 * ema1[endIndex]) - (3.0 * ema2[endIndex]) + ema3[endIndex];
    }

    private static double[] buildSourceArray(List<Candle> candles, int endIndex, Source source) {
        double[] values = new double[endIndex + 1];
        for (int i = 0; i <= endIndex; i++) {
            values[i] = getSourceValue(candles.get(i), source);
        }
        return values;
    }

    private static double[] computeEmaSeries(double[] values, int length, double alpha) {
        int n = values.length;
        double[] ema = new double[n];
        Arrays.fill(ema, Double.NaN);
        if (n < length || length <= 0) {
            return ema;
        }

        int start = 0;
        while (start < n && !isFinite(values[start])) {
            start++;
        }
        if (start + length > n) {
            return ema;
        }

        double sum = 0.0;
        for (int i = start; i < start + length; i++) {
            double v = values[i];
            if (!isFinite(v)) {
                return ema;
            }
            sum += v;
        }

        int seedIndex = start + length - 1;
        ema[seedIndex] = sum / length;

        for (int i = seedIndex + 1; i < n; i++) {
            double v = values[i];
            if (!isFinite(v) || !isFinite(ema[i - 1])) {
                ema[i] = Double.NaN;
            } else {
                ema[i] = alpha * (v - ema[i - 1]) + ema[i - 1];
            }
        }
        return ema;
    }

    private static double computeMaFromSeries(double[] values, int endIndex, int length, MaType type) {
        if (endIndex < length - 1 || endIndex >= values.length) {
            return Double.NaN;
        }

        return switch (type) {
            case SMA -> computeSmaFromSeries(values, endIndex, length);
            case EMA -> computeEmaFromSeries(values, endIndex, length, 2.0 / (length + 1.0));
            case WMA -> computeWmaFromSeries(values, endIndex, length);
            case RMA -> computeEmaFromSeries(values, endIndex, length, 1.0 / length);
            default -> computeEmaFromSeries(values, endIndex, length, 1.0 / length);
        };
    }

    private static double computeSmaFromSeries(double[] values, int endIndex, int length) {
        double sum = 0.0;
        for (int i = endIndex - length + 1; i <= endIndex; i++) {
            double v = values[i];
            if (!isFinite(v)) {
                return Double.NaN;
            }
            sum += v;
        }
        return sum / length;
    }

    private static double computeEmaFromSeries(double[] values, int endIndex, int length, double alpha) {
        if (endIndex < length - 1) {
            return Double.NaN;
        }
        double sum = 0.0;
        for (int i = 0; i < length; i++) {
            double v = values[i];
            if (!isFinite(v)) {
                return Double.NaN;
            }
            sum += v;
        }
        double ema = sum / length;
        for (int i = length; i <= endIndex; i++) {
            double v = values[i];
            if (!isFinite(v)) {
                return Double.NaN;
            }
            ema = alpha * (v - ema) + ema;
        }
        return ema;
    }

    private static double computeWmaFromSeries(double[] values, int endIndex, int length) {
        double weightedSum = 0.0;
        double weightTotal = 0.0;
        int weight = 1;
        for (int i = endIndex - length + 1; i <= endIndex; i++) {
            double v = values[i];
            if (!isFinite(v)) {
                return Double.NaN;
            }
            weightedSum += v * weight;
            weightTotal += weight;
            weight++;
        }
        return weightTotal == 0.0 ? Double.NaN : (weightedSum / weightTotal);
    }

    private static double getSourceValue(Candle candle, Source source) {
        return switch (source) {
            case CLOSE -> candle.close();
        };
    }

    private static boolean isFinite(double value) {
        return Double.isFinite(value);
    }

    private enum MaType {
        EMA,
        SMA,
        WMA,
        HMA,
        VWMA,
        RMA,
        TEMA
    }

    private enum EntryMethod {
        BREAKOUT,
        MA_CROSS
    }

    private enum ConfirmSource {
        CLOSE,
        LOW_HIGH
    }

    private enum Source {
        CLOSE
    }
}
