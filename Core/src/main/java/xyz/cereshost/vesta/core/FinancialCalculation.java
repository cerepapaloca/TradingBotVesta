package xyz.cereshost.vesta.core;

import lombok.experimental.UtilityClass;
import xyz.cereshost.vesta.common.market.CandleSimple;

import java.util.*;

@UtilityClass
public class FinancialCalculation {

    /**
     * Calcula la EMA (Exponential Moving Average) alineada con los precios.
     * Devuelve un array de la misma longitud que prices, con Double.NaN
     * para los índices previos al primer valor calculable.
     */
    public static double[] computeEMA(List<Double> prices, int period) {
        int n = prices.size();
        double[] ema = new double[n];
        Arrays.fill(ema, Double.NaN);
        if (n == 0 || period <= 0 || n < period) return ema;

        // alpha = 2 / (period + 1)
        double alpha = 2.0 / (period + 1.0);

        // primer EMA = SMA del primer "period" valores (en índice period-1)
        double sum = 0.0;
        for (int i = 0; i < period; i++) sum += prices.get(i);
        double prevEma = sum / period;
        ema[period - 1] = prevEma;

        // después, EMA recursiva
        for (int i = period; i < n; i++) {
            double price = prices.get(i);
            prevEma = alpha * (price - prevEma) + prevEma;
            ema[i] = prevEma;
        }
        return ema;
    }

    /**
     * Calcula MACD, Signal y Histogram.
     * shortPeriod: e.g. 12
     * longPeriod: e.g. 26
     * signalPeriod: e.g. 9
     *
     * Devuelve MACDResult con arrays alineados (Double.NaN donde no hay valor).
     */
    public static MACDResult computeMACD(List<Double> closes, int shortPeriod, int longPeriod, int signalPeriod) {
        int n = closes.size();
        double[] macd = new double[n];
        double[] signal = new double[n];
        double[] hist = new double[n];
        Arrays.fill(macd, Double.NaN);
        Arrays.fill(signal, Double.NaN);
        Arrays.fill(hist, Double.NaN);

        if (n == 0 || shortPeriod <= 0 || longPeriod <= shortPeriod) {
            return new MACDResult(macd, signal, hist);
        }

        // EMAs
        double[] emaShort = computeEMA(closes, shortPeriod);
        double[] emaLong = computeEMA(closes, longPeriod);

        // MACD line = EMA_short - EMA_long (válido cuando ambos EMA no son NaN)
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(emaShort[i]) && !Double.isNaN(emaLong[i])) {
                macd[i] = emaShort[i] - emaLong[i];
            } else {
                macd[i] = Double.NaN;
            }
        }

        // Para calcular la signal line (EMA del MACD), necesitamos lista de MACD válidos.
        // Pero para mantener alineamiento, aplicamos computeEMA sobre la lista completa
        // transformando NaN a 0 hasta que haya valores; sin embargo la forma correcta:
        // crear lista de macdValid (sólo donde macd != NaN) y luego reinsertar alineado.
        List<Double> macdValid = new ArrayList<>();
        List<Integer> macdIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(macd[i])) {
                macdValid.add(macd[i]);
                macdIdx.add(i);
            }
        }

        if (macdValid.size() >= signalPeriod) {
            double[] signalCompact = computeEMA(macdValid, signalPeriod);
            // mapear de vuelta a array completo
            for (int k = 0; k < signalCompact.length; k++) {
                int idx = macdIdx.get(k);
                signal[idx] = signalCompact[k];
                if (!Double.isNaN(macd[idx]) && !Double.isNaN(signal[idx])) {
                    hist[idx] = macd[idx] - signal[idx];
                } else {
                    hist[idx] = Double.NaN;
                }
            }
        }
        // Si no hay suficientes valores para signal, quedan NaN.

        return new MACDResult(macd, signal, hist);
    }

    public record MACDResult(double[] macd, double[] signal, double[] histogram) {}


    /**
     * Calcula rolling mean y std (Welford o ventana simple) sobre el volumen base.
     * Returns double[][] where [0] = means, [1] = stds (same length as candles).
     * For the first <window elements we fill with the first computed mean/std or 0.
     */
    public static double[][] computeRollingMeanStd(List<CandleSimple> candles, int window) {
        int n = candles.size();
        double[] means = new double[n];
        double[] stds = new double[n];
        if (n == 0) return new double[][]{means, stds};

        Deque<Double> windowVals = new ArrayDeque<>(window);
        double sum = 0;
        double sumSq = 0;

        for (int i = 0; i < n; i++) {
            double v = candles.get(i).volumen().baseVolume();
            windowVals.addLast(v);
            sum += v;
            sumSq += v * v;
            if (windowVals.size() > window) {
                double old = windowVals.removeFirst();
                sum -= old;
                sumSq -= old * old;
            }
            int k = windowVals.size();
            double mean = (k > 0) ? sum / k : 0;
            double variance = (k > 1) ? Math.max(0, (sumSq - (sum * sum) / k) / (k - 1)) : 0;
            double std = Math.sqrt(variance);
            means[i] = mean;
            stds[i] = std;
        }
        return new double[][]{means, stds};
    }

    public static Map<String, double[]> computeVolumeNormalizations(List<CandleSimple> candles, int window, List<Double> atrList) {
        int n = candles.size();
        double[][] meanStd = computeRollingMeanStd(candles, window);
        double[] means = meanStd[0];
        double[] stds = meanStd[1];

        double[] ratio = new double[n];
        double[] zscore = new double[n];
        double[] perAtr = new double[n];

        for (int i = 0; i < n; i++) {
            double v = candles.get(i).volumen().baseVolume();
            double mean = means[i];
            double std = stds[i];
            // ratio to mean (avoid divide by zero)
            ratio[i] = (mean > 0) ? v / mean : 0.0;
            // z-score (if std 0 use 0), clip to [-3,3]
            double z = (std > 0) ? (v - mean) / std : 0.0;
            if (Double.isFinite(z)) {
                z = Math.max(-3.0, Math.min(3.0, z));
            } else {
                z = 0.0;
            }
            zscore[i] = z;
            // volume per ATR (ATR may be 0)
            double atr = (atrList != null && i < atrList.size()) ? atrList.get(i) : 0.0;
            perAtr[i] = (atr > 0) ? v / atr : 0.0;
        }

        Map<String, double[]> map = new HashMap<>();
        map.put("ratio", ratio);
        map.put("zscore", zscore);
        map.put("perAtr", perAtr);
        return map;
    }
}
