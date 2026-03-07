package xyz.cereshost.vesta.core;

import lombok.experimental.UtilityClass;
import xyz.cereshost.vesta.common.market.CandleSimple;

import java.util.*;

@UtilityClass
public class FinancialCalculation {

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
