package xyz.cereshost.utils;

import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;

/**
 * Normalizador robusto para X (features).
 */
public class XNormalizer {

    @Getter
    private float[] medians;

    @Getter
    private float[] iqrs;

    @Setter
    private float minIqr = 1e-6f;

    public void fit(float[][][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("X vacío");
        }

        int samples = X.length;
        int lookback = X[0].length;
        int features = X[0][0].length;
        int total = samples * lookback;

        medians = new float[features];
        iqrs = new float[features];

        // 🔥 cada feature es independiente
        for (int f = 0; f < features; f++) {
            float[] vals = new float[total];
            int idx = 0;

            for (int i = 0; i < samples; i++) {
                for (int t = 0; t < lookback; t++) {
                    vals[idx++] = X[i][t][f];
                }
            }

            // 🚀 sort multithread
            Arrays.parallelSort(vals);

            float median = percentileFromSorted(vals, 50);
            float q1 = percentileFromSorted(vals, 25);
            float q3 = percentileFromSorted(vals, 75);

            float iqr = q3 - q1;
            if (iqr <= 0f) iqr = minIqr;

            medians[f] = median;
            iqrs[f] = iqr;
        }
    }

    public float[][][] transform(float[][][] X) {
        if (medians == null || iqrs == null) {
            throw new IllegalStateException("Llama a fit() antes de transform()");
        }

        int samples = X.length;
        int lookback = X[0].length;
        int features = X[0][0].length;

        float[][][] out = new float[samples][lookback][features];

        for (int i = 0; i < samples; i++) {
            for (int t = 0; t < lookback; t++) {
                for (int f = 0; f < features; f++) {
                    out[i][t][f] = (X[i][t][f] - medians[f]) / iqrs[f];
                }
            }
        }
        return out;
    }

    public float[][][] inverseTransform(float[][][] Xnorm) {
        if (medians == null || iqrs == null) {
            throw new IllegalStateException("Llama a fit() antes de inverseTransform()");
        }

        int samples = Xnorm.length;
        int lookback = Xnorm[0].length;
        int features = Xnorm[0][0].length;

        float[][][] out = new float[samples][lookback][features];

        for (int i = 0; i < samples; i++) {
            for (int t = 0; t < lookback; t++) {
                for (int f = 0; f < features; f++) {
                    out[i][t][f] = Xnorm[i][t][f] * iqrs[f] + medians[f];
                }
            }
        }
        return out;
    }

    // 🔧 helper optimizado para float[]
    public static float percentileFromSorted(float[] sorted, double pct) {
        int n = sorted.length;
        if (n == 0) return 0f;
        if (n == 1) return sorted[0];

        double rank = (pct / 100.0) * (n - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);

        if (lo == hi) return sorted[lo];

        float lw = sorted[lo];
        float hw = sorted[hi];
        double frac = rank - lo;

        return (float) (lw + (hw - lw) * frac);
    }
}
