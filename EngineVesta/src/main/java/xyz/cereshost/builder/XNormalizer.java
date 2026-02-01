package xyz.cereshost.builder;

import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Normalizador robusto para X (features).
 * - fit(X): calcula mediana, Q1, Q3 por feature (a lo largo de samples × time)
 * - transform(X): (x - median) / iqr
 * - inverseTransform(Xnorm): x = xnorm * iqr + median
 *
 * NOTA: No aplica clipping para mantener invertibilidad.
 */
public class XNormalizer {

    // Getters
    @Getter
    private float[] medians;   // mediana por feature
    @Getter
    private float[] iqrs;      // IQR (Q3-Q1) por feature (no 0)
    @Setter
    private float minIqr = 1e-6f; // si IQR calculada es 0, sustituir por esto

    public void fit(float[][][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("X vacío");
        }
        int features = X[0][0].length;

        medians = new float[features];
        iqrs = new float[features];

        // Recolectar por feature todos los valores
        for (int f = 0; f < features; f++) {
            List<Float> vals = new ArrayList<>(X.length * X[0].length);
            for (int i = 0; i < X.length; i++) {
                for (int t = 0; t < X[i].length; t++) {
                    vals.add(X[i][t][f]);
                }
            }
            Collections.sort(vals);
            medians[f] = percentileFromSorted(vals, 50);
            float q1 = percentileFromSorted(vals, 25);
            float q3 = percentileFromSorted(vals, 75);
            float iqr = q3 - q1;
            if (iqr <= 0f) iqr = minIqr;
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

    // Helpers

    public static float percentileFromSorted(List<Float> sorted, double pct) {
        if (sorted == null || sorted.isEmpty()) return 0f;
        final int n = sorted.size();
        if (n == 1) return sorted.get(0);
        double rank = (pct / 100.0) * (n - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) return sorted.get(lo);
        float lw = sorted.get(lo);
        float hw = sorted.get(hi);
        double frac = rank - lo;
        return (float) (lw + (hw - lw) * frac);
    }

}
