package xyz.cereshost.vesta.core.utils;

import lombok.Getter;
import lombok.Setter;

import java.util.Arrays;
import java.util.Random;

/**
 * Normalizador robusto para X (features).
 */
public class XNormalizer implements Normalizer<float[][][]> {

    @Getter
    private float[] medians;

    @Getter
    private float[] iqrs;

    @Setter
    private float minIqr = 1e-6f;

    @Setter
    private int reservoirSize = 20000;

    private transient float[][] reservoirs;
    private transient long[] seen;
    private transient Random random = new Random(1337);

    @Override
    public void fit(float[][][] X) {
        if (X == null || X.length == 0) {
            throw new IllegalArgumentException("X vacio");
        }

        reservoirs = null;
        seen = null;

        int samples = X.length;
        int lookback = X[0].length;
        int features = X[0][0].length;
        int total = samples * lookback;

        medians = new float[features];
        iqrs = new float[features];

        // cada feature es independiente
        for (int f = 0; f < features; f++) {
            float[] vals = new float[total];
            int idx = 0;

            for (int i = 0; i < samples; i++) {
                for (int t = 0; t < lookback; t++) {
                    vals[idx++] = X[i][t][f];
                }
            }

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

    public void partialFit(float[][][] X) {
        if (X == null || X.length == 0) {
            return;
        }

        int samples = X.length;
        int lookback = X[0].length;
        int features = X[0][0].length;
        ensureReservoirs(features);

        for (float[][] x : X) {
            for (int t = 0; t < lookback; t++) {
                float[] row = x[t];
                for (int f = 0; f < features; f++) {
                    float v = row[f];
                    if (!Float.isFinite(v)) {
                        continue;
                    }
                    long c = ++seen[f];
                    if (c <= reservoirSize) {
                        reservoirs[f][(int) c - 1] = v;
                    } else {
                        long j = nextLong(ensureRandom(), c);
                        if (j < reservoirSize) {
                            reservoirs[f][(int) j] = v;
                        }
                    }
                }
            }
        }
    }

    public void finishFit() {
        if (reservoirs == null || seen == null) {
            throw new IllegalStateException("partialFit() no fue llamado");
        }

        int features = reservoirs.length;
        medians = new float[features];
        iqrs = new float[features];

        for (int f = 0; f < features; f++) {
            int size = (int) Math.min((long) reservoirSize, seen[f]);
            if (size <= 0) {
                medians[f] = 0f;
                iqrs[f] = 1f;
                continue;
            }
            float[] vals = Arrays.copyOf(reservoirs[f], size);
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

    @Override
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

    @Override
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

    // helper optimizado para float[]
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

    private void ensureReservoirs(int features) {
        if (reservoirs != null && reservoirs.length == features) {
            return;
        }
        if (reservoirSize < 1000) {
            reservoirSize = 1000;
        }
        reservoirs = new float[features][reservoirSize];
        seen = new long[features];
    }

    private Random ensureRandom() {
        if (random == null) {
            random = new Random(1337);
        }
        return random;
    }

    private static long nextLong(Random rnd, long bound) {
        if (bound <= 0) {
            throw new IllegalArgumentException("bound must be positive");
        }
        long r = rnd.nextLong();
        long m = bound - 1;
        if ((bound & m) == 0L) {
            return r & m;
        }
        long u = r >>> 1;
        while (u + m - (u % bound) < 0L) {
            u = rnd.nextLong() >>> 1;
        }
        return u % bound;
    }
}
