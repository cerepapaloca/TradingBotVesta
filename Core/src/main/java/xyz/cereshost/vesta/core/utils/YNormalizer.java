package xyz.cereshost.vesta.core.utils;

/**
 * Normalizador para Y en tu esquema:
 * - Y shape esperado: [samples][5] con columnas:
 *   0: TP (regresion)
 *   1: SL (regresion)
 *   2: Long (one-hot)
 *   3: Neutral (one-hot)
 *   4: Short (one-hot)
 *
 * - fit(y): calcula media/desviacion en log1p(TP/SL)
 * - transform(y): aplica log1p y z-score a columnas 0 y 1, deja columnas 2..4 intactas
 * - inverseTransform(yNorm): revierte z-score y log1p
 */
public class YNormalizer implements Normalizer<float[][]> {

    private float[] means;
    private float[] stds;
    private static final float EPSILON = 1e-8f;
    private int numOutputs;

    private double[] meanAcc;
    private double[] m2Acc;
    private long[] countAcc;

    @Override
    public void fit(float[][] y) {
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Los datos y no pueden ser nulos o vacios");
        }
        resetAcc(y[0].length);
        partialFit(y);
        finishFit();
    }

    public void partialFit(float[][] y) {
        if (y == null || y.length == 0) {
            return;
        }
        if (meanAcc == null || meanAcc.length != y[0].length) {
            resetAcc(y[0].length);
        }
        for (float[] row : y) {
            for (int col = 0; col < numOutputs; col++) {
                if (col >= 2 || col >= row.length) {
                    continue;
                }
                float v = row[col];
                if (!Float.isFinite(v)) {
                    continue;
                }
                if (v < 0f) v = 0f;
                double lv = Math.log1p(v);
                long c = ++countAcc[col];
                double delta = lv - meanAcc[col];
                meanAcc[col] += delta / c;
                double delta2 = lv - meanAcc[col];
                m2Acc[col] += delta * delta2;
            }
        }
    }

    public void finishFit() {
        means = new float[numOutputs];
        stds = new float[numOutputs];
        for (int col = 0; col < numOutputs; col++) {
            if (col < 2) {
                long c = countAcc[col];
                if (c == 0) {
                    means[col] = 0f;
                    stds[col] = 1f;
                    continue;
                }
                double mean = meanAcc[col];
                double variance = (m2Acc[col] / c);
                double std = Math.sqrt(Math.max(variance, EPSILON));
                means[col] = (float) mean;
                stds[col] = (float) std;
            } else {
                means[col] = 0.0f;
                stds[col] = 1.0f;
            }
        }
    }

    private void resetAcc(int outputs) {
        numOutputs = outputs;
        meanAcc = new double[outputs];
        m2Acc = new double[outputs];
        countAcc = new long[outputs];
    }

    @Override
    public float[][] transform(float[][] y) {
        if (means == null || stds == null) throw new IllegalStateException("Normalizador no ajustado");
        float[][] normalized = new float[y.length][numOutputs];
        for (int i = 0; i < y.length; i++) {
            for (int col = 0; col < numOutputs; col++) {
                if (col < 2) {
                    float raw = y[i][col];
                    if (!Float.isFinite(raw) || raw < 0f) raw = 0f;
                    double lv = Math.log1p(raw);
                    float z = (float) ((lv - means[col]) / stds[col]);
                    normalized[i][col] = Float.isFinite(z) ? z : 0f;
                } else {
                    normalized[i][col] = y[i][col];
                }
            }
        }
        return normalized;
    }

    @Override
    public float[][] inverseTransform(float[][] yNorm) {
        float[][] original = new float[yNorm.length][yNorm[0].length];
        for (int i = 0; i < yNorm.length; i++) {
            for (int col = 0; col < yNorm[i].length; col++) {
                if (col < 2) {
                    double lv = (yNorm[i][col] * stds[col]) + means[col];
                    double v = Math.expm1(lv);
                    original[i][col] = Double.isFinite(v) && v > 0.0 ? (float) v : 0f;
                } else {
                    original[i][col] = yNorm[i][col];
                }
            }
        }
        return original;
    }
}
