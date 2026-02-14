package xyz.cereshost.utils;

/**
 * Normalizador para Y en tu esquema:
 * - Y shape esperado: [samples][5] con columnas:
 *   0: TP (regresión)
 *   1: SL (regresión)
 *   2: Long (one-hot)
 *   3: Neutral (one-hot)
 *   4: Short (one-hot)
 *
 * - fit(y): calcula media/desviaciÃ³n en log1p(TP/SL)
 * - transform(y): aplica log1p y z-score a columnas 0 y 1, deja columnas 2..4 intactas
 * - inverseTransform(yNorm): revierte z-score y log1p
 */
public class YNormalizer implements Normalizer<float[][]> {

    private float[] means;
    private float[] stds;
    private static final float EPSILON = 1e-8f;
    private int numOutputs;

    @Override
    public void fit(float[][] y) {
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Los datos y no pueden ser nulos o vacíos");
        }

        numOutputs = y[0].length;
        means = new float[numOutputs];
        stds = new float[numOutputs];
        int rows = y.length;
        for (int col = 0; col < numOutputs; col++) {
            // Solo normalizar las primeras 2 columnas (TP y SL)
            if (col < 2) {
                double sum = 0.0;
                double sumSq = 0.0;
                int count = 0;
                for (float[] row : y) {
                    if (col < row.length) {
                        float v = row[col];
                        if (Float.isNaN(v) || Float.isInfinite(v)) continue;
                        if (v < 0f) v = 0f;
                        double lv = Math.log1p(v);
                        sum += lv;
                        sumSq += lv * lv;
                        count++;
                    }
                }

                if (count == 0) {
                    means[col] = 0f;
                    stds[col] = 1f;
                    continue;
                }

                double mean = sum / count;
                double variance = (sumSq / count) - (mean * mean);
                double std = Math.sqrt(Math.max(variance, EPSILON));

                means[col] = (float) mean;
                stds[col] = (float) std;
            } else {
                // Columnas one-hot (LONG, NEUTRAL, SHORT) no se normalizan
                means[col] = 0.0f;
                stds[col] = 1.0f;
            }
        }
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
