package xyz.cereshost.utils;

import org.jetbrains.annotations.NotNull;

import java.util.Arrays;

/**
 * Normalizador para Y en tu esquema:
 * - Y shape esperado: [samples][5] con columnas:
 *   0: TP (regresión)
 *   1: SL (regresión)
 *   2: Long (one-hot)
 *   3: Neutral (one-hot)
 *   4: Short (one-hot)
 *
 * - fit(y): calcula mediana y MAD para TP/SL (columnas 0 y 1)
 * - transform(y): normaliza columnas 0 y 1, deja columnas 2..4 intactas
 * - inverseTransform(yNorm): reconstruye TP/SL a escala original
 *
 * Nota: MAD = median(|x - median|)
 */
public class YNormalizer {

    private float[] medians;
    private float[] mads;
    private float[] mins; // Nuevo: para almacenar los mínimos por columna
    private static final float EPSILON = 1e-8f;
    private int numOutputs;

    public void fit(float[][] y) {
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Los datos y no pueden ser nulos o vacíos");
        }

        numOutputs = y[0].length;
        medians = new float[numOutputs];
        mads = new float[numOutputs];
        mins = new float[numOutputs];
        int rows = y.length;
        for (int col = 0; col < numOutputs; col++) {
            // Solo normalizar las primeras 2 columnas (TP y SL)
            if (col < 2) {
                int size = 0;
                float[] values = new float[rows];
                for (float[] row : y) {
                    if (col < row.length) {
                        values[size++] = row[col];
                    }
                }

                if (size == 0) continue;

                if (size < values.length) {
                    values = Arrays.copyOf(values, size);
                }

                Arrays.parallelSort(values);
                float minVal = values[0];
                float med = median(values);
                float deviation = mad(values, med);

                medians[col] = med;
                mads[col] = Math.max(deviation, EPSILON);
                mins[col] = minVal;
            } else {
                // Columnas one-hot (LONG, NEUTRAL, SHORT) no se normalizan
                medians[col] = 0.0f;
                mads[col] = 1.0f;
                mins[col] = 0.0f;
            }
        }
    }

    public float[][] transform(float[][] y) {
        if (medians == null) throw new IllegalStateException("Normalizador no ajustado");
        float[][] normalized = new float[y.length][numOutputs];
        for (int i = 0; i < y.length; i++) {
            for (int col = 0; col < numOutputs; col++) {
                if (col < 2) {
                    normalized[i][col] = (y[i][col] - mins[col]) / mads[col];
                } else {
                    normalized[i][col] = y[i][col];
                }
            }
        }
        return normalized;
    }

    public float[][] inverseTransform(float[][] yNorm) {
        float[][] original = new float[yNorm.length][yNorm[0].length];
        for (int i = 0; i < yNorm.length; i++) {
            for (int col = 0; col < yNorm[i].length; col++) {
                if (col < 2) {
                    original[i][col] = (yNorm[i][col] * mads[col]) + mins[col];
                } else {
                    original[i][col] = yNorm[i][col];
                }
            }
        }
        return original;
    }

    private float median(float @NotNull [] values) {
        int n = values.length;
        return (n % 2 == 0) ? (values[n/2 - 1] + values[n/2]) / 2.0f : values[n/2];
    }

    private float mad(float @NotNull [] values, float median) {
        float[] medians = new float[values.length];
        int idx = 0;
        for (float v : values) {
            medians[idx++] = Math.abs(v - median);
        }
        Arrays.parallelSort(medians);
        return median(medians);
    }
}