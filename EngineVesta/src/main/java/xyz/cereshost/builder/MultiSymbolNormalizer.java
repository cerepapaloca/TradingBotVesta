package xyz.cereshost.builder;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Normalizador especializado para múltiples símbolos
 * Maneja diferentes escalas de precios entre símbolos
 * AHORA solo para un valor (close)
 */
@Getter
public class MultiSymbolNormalizer {
    // Métodos getter para usar las estadísticas en otros lugares
    private float[] medians;
    private float[] mads; // Median Absolute Deviation
    private static final float EPSILON = 1e-8f;
    private int numOutputs;

    public void fit(float[][] y) {
        if (y == null || y.length == 0) {
            throw new IllegalArgumentException("Los datos y no pueden ser nulos o vacíos");
        }

        numOutputs = y[0].length;
        medians = new float[numOutputs];
        mads = new float[numOutputs];

        // Para cada columna (TP y SL), calcular estadísticas
        for (int col = 0; col < numOutputs; col++) {
            List<Float> values = new ArrayList<>();

            for (float[] row : y) {
                if (col < row.length) {
                    values.add(row[col]);
                }
            }

            if (values.isEmpty()) {
                throw new IllegalArgumentException("No hay datos para la columna " + col);
            }

            Collections.sort(values);
            medians[col] = median(values);
            mads[col] = mad(values, medians[col]);
        }
    }


    public float[][] transform(float @NotNull [][] y) {
        if (medians == null || mads == null) {
            throw new IllegalStateException("Normalizador no ajustado aún");
        }

        if (y == null) {
            return new float[0][];
        }

        float[][] normalized = new float[y.length][];

        for (int i = 0; i < y.length; i++) {
            normalized[i] = new float[y[i].length];

            for (int col = 0; col < y[i].length; col++) {
                if (col < medians.length) {
                    // Normalización robusta con MAD
                    normalized[i][col] = (y[i][col] - medians[col]) / (mads[col] + EPSILON);
                } else {
                    // Si hay más columnas de las esperadas, copiar sin normalizar
                    normalized[i][col] = y[i][col];
                }
            }
        }

        return normalized;
    }

    public float[][] inverseTransform(float[][] yNorm) {
        if (medians == null || mads == null) {
            throw new IllegalStateException("Normalizador no ajustado aún");
        }

        if (yNorm == null) {
            return new float[0][];
        }

        float[][] original = new float[yNorm.length][];

        for (int i = 0; i < yNorm.length; i++) {
            original[i] = new float[yNorm[i].length];

            for (int col = 0; col < yNorm[i].length; col++) {
                if (col < medians.length) {
                    // Desnormalización
                    original[i][col] = yNorm[i][col] * mads[col] + medians[col];
                } else {
                    // Si hay más columnas de las esperadas, copiar directamente
                    original[i][col] = yNorm[i][col];
                }
            }
        }

        return original;
    }

    private float median(List<Float> values) {
        int n = values.size();
        if (n % 2 == 0) {
            return (values.get(n/2 - 1) + values.get(n/2)) / 2.0f;
        } else {
            return values.get(n/2);
        }
    }

    private float mad(List<Float> values, float median) {
        List<Float> deviations = new ArrayList<>();
        for (float value : values) {
            deviations.add(Math.abs(value - median));
        }
        Collections.sort(deviations);
        return median(deviations);
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

}