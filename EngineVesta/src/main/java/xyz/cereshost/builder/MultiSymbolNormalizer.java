package xyz.cereshost.builder;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Getter
public class MultiSymbolNormalizer {
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
        mins = new float[numOutputs]; // Inicializar array de mínimos

        for (int col = 0; col < numOutputs; col++) {
            // CRÍTICO: Si es la columna 2 (Dirección), no calculamos estadísticas de normalización
            if (col == 2) {
                medians[col] = 0.0f;
                mads[col] = 1.0f;
                mins[col] = 0.0f;
                continue;
            }

            List<Float> values = new ArrayList<>();
            for (float[] row : y) {
                if (col < row.length) values.add(row[col]);
            }

            if (values.isEmpty()) continue;

            Collections.sort(values);
            float minVal = values.get(0); // Mínimo de la columna
            float med = median(values);
            float deviation = mad(values, med);

            medians[col] = med;
            mads[col] = Math.max(deviation, EPSILON);
            mins[col] = minVal; // Guardar el mínimo
        }
    }

    public float[][] transform(float[][] y) {
        if (medians == null) throw new IllegalStateException("Normalizador no ajustado");
        float[][] normalized = new float[y.length][numOutputs];
        for (int i = 0; i < y.length; i++) {
            for (int col = 0; col < numOutputs; col++) {
                if (col == 2) {
                    normalized[i][col] = y[i][col];
                } else {
                    normalized[i][col] = (y[i][col] - mins[col]) / mads[col];
                }
            }
        }
        return normalized;
    }

    public float[][] inverseTransform(float[][] yNorm) {
        float[][] original = new float[yNorm.length][yNorm[0].length];
        for (int i = 0; i < yNorm.length; i++) {
            for (int col = 0; col < yNorm[i].length; col++) {
                if (col == 2) {
                    original[i][col] = yNorm[i][col]; // No des-normalizar la probabilidad
                } else if (col < medians.length) {
                    original[i][col] = (yNorm[i][col] * mads[col]) + mins[col];
                }
            }
        }
        return original;
    }

    private float median(@NotNull List<Float> values) {
        int n = values.size();
        return (n % 2 == 0) ? (values.get(n/2 - 1) + values.get(n/2)) / 2.0f : values.get(n/2);
    }

    private float mad(@NotNull List<Float> values, float median) {
        List<Float> deviations = new ArrayList<>();
        for (float v : values) deviations.add(Math.abs(v - median));
        Collections.sort(deviations);
        return median(deviations);
    }
}