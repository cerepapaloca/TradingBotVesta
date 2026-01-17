package xyz.cereshost.builder;

import lombok.Getter;

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
    private float priceMedian;
    private float priceMAD; // Median Absolute Deviation
    private static final float EPSILON = 1e-8f;

    public void fit(float[] y) {
        List<Float> closes = new ArrayList<>();

        for (float value : y) {
            closes.add(value);
        }

        Collections.sort(closes);
        priceMedian = median(closes);
        priceMAD = mad(closes, priceMedian);

        System.out.println("  Close stats - Median: " + priceMedian + ", MAD: " + priceMAD);
    }

    public float[] transform(float[] y) {
        float[] normalized = new float[y.length];

        for (int i = 0; i < y.length; i++) {
            // Normalización robusta con MAD
            normalized[i] = (y[i] - priceMedian) / (priceMAD + EPSILON);

            // Clamping para valores extremos
            normalized[i] = clamp(normalized[i], -10, 10);
        }

        return normalized;
    }

    public float[] inverseTransform(float[] yNorm) {  // Cambiado: float[][] -> float[]
        float[] original = new float[yNorm.length];

        for (int i = 0; i < yNorm.length; i++) {
            original[i] = yNorm[i] * priceMAD + priceMedian;
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