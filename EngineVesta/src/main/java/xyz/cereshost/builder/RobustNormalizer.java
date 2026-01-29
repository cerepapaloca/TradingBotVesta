package xyz.cereshost.builder;

import lombok.Getter;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

@Getter
public class RobustNormalizer {
    private float[] featureMedians;
    private float[] featureIQRs; // Rango intercuartílico
    private float epsilon = 1e-8f;

    public void fit(float[][][] X) {
        int features = X[0][0].length;

        featureMedians = new float[features];
        featureIQRs = new float[features];

        for (int k = 0; k < features; k++) {
            // Extraer todos los valores para esta feature
            List<Float> values = new ArrayList<>();
            for (float[][] sample : X) {
                for (float[] timestep : sample) {
                    values.add(timestep[k]);
                }
            }

//            // Calcular percentiles robustos
//            Collections.sort(values);
            int n = values.size();

            float q25 = values.get((int) (n * 0.25));
            float q50 = values.get((int) (n * 0.50));
            float q75 = values.get((int) (n * 0.75));

            featureMedians[k] = q50;
            featureIQRs[k] = q75 - q25 + epsilon;
        }
    }

    public float[][][] transform(float[][][] X) {
        if (featureMedians == null || featureIQRs == null) {
            throw new IllegalStateException("Normalizer not fitted yet");
        }

        int inputFeatures = X[0][0].length;
        if (inputFeatures != featureMedians.length) {
            throw new IllegalArgumentException(
                    String.format("Feature dimension mismatch. Expected %d, got %d",
                            featureMedians.length, inputFeatures)
            );
        }

        float[][][] normalized = new float[X.length][X[0].length][X[0][0].length];

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                for (int k = 0; k < X[0][0].length; k++) {
                    // Normalización robusta: (x - mediana) / IQR
                    normalized[i][j][k] = (X[i][j][k] - featureMedians[k]) / featureIQRs[k];

                    // Clamping para evitar valores extremos
                    //normalized[i][j][k] = Math.max(-10, Math.min(10, ));
                }
            }
        }
        return normalized;
    }
}
