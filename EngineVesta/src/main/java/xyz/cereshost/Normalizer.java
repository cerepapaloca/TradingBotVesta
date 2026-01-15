package xyz.cereshost;

public class Normalizer {
    private float[] featureMeans;
    private float[] featureStds;
    private float[] targetMeans;
    private float[] targetStds;

    public void fit(float[][][] X, float[][] y) {
        int samples = X.length;
        int lookback = X[0].length;
        int features = X[0][0].length;

        // Normalizar características
        featureMeans = new float[features];
        featureStds = new float[features];

        for (int k = 0; k < features; k++) {
            float sum = 0;
            float sumSq = 0;
            int count = 0;

            for (float[][] x : X) {
                for (int j = 0; j < lookback; j++) {
                    float val = x[j][k];
                    sum += val;
                    sumSq += val * val;
                    count++;
                }
            }

            featureMeans[k] = sum / count;
            featureStds[k] = (float) Math.sqrt(sumSq / count - featureMeans[k] * featureMeans[k]);
            if (featureStds[k] == 0) featureStds[k] = 1.0f; // evitar división por cero
        }

        // Normalizar targets
        targetMeans = new float[2];
        targetStds = new float[2];

        for (int k = 0; k < 2; k++) {
            float sum = 0;
            float sumSq = 0;

            for (int i = 0; i < samples; i++) {
                float val = y[i][k];
                sum += val;
                sumSq += val * val;
            }

            targetMeans[k] = sum / samples;
            targetStds[k] = (float) Math.sqrt(sumSq / samples - targetMeans[k] * targetMeans[k]);
            if (targetStds[k] == 0) targetStds[k] = 1.0f;
        }
    }

    public float[][][] transformX(float[][][] X) {
        float[][][] normalized = new float[X.length][X[0].length][X[0][0].length];

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                for (int k = 0; k < X[0][0].length; k++) {
                    normalized[i][j][k] = (X[i][j][k] - featureMeans[k]) / featureStds[k];
                }
            }
        }
        return normalized;
    }

    public float[][] transformY(float[][] y) {
        float[][] normalized = new float[y.length][y[0].length];

        for (int i = 0; i < y.length; i++) {
            for (int j = 0; j < 2; j++) {
                normalized[i][j] = (y[i][j] - targetMeans[j]) / targetStds[j];
            }
        }
        return normalized;
    }
}