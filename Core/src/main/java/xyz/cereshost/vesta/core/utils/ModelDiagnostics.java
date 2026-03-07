package xyz.cereshost.vesta.core.utils;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.io.IOdata;

import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

public final class ModelDiagnostics {

    private static final int SAMPLE_LIMIT = 2048;
    private static final int CHUNK_SIZE = 1024;

    private ModelDiagnostics() {}

    public static void run() throws IOException {
        if (!IOdata.isBuiltData()) {
            Vesta.error("No hay cache de entrenamiento. Ejecuta primero build/training.");
            return;
        }

        Model model = IOdata.loadModel();
        Pair<XNormalizer, YNormalizer> norms = IOdata.loadNormalizers();
        TrainingData data = IOdata.getBuiltData();
//        if (data.isLoadInRam()) {
//            data.prepareNormalize();
//        } else {
//            data.setXNormalizer(norms.getKey());
//            data.setYNormalizer(norms.getValue());
//        }

        Pair<float[][][], float[][]> test = data.getTestNormalize();
        float[][][] x = test.getKey();
        float[][] y = test.getValue();

        if (x == null || x.length == 0) {
            Vesta.error("Test data vacia.");
            return;
        }

        int sampleSize = Math.min(SAMPLE_LIMIT, x.length);
        int[] sampleIdx = sampleIndices(x.length, sampleSize, 1337);
        float[][][] sampleX = new float[300_000][x[0].length][x[0][0].length];
        float[][] sampleY = new float[300_000][y[0].length];
        for (int i = 0; i < sampleSize; i++) {
            int idx = sampleIdx[i];
            sampleX[i] = x[idx];
            sampleY[i] = y[idx];
        }

        float[][][] lastOnly1 = maskHistory(sampleX, 1);
        float[][][] lastOnly5 = maskHistory(sampleX, 5);
        float[][][] shuffled = shuffleTime(sampleX, 1337);
        float[][][] reversed = reverseTime(sampleX);

        Predictions base = predict(model, norms.getValue(), sampleX);
        Predictions pLast1 = predict(model, norms.getValue(), lastOnly1);
        Predictions pLast5 = predict(model, norms.getValue(), lastOnly5);
        Predictions pShuf = predict(model, norms.getValue(), shuffled);
        Predictions pRev = predict(model, norms.getValue(), reversed);

        float[] trueRatio = ratioFromY(norms.getValue(), sampleY);

        Vesta.info("----- Diagnostico de Sensibilidad -----");
        printStats("Ratio real", trueRatio);
        printStats("Ratio pred", base.ratio);

        Vesta.info("Sensibilidad Ratio | last=1  MAE: %.6f", mae(base.ratio, pLast1.ratio));
        Vesta.info("Sensibilidad Ratio | last=5  MAE: %.6f", mae(base.ratio, pLast5.ratio));
        Vesta.info("Sensibilidad Ratio | shuffle MAE: %.6f", mae(base.ratio, pShuf.ratio));
        Vesta.info("Sensibilidad Ratio | reverse MAE: %.6f", mae(base.ratio, pRev.ratio));
        printLookbackSweep(model, norms.getValue(), base, sampleX);
        Vesta.info("Correlacion (pred vs real): %.4f", corr(base.ratio, trueRatio));
        Vesta.info("Frac |pred| > 0.95: %.2f%%", 100.0 * fracAbsGreater(base.ratio, 0.95f));
        Vesta.info("Frac |pred| < 0.10: %.2f%%", 100.0 * fracAbsLess(base.ratio, 0.10f));
    }

    private static Predictions predict(Model model, YNormalizer yNormalizer, float[][][] x) {
        try (NDManager manager = model.getNDManager().newSubManager()) {
            NDArray xNd = EngineUtils.concat3DArrayToNDArray(x, manager, CHUNK_SIZE);
            ParameterStore ps = new ParameterStore(manager, false);
            NDArray yNd = model.getBlock().forward(ps, new NDList(xNd), false).singletonOrThrow();
            float[] flat = yNd.toFloatArray();
            int batch = x.length;
            float[][] raw = new float[batch][5];
            float[] ratio = new float[batch];
            for (int i = 0; i < batch; i++) {
                int base = i * 5;
                raw[i][0] = flat[base];
                raw[i][1] = flat[base + 1];
                raw[i][2] = 0f;
                raw[i][3] = 0f;
                raw[i][4] = 0f;
            }
            float[][] denorm = yNormalizer.inverseTransform(raw);
            for (int i = 0; i < batch; i++) {
                ratio[i] = ratioFromMoves(denorm[i][0], denorm[i][1]);
            }
            return new Predictions(ratio);
        }
    }

    private static float[][][] maskHistory(float[][][] src, int keepLast) {
        int batch = src.length;
        int seqLen = src[0].length;
        int feat = src[0][0].length;
        int keep = Math.max(1, Math.min(keepLast, seqLen));
        float[][][] out = new float[batch][seqLen][feat];
        for (int i = 0; i < batch; i++) {
            for (int t = 0; t < seqLen; t++) {
                if (t >= seqLen - keep) {
                    System.arraycopy(src[i][t], 0, out[i][t], 0, feat);
                } else {
                    Arrays.fill(out[i][t], 0f);
                }
            }
        }
        return out;
    }

    private static float[][][] shuffleTime(float[][][] src, long seed) {
        int batch = src.length;
        int seqLen = src[0].length;
        int feat = src[0][0].length;
        int[] perm = new int[seqLen];
        for (int i = 0; i < seqLen; i++) perm[i] = i;
        Random rnd = new Random(seed);
        for (int i = seqLen - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }
        float[][][] out = new float[batch][seqLen][feat];
        for (int i = 0; i < batch; i++) {
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(src[i][perm[t]], 0, out[i][t], 0, feat);
            }
        }
        return out;
    }

    private static float[][][] reverseTime(float[][][] src) {
        int batch = src.length;
        int seqLen = src[0].length;
        int feat = src[0][0].length;
        float[][][] out = new float[batch][seqLen][feat];
        for (int i = 0; i < batch; i++) {
            for (int t = 0; t < seqLen; t++) {
                System.arraycopy(src[i][seqLen - 1 - t], 0, out[i][t], 0, feat);
            }
        }
        return out;
    }

    private static float[] column(float[][] data, int col) {
        float[] out = new float[data.length];
        for (int i = 0; i < data.length; i++) {
            out[i] = col < data[i].length ? data[i][col] : 0f;
        }
        return out;
    }

    private static float mae(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            sum += Math.abs(a[i] - b[i]);
        }
        return (float) (sum / Math.max(1, n));
    }

    private static float corr(float[] a, float[] b) {
        int n = Math.min(a.length, b.length);
        if (n == 0) return 0f;
        double meanA = 0.0;
        double meanB = 0.0;
        for (int i = 0; i < n; i++) {
            meanA += a[i];
            meanB += b[i];
        }
        meanA /= n;
        meanB /= n;
        double num = 0.0;
        double denA = 0.0;
        double denB = 0.0;
        for (int i = 0; i < n; i++) {
            double da = a[i] - meanA;
            double db = b[i] - meanB;
            num += da * db;
            denA += da * da;
            denB += db * db;
        }
        double den = Math.sqrt(denA * denB);
        return den == 0.0 ? 0f : (float) (num / den);
    }

    private static double fracAbsGreater(float[] data, float threshold) {
        int n = data.length;
        if (n == 0) return 0.0;
        int count = 0;
        for (float v : data) {
            if (Math.abs(v) > threshold) count++;
        }
        return (double) count / n;
    }

    private static double fracAbsLess(float[] data, float threshold) {
        int n = data.length;
        if (n == 0) return 0.0;
        int count = 0;
        for (float v : data) {
            if (Math.abs(v) < threshold) count++;
        }
        return (double) count / n;
    }

    private static void printStats(String name, float[] data) {
        float[] copy = Arrays.copyOf(data, data.length);
        Arrays.sort(copy);
        float p1 = percentile(copy, 1);
        float p10 = percentile(copy, 10);
        float p50 = percentile(copy, 50);
        float p90 = percentile(copy, 90);
        float p99 = percentile(copy, 99);
        float mean = mean(copy);
        float std = std(copy, mean);
        Vesta.info("%s | mean=%.4f std=%.4f p1=%.4f p10=%.4f p50=%.4f p90=%.4f p99=%.4f",
                name, mean, std, p1, p10, p50, p90, p99);
    }

    private static void printLookbackSweep(Model model, YNormalizer yNormalizer, Predictions base, float[][][] sampleX) {
        int seqLen = sampleX[0].length;
        int[] ks = new int[]{1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 120, 140, 160, 180, 200, 220, 239};
        Vesta.info("Sweep lookback (mantener ultimos K pasos):");
        for (int k : ks) {
            int keep = Math.min(k, seqLen);
            float[][][] masked = maskHistory(sampleX, keep);
            Predictions p = predict(model, yNormalizer, masked);
            Vesta.info("  K=%d -> MAE: %.6f", keep, mae(base.ratio, p.ratio));
        }
    }

    private static float mean(float[] data) {
        if (data.length == 0) return 0f;
        double sum = 0.0;
        for (float v : data) sum += v;
        return (float) (sum / data.length);
    }

    private static float std(float[] data, float mean) {
        if (data.length == 0) return 0f;
        double sum = 0.0;
        for (float v : data) {
            double d = v - mean;
            sum += d * d;
        }
        return (float) Math.sqrt(sum / data.length);
    }

    private static float percentile(float[] sorted, int pct) {
        if (sorted.length == 0) return 0f;
        if (sorted.length == 1) return sorted[0];
        double rank = (pct / 100.0) * (sorted.length - 1);
        int lo = (int) Math.floor(rank);
        int hi = (int) Math.ceil(rank);
        if (lo == hi) return sorted[lo];
        double w = rank - lo;
        return (float) (sorted[lo] * (1.0 - w) + sorted[hi] * w);
    }

    private static int[] sampleIndices(int total, int sampleSize, long seed) {
        int[] idx = new int[total];
        for (int i = 0; i < total; i++) idx[i] = i;
        Random rnd = new Random(seed);
        for (int i = total - 1; i > 0; i--) {
            int j = rnd.nextInt(i + 1);
            int tmp = idx[i];
            idx[i] = idx[j];
            idx[j] = tmp;
        }
        return Arrays.copyOf(idx, sampleSize);
    }

    private static float[] ratioFromY(YNormalizer yNormalizer, float[][] yNorm) {
        float[][] denorm = yNormalizer.inverseTransform(yNorm);
        float[] out = new float[denorm.length];
        for (int i = 0; i < denorm.length; i++) {
            out[i] = ratioFromMoves(denorm[i][0], denorm[i][1]);
        }
        return out;
    }

    private static float ratioFromMoves(float upMove, float downMove) {
        float maxMove = Math.max(upMove, downMove);
        if (!Float.isFinite(maxMove) || maxMove <= 0f) return 0f;
        float ratio = (upMove - downMove) / maxMove;
        return PredictionUtils.clampRatio(ratio);
    }

    private record Predictions(float[] ratio) {}
}
