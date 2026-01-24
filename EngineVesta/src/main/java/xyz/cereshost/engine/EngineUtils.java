package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.Trainer;
import ai.djl.util.Pair;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.common.Vesta;

import java.util.*;

@UtilityClass
public class EngineUtils {

    public static void checkEngines() {
        Vesta.info("=== Verificando Engines DJL ===");

        for (String engineName : ai.djl.engine.Engine.getAllEngines()) {
            Vesta.info("\nEngine: " + engineName);
            ai.djl.engine.Engine engine = ai.djl.engine.Engine.getEngine(engineName);
            if (engine != null) {
                Vesta.info("  Version: " + engine.getVersion());
                Vesta.info("  Dispositivos disponibles:");

                for (Device device : engine.getDevices()) {
                    Vesta.info("    - " + device +
                            " (GPU: " + device.isGpu() +
                            ", ID: " + device.getDeviceId() +
                            ", C: " + engine.hasCapability(StandardCapabilities.CUDA) + ")");
                }
            } else {
                Vesta.info("  No disponible");
            }
        }
    }

    /**
     * Mezcla los datos aleatoriamente (actualizado para float[][])
     */
    public void shuffleData(float[][][] X, float[][] y) {
        Random rand = new Random(42);
        for (int i = X.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            // Intercambiar X
            float[][] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            // Intercambiar y (ahora con dos columnas)
            float[] tempY = y[i];
            y[i] = y[j];
            y[j] = tempY;
        }
    }

    /**
     * Aplanar array 3D a 1D
     */
    public float @NotNull [] flatten3DArray(float[][] @NotNull [] array) {
        int samples = array.length;
        int lookback = array[0].length;
        int features = array[0][0].length;

        float[] flat = new float[samples * lookback * features];
        int idx = 0;

        for (float[][] sample : array) {
            for (int j = 0; j < lookback; j++) {
                System.arraycopy(sample[j], 0, flat, idx, features);
                idx += features;
            }
        }
        return flat;
    }

    /**
     * Limpiar valores NaN
     */
    public static void cleanNaNValues(float[][] @NotNull [] array) {
        for (float[][] sample : array) {
            for (float[] timestep : sample) {
                for (int k = 0; k < timestep.length; k++) {
                    if (Float.isNaN(timestep[k]) || Float.isInfinite(timestep[k])) {
                        timestep[k] = 0f;
                    }
                }
            }
        }
    }

    /**
     * Combina múltiples datasets en uno solo (actualizado para float[][])
     */
    @Contract("_, _ -> new")
    public static @NotNull Pair<float[][][], float[][]> combineDatasets(@NotNull List<float[][][]> allX, List<float[][]> allY) {
        int totalSamples = 0;
        for (float[][][] X : allX) {
            totalSamples += X.length;
        }

        int lookback = allX.get(0)[0].length;
        int features = allX.get(0)[0][0].length;

        float[][][] Xcombined = new float[totalSamples][lookback][features];
        float[][] ycombined = new float[totalSamples][3]; // Aumentado a 3 columnas

        int currentIndex = 0;
        for (int s = 0; s < allX.size(); s++) {
            float[][][] X = allX.get(s);
            float[][] y = allY.get(s);

            for (int i = 0; i < X.length; i++) {
                Xcombined[currentIndex] = X[i];
                ycombined[currentIndex][0] = y[i][0]; // Fuerza Alcista
                ycombined[currentIndex][1] = y[i][1]; // Fuerza Bajista
                ycombined[currentIndex][2] = y[i][2]; // Dirección (0 o 1)
                currentIndex++;
            }
        }
        return new Pair<>(Xcombined, ycombined);
    }

    /**
     * Evalúa el modelo con lógica de 3 salidas: Regresión (UP/DOWN) + Clasificación (DIR)
     */
    public static ResultsEvaluate evaluateModel(Trainer trainer, NDArray X_test, NDArray y_test, MultiSymbolNormalizer yNormalizer) {
        NDList predictions = trainer.evaluate(new NDList(X_test));
        NDArray yPred = predictions.singletonOrThrow();

        // Convertir a float arrays para manipulación manual
        float[] yTestFlat = y_test.toFloatArray();
        float[] yPredFlat = yPred.toFloatArray();

        long[] shape = y_test.getShape().getShape();
        int batchSize = (int) shape[0];
        // Asumimos que numOutputs es 3

        double totalMaeUP = 0;
        double totalMaeDOWN = 0;
        int correctDirections = 0;
        int hitsProfitability = 0;

        List<ResultPrediction> results = new ArrayList<>();

        for (int i = 0; i < batchSize; i++) {
            int idx = i * 3;

            // 1. Extraer RAW values (Tal como salen del modelo/dataset)
            float rawRealUP = yTestFlat[idx];
            float rawRealDOWN = yTestFlat[idx + 1];
            float rawRealDir = yTestFlat[idx + 2]; // 0 o 1 (No debe estar normalizado)

            float rawPredUP = yPredFlat[idx];
            float rawPredDOWN = yPredFlat[idx + 1];
            float rawPredDir = yPredFlat[idx + 2]; // Probabilidad

            // 2. DES-NORMALIZAR SOLO UP Y DOWN
            // Creamos un array temporal solo con las partes de regresión
            float[][] tempInput = new float[][]{{rawRealUP, rawRealDOWN}, {rawPredUP, rawPredDOWN}};

            // Usamos el normalizer que (asumimos) fue entrenado solo con 2 columnas o maneja el split
            // FIX: Si el normalizer espera 3 columnas, esto fallará.
            // Asumiremos que el usuario arregló el normalizer O haremos un truco:
            // Des-normalizamos manualmente si tenemos acceso a media/std, o usamos el normalizer:

            float[][] tempOutput;
            try {
                // Intentamos desnormalizar el par [UP, DOWN].
                // NOTA: Esto requiere que tu yNormalizer acepte arrays de 2 columnas.
                tempOutput = yNormalizer.inverseTransform(tempInput);
            } catch (Exception e) {
                // Fallback si el normalizer es estricto con el tamaño:
                // Pasamos dummy 0 en la tercera columna para engañarlo, luego ignoramos el resultado
                float[][] dummyInput = new float[][]{
                        {rawRealUP, rawRealDOWN, 0},
                        {rawPredUP, rawPredDOWN, 0}
                };
                float[][] dummyOutput = yNormalizer.inverseTransform(dummyInput);
                tempOutput = new float[][]{
                        {dummyOutput[0][0], dummyOutput[0][1]},
                        {dummyOutput[1][0], dummyOutput[1][1]}
                };
            }

            float realUP = tempOutput[0][0];
            float realDOWN = tempOutput[0][1];
            float predUP = tempOutput[1][0];
            float predDOWN = tempOutput[1][1];

            // La dirección NO se toca
            float realDir = rawRealDir;
            float predDirProb = rawPredDir;

            // --- Métricas ---
            totalMaeUP += Math.abs(realUP - predUP);
            totalMaeDOWN += Math.abs(realDOWN - predDOWN);

            // Dirección: Umbral 0.5
            boolean isRealLong = realDir > 0.5f;
            boolean isPredLong = predDirProb > 0.5f;
            if (isRealLong == isPredLong) correctDirections++;

            // Rentabilidad (TP > SL)
            if ((realUP > realDOWN) == (predUP > predDOWN)) hitsProfitability++;

            results.add(new ResultPrediction(predUP, predDOWN, predDirProb, realUP, realDOWN, realDir, i));
        }

        double avgMaeUP = totalMaeUP / batchSize;
        double avgMaeDOWN = totalMaeDOWN / batchSize;
        double directionAccuracy = (double) correctDirections / batchSize * 100.0;
        double profitHitRate = (double) hitsProfitability / batchSize * 100.0;

        Vesta.info("=== Evaluación (3-Outputs) ===");
        Vesta.info("MAE UP: %.6f | MAE DOWN: %.6f", avgMaeUP, avgMaeDOWN);
        Vesta.info("Acc Dirección: %.2f%%", directionAccuracy);
        Vesta.info("Hit Rate Estructural: %.2f%%", profitHitRate);

        return new ResultsEvaluate("VestaHybrid", avgMaeUP, avgMaeDOWN, directionAccuracy, results);
    }

    public record ResultsEvaluate(
            String modelName,
            double avgMaeTP,
            double avgMaeSL,
            double hitRate,
            List<ResultPrediction> resultPrediction
    ) {}

    public record ResultPrediction(
            float predTP, float predSL, float predDir,
            float realTP, float realSL, float realDir,
            long timestamp
    ) {
        public float lsDiff() {
            return realSL - predSL;
        }

        public float tpDiff() {
            return realTP - predTP;
        }

        public float dirDiff() {
            return realDir - predDir;
        }
    }

    /**
     * Limpia datos (actualizado para float[][])
     */
    public static Pair<float[][][], float[][]> clearData(float[][][] X, float[][] y) {
        final float EPS = 1e-9f;

        Map<String, Integer> seen = new HashMap<>();
        List<float[][]> uniqueX = new ArrayList<>();
        List<float[]> uniqueY = new ArrayList<>(); // Cambiado a float[]

        int removedDuplicates = 0;
        int removedBadValue = 0;

        for (int i = 0; i < X.length; i++) {
            float tp = y[i][0];
            float sl = y[i][1];

            // 1) filtrar NaN / Inf / valores inválidos
            if (Float.isNaN(tp) || Float.isInfinite(tp) ||
                    Float.isNaN(sl) || Float.isInfinite(sl)) {
                removedBadValue++;
                Vesta.waring("Eliminando muestra con TP/SL inválido en índice " + i);
                continue;
            }

            // 2) filtrar valores negativos o cero
            if (tp <= 0 || sl <= 0) {
                removedBadValue++;
                Vesta.waring("Eliminando muestra con TP/SL <= 0 en índice " + i);
                continue;
            }

            // 3) detectar duplicados
            String hash = Arrays.deepToString(X[i]);
            if (!seen.containsKey(hash)) {
                seen.put(hash, i);
                uniqueX.add(X[i]);
                uniqueY.add(new float[]{tp, sl});
            } else {
                removedDuplicates++;
                Vesta.waring("Eliminando duplicado en índice " + i);
            }
        }

        // Convertir de vuelta a arrays
        float[][][] Xunique = new float[uniqueX.size()][][];
        float[][] yunique = new float[uniqueY.size()][2];

        for (int i = 0; i < Xunique.length; i++) {
            Xunique[i] = uniqueX.get(i);
            yunique[i][0] = uniqueY.get(i)[0];
            yunique[i][1] = uniqueY.get(i)[1];
        }

        Vesta.info("Datos limpiados: " + Xunique.length + " muestras válidas");
        Vesta.info("Eliminados " + removedDuplicates + " duplicados");
        Vesta.info("Eliminadas " + removedBadValue + " muestras con valores inválidos");

        return new Pair<>(Xunique, yunique);
    }

    public record SimulationResult(
            double initialBalance,
            double finalBalance,
            double netPnL,
            double roiPercent,
            int totalTrades,
            int winTrades,
            int lossTrades,
            double maxDrawdown
    ) {}
}