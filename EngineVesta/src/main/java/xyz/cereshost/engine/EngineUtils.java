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

import static xyz.cereshost.engine.PredictionEngine.THRESHOLD_PRICE;
import static xyz.cereshost.engine.PredictionEngine.THRESHOLD_RELATIVE;

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
    public static ResultsEvaluate evaluateModel(Trainer trainer, NDArray X_test, NDArray y_test,
                                                MultiSymbolNormalizer yNormalizer) {

        NDList predictions = trainer.evaluate(new NDList(X_test));
        NDArray yPred = predictions.singletonOrThrow();

        // Convertir a float arrays para manipulación manual
        float[] yTestFlat = y_test.toFloatArray();
        float[] yPredFlat = yPred.toFloatArray();

        long[] shape = y_test.getShape().getShape();
        int batchSize = (int) shape[0];

        // Ahora tenemos 5 salidas en predicciones, pero el target sigue teniendo 3
        int targetCols = 3;  // [TP, SL, dirección_continua]
        int predCols = 5;    // [TP, SL, LONG_prob, NEUTRAL_prob, SHORT_prob]

        double totalMaeUP = 0;
        double totalMaeDOWN = 0;
        double totalMaeDir = 0;

        List<ResultPrediction> results = new ArrayList<>();

        for (int i = 0; i < batchSize; i++) {
            // Índices para los arrays planos
            int targetIdx = i * targetCols;
            int predIdx = i * predCols;

            // 1. Extraer valores del target (3 columnas)
            float rawRealTP = yTestFlat[targetIdx];
            float rawRealSL = yTestFlat[targetIdx + 1];
            float rawRealLong = yTestFlat[targetIdx + 2]; // Dirección continua [-1, 1]
            float rawRealNeutral = yTestFlat[targetIdx + 3]; // Dirección continua [-1, 1]
            float rawRealShort = yTestFlat[targetIdx + 24]; // Dirección continua [-1, 1]

            float rawRealDir = PredictionEngine.computeDirection(
                    new float[]{rawRealLong, rawRealNeutral, rawRealShort}
            );

            // 2. Extraer predicciones (5 columnas)
            float rawPredTP = yPredFlat[predIdx];
            float rawPredSL = yPredFlat[predIdx + 1];
            float rawPredLong = yPredFlat[predIdx + 2];    // Probabilidad LONG
            float rawPredNeutral = yPredFlat[predIdx + 3]; // Probabilidad NEUTRAL
            float rawPredShort = yPredFlat[predIdx + 4];   // Probabilidad SHORT

            // 3. Convertir probabilidades a dirección continua [-1, 1]
            float predDirection = PredictionEngine.computeDirection(
                    new float[]{rawPredLong, rawPredNeutral, rawPredShort}
            );

            // 4. Crear arrays temporales para desnormalización
            // Target: [TP, SL] (2 columnas)
            float[][] targetArray = new float[][]{{rawRealTP, rawRealSL}};
            float[][] predArray = new float[][]{{rawPredTP, rawPredSL}};

            // 5. Desnormalizar TP y SL
            float[][] denormTarget = yNormalizer.inverseTransform(targetArray);
            float[][] denormPred = yNormalizer.inverseTransform(predArray);

            float realTP = denormTarget[0][0];
            float realSL = denormTarget[0][1];
            float predTP = denormPred[0][0];
            float predSL = denormPred[0][1];

            // 6. Métricas
            totalMaeUP += Math.abs(realTP - predTP);
            totalMaeDOWN += Math.abs(realSL - predSL);
            totalMaeDir += Math.abs(rawRealDir - predDirection); // MAE de dirección

            // 7. Crear ResultPrediction con el nuevo formato
            results.add(new ResultPrediction(
                    predTP, predSL, predDirection, // Predicciones
                    realTP, realSL, rawRealDir,    // Valores reales
                    i
            ));
        }

        double avgMaeUP = totalMaeUP / batchSize;
        double avgMaeDOWN = totalMaeDOWN / batchSize;
        double avgMaeDir = totalMaeDir / batchSize;

        return new ResultsEvaluate("VestaIA", avgMaeUP, avgMaeDOWN, results);
    }

    public record ResultsEvaluate(
            String modelName,
            double avgMaeTP,
            double avgMaeSL,
            List<ResultPrediction> resultPrediction
    ) {
        public float hitRateSimple(){
            int hits = 0;
            int nohits = 0;
            for (ResultPrediction prediction : resultPrediction) {
                if (prediction.predDir() == 0) continue;
                // ley de los signos
                if (prediction.realDir() * prediction.predDir() > 0) {
                    hits++;
                }else {
                    nohits++;
                }

            }
            int total = nohits + hits;
            return total > 0 ? ((float) hits / total) *100 : 0;
        }

        public float hitRateAdvanced() {
            int hits = 0;
            int total = 0;

            float threshold = (float) THRESHOLD_RELATIVE;

            for (ResultPrediction prediction : resultPrediction) {
                float pred = prediction.predDir();
                float real = prediction.realDir();
                if (pred > threshold && real > 0) hits++;// Long
                else if (pred < -threshold && real < 0) hits++; // Short
                else if (pred <= threshold && pred >= -threshold && real == 0) hits++; // Neutral
                total++;
            }

            return total > 0 ? ((float) hits / total) * 100 : 0;
        }

        public float hitRateSafe() {
            int hits = 0;
            int fails = 0;

            float threshold = (float) THRESHOLD_RELATIVE;

            for (ResultPrediction prediction : resultPrediction) {
                float pred = prediction.predDir();
                float real = prediction.realDir();

                if (real > threshold) { // Long real
                    if (pred > threshold) hits++;
                    else fails++;
                }
                else if (real < -threshold) { // Short real
                    if (pred < -threshold) hits++;
                    else fails++;
                }
                // si está entre -threshold y +threshold no cuenta
            }

            int total = hits + fails;
            // A / (A + B) * 100
            return total > 0 ? ((float) hits / total) * 100f : 0f;
        }

        public int @NotNull [] hitRateLong() {
            int[] hits = new int[3];
            for (ResultPrediction prediction : resultPrediction){
                if (prediction.realDir() > THRESHOLD_RELATIVE){
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int @NotNull [] hitRateShort() {
            int[] hits = new int[3];
            for (ResultPrediction prediction : resultPrediction){
                if (prediction.realDir() < -THRESHOLD_RELATIVE){
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int @NotNull [] hitRateNeutral() {
            int[] hits = new int[3];
            for (ResultPrediction prediction : resultPrediction){
                if (prediction.realDir() > -THRESHOLD_RELATIVE && prediction.realDir() < THRESHOLD_RELATIVE) {
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        private void computeDir(int[] hits, @NotNull ResultPrediction prediction) {
            boolean signalLong = prediction.predDir() > THRESHOLD_RELATIVE;
            boolean signalShort = prediction.predDir() < -THRESHOLD_RELATIVE;
            if (signalLong) {
                hits[0]++; // Long
            } else if (signalShort) {
                hits[1]++; // Short
            } else {
                hits[2]++; // Neutral
            }
        }

    }

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
                Vesta.warning("Eliminando muestra con TP/SL inválido en índice " + i);
                continue;
            }

            // 2) filtrar valores negativos o cero
            if (tp <= 0 || sl <= 0) {
                removedBadValue++;
                Vesta.warning("Eliminando muestra con TP/SL <= 0 en índice " + i);
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
                Vesta.warning("Eliminando duplicado en índice " + i);
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