package xyz.cereshost.utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static xyz.cereshost.engine.PredictionEngine.THRESHOLD_RELATIVE;

@UtilityClass
public class EngineUtils {

    public static final float MIN_DIRECTION_RATIO = 2.0f;

    public record DirectionFilterResult(
            int direction,
            float maxMove,
            float minMove,
            float ratio,
            float confidence
    ) {
        public Trading.DireccionOperation toOperation() {
            return directionToOperation(direction);
        }
    }

    public static @NotNull DirectionFilterResult filterDirectionByExtremes(float maxValue, float minValue) {
        float absMax = Math.abs(maxValue);
        float absMin = Math.abs(minValue);

        if (!Float.isFinite(absMax) || !Float.isFinite(absMin)) {
            return new DirectionFilterResult(0, absMax, absMin, 0f, 0f);
        }

        float diff = Math.abs(absMax - absMin);
        if (diff < (float) PredictionEngine.THRESHOLD_PRICE) {
            return new DirectionFilterResult(0, absMax, absMin, 0f, 0f);
        }

        int direction = absMax > absMin ? 1 : -1;
        float ratio = direction == 1
                ? (absMin == 0f ? Float.POSITIVE_INFINITY : absMax / absMin)
                : (absMax == 0f ? Float.POSITIVE_INFINITY : absMin / absMax);

        if (ratio < MIN_DIRECTION_RATIO) {
            return new DirectionFilterResult(0, absMax, absMin, ratio, 0f);
        }

        float confidence = (absMax + absMin == 0f) ? 0f : diff / (absMax + absMin);
        return new DirectionFilterResult(direction, absMax, absMin, ratio, confidence);
    }

    public static float @NotNull [] directionToOneHot(int direction) {
        if (direction > 0) {
            return new float[]{1f, 0f, 0f};
        }
        if (direction < 0) {
            return new float[]{0f, 0f, 1f};
        }
        return new float[]{0f, 1f, 0f};
    }

    public static @NotNull Trading.DireccionOperation directionToOperation(int direction) {
        if (direction > 0) {
            return Trading.DireccionOperation.LONG;
        }
        if (direction < 0) {
            return Trading.DireccionOperation.SHORT;
        }
        return Trading.DireccionOperation.NEUTRAL;
    }

//    public static void checkEngines() {
//        Vesta.info("=== Verificando Engines DJL ===");
//
//        for (String engineName : ai.djl.engine.Engine.getAllEngines()) {
//            Vesta.info("\nEngine: " + engineName);
//            ai.djl.engine.Engine engine = ai.djl.engine.Engine.getEngine(engineName);
//            if (engine != null) {
//                Vesta.info("  Version: " + engine.getVersion());
//                Vesta.info("  Dispositivos disponibles:");
//
//                for (Device device : engine.getDevices()) {
//                    Vesta.info("    - " + device +
//                            " (GPU: " + device.isGpu() +
//                            ", ID: " + device.getDeviceId() +
//                            ", C: " + engine.hasCapability(StandardCapabilities.CUDA) + ")");
//                }
//            } else {
//                Vesta.info("  No disponible");
//            }
//        }
//    }

    public static List<Dataset> splitIntoDatasets(
            NDArray X,
            NDArray y,
            int splits,
            int batchSize,
            Device device
    ) throws IOException, TranslateException {

        long samples = X.getShape().get(0);
        long splitSize = samples / splits;

        List<Dataset> datasets = new ArrayList<>();

        for (int i = 0; i < splits; i++) {
            long start = i * splitSize;
            long end = (i == splits - 1) ? samples : start + splitSize;

            NDArray Xpart = X.get(
                    new NDIndex(start + ":" + end + ",:,:")
            );
            NDArray ypart = y.get(
                    new NDIndex(start + ":" + end)
            );

            Dataset ds = new ArrayDataset.Builder()
                    .setData(Xpart)
                    .optLabels(ypart)
                    .setSampling(batchSize, true)
                    .optDevice(device)
                    .build();

            ds.prepare();
            datasets.add(ds);
        }

        return datasets;
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

    public static float[] flatten3DArraySlice(
            float[][][] array,
            int offset,
            int chunkSize
    ) {
        if (array == null || array.length == 0 || offset >= array.length) {
            return new float[0];
        }

        int totalSamples = array.length;
        int lookback = array[0].length;
        int features = array[0][0].length;

        // Calcular cuántas muestras podemos extraer realmente sin salirnos del array
        int actualSamples = Math.min(chunkSize, totalSamples - offset);
        if (actualSamples <= 0) return new float[0];

        float[] flat = new float[actualSamples * lookback * features];
        int idx = 0;

        for (int i = offset; i < offset + actualSamples; i++) {
            for (int t = 0; t < lookback; t++) {
                System.arraycopy(array[i][t], 0, flat, idx, features);
                idx += features;
            }
        }
        return flat;
    }

    public static NDArray create3D(
            NDManager manager,
            float[][][] data
    ) {
        int samples  = data.length;
        int lookback = data[0].length;
        int features = data[0][0].length;

        FloatBuffer buffer = FloatBuffer.allocate(samples * lookback * features);

        for (float[][] datum : data) {
            for (int t = 0; t < lookback; t++) {
                buffer.put(datum[t]);
            }
        }

        buffer.rewind();
        return manager.create(buffer, new Shape(samples, lookback, features));
    }

    public static NDArray concat3DArrayToNDArray(
            float[][][] sourceArray,
            NDManager manager,
            int chunkSize // Tamaño del bloque, ej: 512 o 1024
    ) {
        if (sourceArray == null || sourceArray.length == 0) {
            return manager.create(new Shape(0, 0, 0));
        }

        int totalSamples = sourceArray.length;
        int lookback = sourceArray[0].length;
        int features = sourceArray[0][0].length;

        List<NDArray> ndList = new ArrayList<>();

        // Recorremos el array original en saltos de 'chunkSize'
        for (int offset = 0; offset < totalSamples; offset += chunkSize) {

            // 1. Extraemos y aplanamos la "rebanada" usando el método anterior
            float[] flatSlice = flatten3DArraySlice(sourceArray, offset, chunkSize);

            // 2. Calculamos cuántos samples hay realmente en este trozo (el último puede ser más pequeño)
            int currentSamples = flatSlice.length / (lookback * features);

            // 3. Creamos el NDArray para este trozo específico
            NDArray ndChunk = manager.create(flatSlice, new Shape(currentSamples, lookback, features));

            // 4. Lo añadimos a la lista para concatenación
            ndList.add(ndChunk);
        }

        // 5. Unimos todos los trozos en un solo NDArray en el eje 0 (Samples)
        NDArray result = NDArrays.concat(new NDList(ndList), 0);

        // Opcional: Cerrar los NDArrays intermedios para liberar memoria nativa rápido
        for (NDArray chunk : ndList) {
            if (chunk != result) chunk.close();
        }

        return result;
    }

    /**
     * Evalúa el modelo con lógica de 3 salidas: Regresión (UP/DOWN) + Clasificación (DIR)
     */
    public static ResultsEvaluate evaluateModel(
            Trainer trainer,
            NDArray X_test,
            NDArray y_test,
            YNormalizer yNormalizer,
            int chunkSize // Recomendado: 512 o 1024
    ) {
        long totalSamples = X_test.getShape().get(0);
        int targetCols = (int) y_test.getShape().get(1); // Dinamico: usualmente 5 [Max, Min, 0, 0, 0]
        int predCols = 5; // Formato Vesta: [Max, Min, 0, 0, 0]

        double totalMaeUP = 0;
        double totalMaeDOWN = 0;
        double totalMaeDir = 0;
        List<ResultEvaluate> allResults = new ArrayList<>();

        // Procesar por bloques
        for (int start = 0; start < totalSamples; start += chunkSize) {
            int end = (int) Math.min(start + chunkSize, totalSamples);
            int currentBatchSize = end - start;

            // 1. Slicing de los datos de prueba (sin copiar memoria si es posible)
            try (NDArray xChunk = X_test.get(new NDIndex("{}:{}", start, end));
                 NDArray yChunk = y_test.get(new NDIndex("{}:{}", start, end))) {

                // 2. Inferencia del bloque
                NDList predictions = trainer.evaluate(new NDList(xChunk));
                NDArray yPred = predictions.singletonOrThrow();

                // 3. Conversion a arrays planos para procesamiento rapido en CPU
                float[] yTestFlat = yChunk.toFloatArray();
                float[] yPredFlat = yPred.toFloatArray();

                // 4. Procesar cada muestra dentro del chunk
                for (int i = 0; i < currentBatchSize; i++) {
                    int targetIdx = i * targetCols;
                    int predIdx = i * predCols;

                    // Extraer Target (ajusta los indices segun tu estructura real)
                    float rawRealTP = yTestFlat[targetIdx];
                    float rawRealSL = yTestFlat[targetIdx + 1];

                    // Extraer Prediccion
                    float rawPredTP = yPredFlat[predIdx];
                    float rawPredSL = yPredFlat[predIdx + 1];

                    // 5. Desnormalizacion (TP y SL)
                    float[][] denormTarget = yNormalizer.inverseTransform(new float[][]{{rawRealTP, rawRealSL}});
                    float[][] denormPred = yNormalizer.inverseTransform(new float[][]{{rawPredTP, rawPredSL}});

                    float realTP = denormTarget[0][0];
                    float realSL = denormTarget[0][1];
                    float predTP = denormPred[0][0];
                    float predSL = denormPred[0][1];

                    DirectionFilterResult realDecision = filterDirectionByExtremes(realTP, realSL);
                    DirectionFilterResult predDecision = filterDirectionByExtremes(predTP, predSL);
                    float rawRealDir = realDecision.direction();
                    float predDirection = predDecision.direction();
                    float[] predOneHot = directionToOneHot(predDecision.direction());
                    float[] realOneHot = directionToOneHot(realDecision.direction());

                    // 6. Acumular metricas
                    totalMaeUP += Math.abs(realTP - predTP);
                    totalMaeDOWN += Math.abs(realSL - predSL);
                    totalMaeDir += Math.abs(rawRealDir - predDirection);

                    // 7. Guardar resultado individual
                    allResults.add(new ResultEvaluate(
                            predTP, predSL, predDirection,
                            realTP, realSL, rawRealDir,
                            predOneHot[0],
                            realOneHot[0],
                            predOneHot[1],
                            realOneHot[1],
                            predOneHot[2],
                            realOneHot[2],
                            start + i // Indice global
                    ));
                }

                // Liberar prediccion del chunk
                predictions.close();
            }
        }

        // 8. Consolidar promedios finales
        double avgMaeUP = totalMaeUP / totalSamples;
        double avgMaeDOWN = totalMaeDOWN / totalSamples;
        double avgMaeDir = totalMaeDir / totalSamples;

        Vesta.info("Evaluacion Finalizada -> MAE TP: %.6f, MAE SL: %.6f, MAE Dir: %.6f", avgMaeUP, avgMaeDOWN, avgMaeDir);

        return new ResultsEvaluate("VestaIA_Chunked", avgMaeUP, avgMaeDOWN, allResults);
    }

    public record ResultsEvaluate(
            String modelName,
            double avgMaeTP,
            double avgMaeSL,
            List<ResultEvaluate> resultEvaluate
    ) {
        public float hitRateSimple(){
            int hits = 0;
            int fails = 0;
            for (ResultEvaluate prediction : resultEvaluate) {
                if (prediction.getPredDirection() == Trading.DireccionOperation.NEUTRAL || prediction.getRealDirection() == Trading.DireccionOperation.NEUTRAL) continue;
                // ley de los signos
                if (prediction.getPredDirection() == prediction.getRealDirection()) {
                    hits++;
                }else {
                    fails++;
                }

            }
            int total = fails + hits;
            return total > 0 ? ((float) hits / total) *100 : 0;
        }

        public float hitRateAdvanced() {
            int hits = 0;
            int total = 0;

            for (ResultEvaluate prediction : resultEvaluate) {
                if (prediction.getPredDirection() == prediction.getRealDirection()) {
                    hits++;
                }
                total++;
            }

            return total > 0 ? ((float) hits / total) * 100 : 0;
        }

        public float hitRateSafe() {
            int hits = 0;
            int fails = 0;

            float threshold = (float) THRESHOLD_RELATIVE;

            for (ResultEvaluate prediction : resultEvaluate) {
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

        public int @NotNull [] hitRateLongOneFloat() {
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.realDir() > THRESHOLD_RELATIVE){
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int @NotNull [] hitRateShortOneFloat() {
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.realDir() < -THRESHOLD_RELATIVE){
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int @NotNull [] hitRateNeutralOneFloat() {
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.realDir() > -THRESHOLD_RELATIVE && prediction.realDir() < THRESHOLD_RELATIVE) {
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int[] hitRate(Trading.DireccionOperation direccion){
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.getRealDirection() == direccion) {
                    switch (prediction.getPredDirection()) {
                        case LONG -> hits[0]++;
                        case SHORT -> hits[1]++;
                        case NEUTRAL -> hits[2]++;
                    }
                }
            }
            return hits;
        }

        public float hitRateConfident(float minConfidence) {
            int hits = 0;
            int total = 0;

            // Usamos el threshold para definir qué es un acierto real
            float threshold = (float) THRESHOLD_RELATIVE;

            for (ResultEvaluate prediction : resultEvaluate) {
                float pred = prediction.predDir();
                float real = prediction.realDir();

                // 1. FILTRO DE CONFIANZA:
                // Solo evaluamos si el valor absoluto de la predicción supera el mínimo (ej: 0.7)
                if (Math.abs(pred) >= minConfidence) {
                    total++;

                    // 2. VERIFICACIÓN DE ACIERTO:
                    // Debe coincidir el signo y el real debe haber superado el umbral
                    if (pred > 0 && real > threshold) {
                        hits++; // Acierto Long seguro
                    } else if (pred < 0 && real < -threshold) {
                        hits++; // Acierto Short seguro
                    }
                }
            }
            return total > 0 ? ((float) hits / total) * 100f : 0f;
        }

        private void computeDir(int[] hits, @NotNull EngineUtils.ResultEvaluate prediction) {
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

    public record ResultEvaluate(
            float predTP, float predSL, float predDir,
            float realTP, float realSL, float realDir,
            float predLongSing, float realLongSing,
            float predNeutralSing, float realNeutralSing,
            float predShortSing, float realShortSing,
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

        public Trading.DireccionOperation getRealDirection() {
            return getDireccion(realLongSing, realNeutralSing, realShortSing);
        }

        public Trading.DireccionOperation getPredDirection() {
            return getDireccion(predLongSing, predNeutralSing, predShortSing);
        }
    }

    public static Trading.DireccionOperation getDireccion(float... sings) {

        int maxIndex = 0;
        float maxValue = sings[0];

        for (int i = 1; i < sings.length; i++) {
            if (sings[i] > maxValue) {
                maxValue = sings[i];
                maxIndex = i;
            }
        }
        return switch (maxIndex) {
            case 0 -> Trading.DireccionOperation.LONG;
            case 1 -> Trading.DireccionOperation.NEUTRAL;
            case 2 -> Trading.DireccionOperation.SHORT;
            default -> throw new IllegalStateException("Unexpected value: " + maxIndex);
        };
    }

    private static float[][][] extractSplit3D(float[][][] data, int startIndex, int endIndex) {
        int resultSize = endIndex - startIndex;
        float[][][] result = new float[resultSize][][];
        System.arraycopy(data, startIndex, result, 0, resultSize);
        return result;
    }

    private static float[][] extractSplit2D(float[][] data, int startIndex, int endIndex) {
        int resultSize = endIndex - startIndex;
        float[][] result = new float[resultSize][];
        System.arraycopy(data, startIndex, result, 0, resultSize);
        return result;
    }

    public static Pair<float[][][], float[][]> getSingleSplitWithLabels(
            float[][][] xData, float[][] yData, int numSplits, int splitIndex) {

        // Verificar que ambos arrays tienen la misma longitud
        if (xData.length != yData.length) {
            throw new IllegalArgumentException(
                    "xData y yData deben tener la misma longitud. xData: " +
                            xData.length + ", yData: " + yData.length
            );
        }

        Pair<Integer, Integer> indices = calculateSplitIndices(xData.length, numSplits, splitIndex);

        float[][][] xSplit = extractSplit3D(xData, indices.getKey(), indices.getValue());
        float[][] ySplit = extractSplit2D(yData, indices.getKey(), indices.getValue());

        return new Pair<>(xSplit, ySplit);
    }

    private static Pair<Integer, Integer> calculateSplitIndices(int totalSamples, int numSplits, int splitIndex) {
        if (numSplits <= 0) {
            throw new IllegalArgumentException("numSplits debe ser mayor a 0");
        }

        if (splitIndex < 0 || splitIndex >= numSplits) {
            throw new IllegalArgumentException("splitIndex debe estar entre 0 y " + (numSplits - 1));
        }

        int splitSize = totalSamples / numSplits;
        int remainder = totalSamples % numSplits;

        int startIndex;
        int endIndex;

        if (splitIndex < remainder) {
            // Este split tiene un elemento extra
            startIndex = splitIndex * (splitSize + 1);
            endIndex = startIndex + (splitSize + 1);
        } else {
            // Este split tiene tamaño normal
            startIndex = remainder * (splitSize + 1) + (splitIndex - remainder) * splitSize;
            endIndex = startIndex + splitSize;
        }

        // Asegurar que no excedamos los límites
        endIndex = Math.min(endIndex, totalSamples);

        return new Pair<>(startIndex, endIndex);
    }

    private static final ConcurrentHashMap<Float, NDArray> cacheFloatsStatic = new ConcurrentHashMap<>();

    public static NDArray floatToNDArray(float value, NDManager manager) {
        NDArray ndArray = cacheFloatsStatic.computeIfAbsent(value, manager::create);
        ndArray.detach();
        return ndArray;
    }

    public static void clearCacheFloats() {
        for (NDArray ndArray : cacheFloatsStatic.values()) {
            ndArray.close();
        }
        cacheFloatsStatic.clear();
    }
}
