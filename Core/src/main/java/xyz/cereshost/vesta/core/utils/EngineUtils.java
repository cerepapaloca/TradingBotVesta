package xyz.cereshost.vesta.core.utils;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.core.trading.Trading;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

import static xyz.cereshost.vesta.core.engine.PredictionEngine.THRESHOLD_RELATIVE;

@UtilityClass
public class EngineUtils {

    public static final float MIN_DIRECTION_RATIO = 2.0f;
    public static final float THRESHOLD_RATIO = 0.5f;

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

    public static int directionFromRatio(float ratio) {
        return PredictionUtils.directionFromRatioThreshold(ratio, THRESHOLD_RATIO);
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

    public static void cleanNaNValues(float[] @NotNull [] array) {
        for (float[] sample : array) {
            for (int k = 0; k < sample.length; k++) {
                if (Float.isNaN(sample[k]) || Float.isInfinite(sample[k])) {
                    sample[k] = 0f;
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
        int targetCols = (int) y_test.getShape().get(1); // Dinamico: usualmente 5
        int predCols = 5; // Formato Vesta: 5 salidas

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

                    // Extraer Target: output0 (upMove) y output1 (downMove)
                    float rawRealUp = yTestFlat[targetIdx];
                    float rawRealDown = yTestFlat[targetIdx + 1];

                    // Extraer Prediccion
                    float rawPredUp = yPredFlat[predIdx];
                    float rawPredDown = yPredFlat[predIdx + 1];

                    // 5. Desnormalizacion (outputs 0 y 1) en porcentaje
                    float[][] denormTarget = yNormalizer.inverseTransform(new float[][]{{rawRealUp, rawRealDown, 0, 0, 0}});
                    float[][] denormPred = yNormalizer.inverseTransform(new float[][]{{rawPredUp, rawPredDown, 0, 0, 0}});

                    float realUpPercent = Math.max(0f, denormTarget[0][0] * 100.0f);
                    float realDownPercent = Math.max(0f, denormTarget[0][1] * 100.0f);
                    float predUpPercent = Math.max(0f, denormPred[0][0] * 100.0f);
                    float predDownPercent = Math.max(0f, denormPred[0][1] * 100.0f);

                    float rawRealDir = ratioFromMoves(realUpPercent, realDownPercent);
                    float predDirection = ratioFromMoves(predUpPercent, predDownPercent);

                    // 6. Acumular metricas
                    totalMaeUP += Math.abs(realUpPercent - predUpPercent);
                    totalMaeDOWN += Math.abs(realDownPercent - predDownPercent);
                    totalMaeDir += Math.abs(rawRealDir - predDirection);

                    // 7. Guardar resultado individual
                    allResults.add(new ResultEvaluate(
                            predUpPercent, predDownPercent, predDirection,
                            realUpPercent, realDownPercent, rawRealDir,
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

        return new ResultsEvaluate("VestaIA", avgMaeUP, avgMaeDOWN, allResults);
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
            return total > 0 ? ((float) hits / total) *100 : -1;
        }

        public float hitRateAdvanced() {
            int hits = 0;
            int total = 0;

            for (ResultEvaluate prediction : resultEvaluate) {
                int realDir = directionFromRatio(prediction.getRealDir());
                int predDir = directionFromRatio(prediction.getPredDir());
                if (realDir == predDir) {
                    hits++;
                }
                total++;
            }

            return total > 0 ? ((float) hits / total) * 100 : -1;
        }

        public float hitRateSafe() {
            int hits = 0;
            int fails = 0;

            float threshold = THRESHOLD_RATIO;

            for (ResultEvaluate prediction : resultEvaluate) {
                float pred = prediction.getPredDir();
                float real = prediction.getRealDir();
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
            return total > 0 ? ((float) hits / total) * 100f : -1f;
        }

        public int @NotNull [] hitRateLongOneFloat() {
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.getRealDir() > THRESHOLD_RATIO){
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int @NotNull [] hitRateShortOneFloat() {
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.getRealDir() < -THRESHOLD_RATIO){
                    computeDir(hits, prediction);
                }
            }
            return hits;
        }

        public int @NotNull [] hitRateNeutralOneFloat() {
            int[] hits = new int[3];
            for (ResultEvaluate prediction : resultEvaluate){
                if (prediction.getRealDir() > -THRESHOLD_RATIO && prediction.getRealDir() < THRESHOLD_RATIO) {
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
                float pred = prediction.getPredDir();
                float real = prediction.getRealDir();

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
            boolean signalLong = prediction.getPredDir() > THRESHOLD_RATIO;
            boolean signalShort = prediction.getPredDir() < -THRESHOLD_RATIO;
            if (signalLong) {
                hits[0]++; // Long
            } else if (signalShort) {
                hits[1]++; // Short
            } else {
                hits[2]++; // Neutral
            }
        }

    }

    private static float ratioFromMoves(float upMove, float downMove) {
        float maxMove = Math.max(upMove, downMove);
        if (!Float.isFinite(maxMove) || maxMove <= 0f) return 0f;
        float ratio = (upMove - downMove) / maxMove;
        return PredictionUtils.clampRatio(ratio);
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    public static class ResultEvaluate extends PredictionEngine.PredictionResult {
        private final float predDir;
        private final float realTP;
        private final float realSL;
        private final float realDir;
        private final long timestamp;

        public ResultEvaluate(
                float predTP, float predSL, float predDir,
                float realTP, float realSL, float realDir,
                long timestamp
        ) {
            super(0d, predTP, predSL, 0d, 0d, Math.abs(predDir), directionFromRatio(predDir));
            this.predDir = predDir;
            this.realTP = realTP;
            this.realSL = realSL;
            this.realDir = realDir;
            this.timestamp = timestamp;
        }

        public float lsDiff() {
            return realSL - (float) getSlPrice();
        }

        public float tpDiff() {
            return realTP - (float) getTpPrice();
        }

        public float dirDiff() {
            return realDir - predDir;
        }

        public Trading.DireccionOperation getRealDirection() {
            return directionToOperation(directionFromRatio(realDir));
        }

        public Trading.DireccionOperation getPredDirection() {
            return directionToOperation(directionFromRatio(predDir));
        }
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

    public static Pair<float[][][], float[][]> getSingleSplitWithLabels(Pair<float[][][], float[][]> pair, int numSplits, int splitIndex){
        return getSingleSplitWithLabels(pair.getKey(), pair.getValue(), numSplits, splitIndex);
    }

    public static Pair<float[][][], float[][]> getSingleSplitWithLabels( float[][][] xData, float[][] yData, int numSplits, int splitIndex) {

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


    public void summary(Block modelo) {
        System.out.println("-------------------------------------------------------------------");
        System.out.printf("%-35s %-15s %-15s\n", "Capa (Tipo)", "Forma Salida", "Parámetros #");
        System.out.println("===================================================================");

        long totalParametros = 0;

        // getChildren() nos permite ver los bloques que componen el modelo
        for (Pair<String, Block> par : modelo.getChildren()) {
            String nombreCapa = par.getKey();
            Block capa = par.getValue();
            String tipoCapa = capa.getClass().getSimpleName();

            long parametrosCapa = 0;

            // Recolectamos los parámetros específicos de esta capa
            for (Parameter param : capa.getParameters().values()) {
                if (param.isInitialized()) {
                    parametrosCapa += param.getArray().getShape().size();
                }
            }
            totalParametros += parametrosCapa;

            // Formateamos el nombre para que se vea limpio (Ej: "Linear_1 (Linear)")
            String nombreMostrado = nombreCapa + " (" + tipoCapa + ")";

            // La forma de salida se marca como Dinámica por la naturaleza de DJL
            // a menos que uses un ShapeRecordingNDManager durante un forward pass.
            String formaSalida = "Dinámica";

            Vesta.info("%-35s %-15s %-15,d", nombreMostrado, formaSalida, parametrosCapa);
        }

        Vesta.info("===================================================================");
        Vesta.info("Total de parámetros: %,d\n", totalParametros);
        Vesta.info("-------------------------------------------------------------------");
    }
}
