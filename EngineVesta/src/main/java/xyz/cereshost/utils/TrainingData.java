package xyz.cereshost.utils;


import ai.djl.util.Pair;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.io.IOdata;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.function.BiFunction;

@SuppressWarnings("DataFlowIssue")
@Getter
@Setter
public class TrainingData {

    private static final int INDEX_FOR_VALIDATION = 0;
    private static final int INDEX_FOR_TEST = 1;
    private static final int SPLITS_PAIRS = 1;

    private final boolean loadInRam;
    private final int samplesSize;
    private final int features;
    private final int lookback;
    private final int yCols;

    @Nullable
    private Pair<float[][][], float[][]> pair;
    @Nullable
    private List<Path> files;

    @Getter(AccessLevel.NONE)
    private int testSize = -1;

    public TrainingData(@NotNull Pair<float[][][], float[][]> pair) {
        this.loadInRam = true;
        this.pair = pair;
        this.samplesSize = pair.getKey().length;
        this.lookback = pair.getKey()[0].length;
        this.features = pair.getKey()[0][0].length;
        this.yCols = pair.getValue()[0].length;
    }

    public TrainingData(@NotNull List<Path> files, int samplesTotal, int lookback, int features, int ycols) {
        this.loadInRam = false;
        this.files = files;
        this.samplesSize = samplesTotal;
        this.lookback = lookback;
        this.features = features;
        this.yCols = ycols;
    }

    public long getSampleSize(){
        return samplesSize;
    }

    public int getTestSize(){
        if (testSize != -1){
            return testSize;
        }
        if (loadInRam) {
            testSize = (int) (samplesSize * 0.15);
            return testSize;
        }else {
            try {
                testSize = IOdata.loadTrainingCache(files.get(INDEX_FOR_TEST)).getKey().length;
                return testSize;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Getter(AccessLevel.NONE)
    private int valSize = -1;

    public int getValSize(){
        if (valSize != -1){
            return valSize;
        }
        if (loadInRam) {
            valSize = (int) Math.min(samplesSize *0.15, 70_000);
            return valSize;
        }else {
            try {
                valSize = IOdata.loadTrainingCache(files.get(INDEX_FOR_VALIDATION)).getKey().length;
                return valSize;
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    @Getter(AccessLevel.NONE)
    private int trainSize = -1;

    public long getTrainSize(){
        if (trainSize != -1){
            return trainSize;
        }
        trainSize = samplesSize - getValSize() - getTestSize();
        return trainSize;
    }


    private Pair<float[][][], float[][]> trainNormalize;
    private Pair<float[][][], float[][]> valNormalize;
    private Pair<float[][][], float[][]> testNormalize;

    private XNormalizer xNormalizer;
    private YNormalizer yNormalizer;

    public void prepareNormalize(){
        if (loadInRam){
            computeNormalizeFromRAM();
        }else {
            computeNormalizeFromROM();
        }
    }


    public Pair<float[][][], float[][]> getTrainNormalize(int max, int idx) {
        if (loadInRam){
            return EngineUtils.getSingleSplitWithLabels(trainNormalize.getKey(), trainNormalize.getValue(), max, idx % max);
        }else {
            List<Path> trainingList = files.subList(2, files.size());
            if (trainingList.isEmpty()) {
                throw new IllegalStateException("No hay data de entrenamiento para cargar.");
            }
            return getPairNormalizeFromDisk(trainingList.get(idx % trainingList.size()));
        }
    }

    public Pair<float[][][], float[][]> getValNormalize() {
        if (loadInRam){
            return valNormalize;
        }else {
            return getPairNormalizeFromDisk(files.get(INDEX_FOR_VALIDATION));
        }
    }

    public Pair<float[][][], float[][]> getTestNormalize() {
        if (loadInRam){
            return testNormalize;
        }else {
            return getPairNormalizeFromDisk(files.get(INDEX_FOR_TEST));
        }
    }

    public void closePosTraining(){
        trainNormalize = null;
        valNormalize = null;
        pair = null;
        pairsLoaded.clear();
    }

    public void closeAll(){
        trainNormalize = null;
        valNormalize = null;
        testNormalize = null;
        pair = null;
        pairsLoaded.clear();
    }

    private @NotNull Pair<float[][][], float[][]> getPairNormalizeFromDisk(@Nullable Path files) {

        try {
            Pair<float[][][], float[][]> pair = IOdata.loadTrainingCache(files);
            EngineUtils.cleanNaNValues(pair.getKey());
            EngineUtils.cleanNaNValues(pair.getValue());
            float[][][] x = xNormalizer.transform(pair.getKey());
            float[][] y = yNormalizer.transform(pair.getValue());
            return new Pair<>(x, y);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void computeNormalizeFromROM(){
        List<Path> trainingList = files.subList(2, files.size());
        if (trainingList.isEmpty()) {
            throw new IllegalStateException("No hay data de entrenamiento para normalizar.");
        }
        Vesta.info("ℹ️ Iniciando Normalización por cache");
        int trainSamples = (int) getTrainSize();
        float[][][] X_final = new float[trainSamples][lookback][features];
        float[][] y_final = new float[trainSamples][yCols];

        List<Future<TrainingChunk>> futures = new ArrayList<>();

        for (int i = 0; i < trainingList.size(); i++) {
            final int index = i;
            final Path path = trainingList.get(i);

            futures.add(VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> {
                Pair<float[][][], float[][]> pair = IOdata.loadTrainingCache(path);
                return new TrainingChunk(index, pair.getKey(), pair.getValue());
            }));
        }


        List<TrainingChunk> chunks = new ArrayList<>();
        for (Future<TrainingChunk> f : futures) {
            try {
                TrainingChunk chunk = f.get();
                chunks.add(chunk);
                Vesta.info("📀 (%d/%d) Datos cargado de disco", chunk.index() +1, trainingList.size());
            } catch (ExecutionException | InterruptedException e) {
                throw new RuntimeException("Error cargando datos", e.getCause());
            }
        }

        chunks.sort(Comparator.comparingInt(c -> c.index));
        Vesta.info("🔗 Copiando Arrays");
        int currentIdx = 0;
        for (TrainingChunk c : chunks) {
            int len = c.x.length;
            System.arraycopy(c.x, 0, X_final, currentIdx, len);
            System.arraycopy(c.y, 0, y_final, currentIdx, len);
            currentIdx += len;
        }
        if (currentIdx != trainSamples) {
            X_final = Arrays.copyOf(X_final, currentIdx);
            y_final = Arrays.copyOf(y_final, currentIdx);
        }
        Vesta.info("📦 Normalizando X");
        XNormalizer xNormalizer = new XNormalizer();
        EngineUtils.cleanNaNValues(X_final);
        EngineUtils.cleanNaNValues(y_final);
        xNormalizer.fit(X_final);
        this.xNormalizer = xNormalizer;
        Vesta.info("📦 Normalizando Y");
        YNormalizer yNormalizer = new YNormalizer();
        yNormalizer.fit(y_final);
        this.yNormalizer = yNormalizer;
        ChartUtils.showTPSLDistribution("Distribución de los datos", y_final, "???");
        Vesta.info("✅ Normalizando Terminada");
    }

    private record TrainingChunk(int index, float[][][] x, float[][] y) {}

    private void computeNormalizeFromRAM(){
        try{
            // Verificar NaN sólo en arrays normalizados (por si acaso)
            EngineUtils.cleanNaNValues(pair.getKey());
            EngineUtils.cleanNaNValues(pair.getValue());
            BiFunction<long[], long[], float[][][]> slice3D = getSlice3D(pair.getKey());
            int trainSizeLocal = (int) getTrainSize();
            int valSizeLocal = getValSize();
            splitSample split = getSplitSample(slice3D, trainSizeLocal, valSizeLocal, samplesSize, pair.getValue());
            Normalize result = getNormalize(split);


            trainNormalize = new Pair<>(result.getX_train_norm(), result.getY_train_norm());
            valNormalize = new Pair<>(result.getX_val_norm(), result.getY_val_norm());
            testNormalize = new Pair<>(result.getX_test_norm(), result.getY_test_norm());
            xNormalizer = result.getXNormalizer();
            yNormalizer = result.getYNormalizer();
            pair = null;
        }catch (InterruptedException | ExecutionException e){
            e.printStackTrace();
        }
    }

    private int index = 0;
    private int maxLoaded = 1;
    private ArrayDeque<CompletableFuture<Pair<float[][][], float[][]>>> pairsLoaded = new ArrayDeque<>();
    @Nullable
    private Random random = null;
    private AutoStopListener autoStopListener = null;
    private ModeData modeData = null;


    public void preLoad(int amount, ModeData mode){
        this.modeData = mode;
        maxLoaded = amount;
        if (Objects.requireNonNull(mode) == ModeData.RAMDOM) {
            this.random = new Random();
        }
        if (files != null){
            List<Path> trainingList = files.subList(2, files.size());
            for (int i = 0; i < amount; i++){
                int j = i;
                pairsLoaded.add(CompletableFuture.supplyAsync(() ->
                    getPairNormalizeFromDisk(trainingList.get(j))
                ));
            }
        }
    }

    public Pair<float[][][], float[][]> nextData(){
        if (modeData == null){
            throw new IllegalStateException("ModeData is null");
        }

        switch (modeData){
            case RAMDOM -> index = Math.abs(random.nextInt());
            case SECUENCIAL -> index++;
        }
        Pair<float[][][], float[][]> result;
        if (loadInRam){
            result = EngineUtils.getSingleSplitWithLabels(trainNormalize.getKey(), trainNormalize.getValue(), SPLITS_PAIRS, index % SPLITS_PAIRS);
        }else {
            try {
                result = pairsLoaded.pollFirst().get();
                List<Path> trainingList = files.subList(2, files.size());
                while (pairsLoaded.size() < maxLoaded) {
                    pairsLoaded.add(CompletableFuture.supplyAsync(() ->
                            getPairNormalizeFromDisk(trainingList.get(index % trainingList.size())), VestaEngine.EXECUTOR_TRAINING)
                    );
                }
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }

        return result;
    }

    public static @NotNull BiFunction<long[], long[], float[][][]> getSlice3D(float[][][] xCombined) {
        final float[][][] finalXCombined = xCombined;
        return (long[] range, long[] dummy) -> {
            long start = range[0];
            long end = range[1];
            int len = (int) (end - start);
            float[][][] out = new float[len][][];
            for (long i = start; i < end; i++) {
                out[(int) (i - start)] = finalXCombined[(int) i];
            }
            return out;
        };
    }

    public static @NotNull splitSample getSplitSample(BiFunction<long[], long[], float[][][]> slice3D, long trainSize, long valSize, long samples, float[][] yCombined) throws InterruptedException, ExecutionException {
        // Crear splits en arrays Java antes de normalizar
        CompletableFuture<float[][][]> X_train_arr = CompletableFuture.supplyAsync(() -> slice3D.apply(new long[]{0, trainSize}, null), VestaEngine.EXECUTOR_AUXILIAR_BUILD);
        CompletableFuture<float[][][]> X_val_arr =   CompletableFuture.supplyAsync(() -> slice3D.apply(new long[]{trainSize, trainSize + valSize}, null), VestaEngine.EXECUTOR_AUXILIAR_BUILD);
        CompletableFuture<float[][][]> X_test_arr =  CompletableFuture.supplyAsync(() -> slice3D.apply(new long[]{trainSize + valSize, samples}, null), VestaEngine.EXECUTOR_AUXILIAR_BUILD);

        CompletableFuture<float[][]> y_train_arr = CompletableFuture.supplyAsync(() -> java.util.Arrays.copyOfRange(yCombined, 0, (int) trainSize), VestaEngine.EXECUTOR_AUXILIAR_BUILD);
        CompletableFuture<float[][]> y_val_arr =   CompletableFuture.supplyAsync(() -> java.util.Arrays.copyOfRange(yCombined,(int) trainSize, (int) (trainSize + valSize)), VestaEngine.EXECUTOR_AUXILIAR_BUILD);
        CompletableFuture<float[][]> y_test_arr =  CompletableFuture.supplyAsync(() -> java.util.Arrays.copyOfRange(yCombined,(int)  (trainSize + valSize),(int) samples), VestaEngine.EXECUTOR_AUXILIAR_BUILD);

        return new splitSample(X_train_arr.get(), X_val_arr.get(), X_test_arr.get(), y_train_arr.get(), y_val_arr.get(), y_test_arr.get());
    }

    public record splitSample(float[][][] X_train_arr, float[][][] X_val_arr, float[][][] X_test_arr, float[][] y_train_arr, float[][] y_val_arr, float[][] y_test_arr) {
    }

    public static @NotNull Normalize getNormalize(splitSample split) throws InterruptedException, ExecutionException {

        float[][][] X_train_arr = split.X_train_arr;
        float[][][] X_val_arr = split.X_val_arr;
        float[][][] X_test_arr = split.X_test_arr;
        float[][] y_train_arr = split.y_train_arr;
        float[][] y_val_arr = split.y_val_arr;
        float[][] y_test_arr = split.y_test_arr;
        // Normalizadores: FIT sólo con TRAIN
        XNormalizer xNormalizer = new XNormalizer();
        xNormalizer.fit(X_train_arr); // fit con train solamente

        YNormalizer yNormalizer = new YNormalizer();
        yNormalizer.fit(y_train_arr); // fit con train solamente

        // Transformar train/val/test
        CompletableFuture<float[][][]> X_train_norm = new CompletableFuture<>();
        CompletableFuture<float[][][]> X_val_norm = new CompletableFuture<>();
        CompletableFuture<float[][][]> X_test_norm = new CompletableFuture<>();

        VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> X_train_norm.complete(xNormalizer.transform(X_train_arr)));
        VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> X_val_norm.complete(xNormalizer.transform(X_val_arr)));
        VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> X_test_norm.complete(xNormalizer.transform(X_test_arr)));

        CompletableFuture<float[][]> y_train_norm = new CompletableFuture<>();
        CompletableFuture<float[][]> y_val_norm = new CompletableFuture<>();
        CompletableFuture<float[][]> y_test_norm = new CompletableFuture<>();

        VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> y_train_norm.complete(yNormalizer.transform(y_train_arr)));
        VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> y_val_norm.complete(yNormalizer.transform(y_val_arr)));
        VestaEngine.EXECUTOR_AUXILIAR_BUILD.submit(() -> y_test_norm.complete(yNormalizer.transform(y_test_arr)));
        return new Normalize(xNormalizer, yNormalizer, X_train_norm.get(), X_val_norm.get(), X_test_norm.get(), y_train_norm.get(), y_val_norm.get(), y_test_norm.get());
    }

    public IOdata.CacheProperties getCacheProperties(List<String> market) {
        return new IOdata.CacheProperties(lookback, features, yCols, market, Main.MAX_MONTH_TRAINING, samplesSize);
    }

    @Getter
    @Data
    public static final class Normalize {
        private final XNormalizer xNormalizer;
        private final YNormalizer yNormalizer;
        private final float[][][] X_train_norm;
        private final float[][][] X_val_norm;
        private final float[][][] X_test_norm;
        private final float[][] y_train_norm;
        private final float[][] y_val_norm;
        private final float[][] y_test_norm;

        public void clearX_train_norm() {
            Arrays.fill(X_train_norm, null);
        }
        private void clearX_val_norm() {
            Arrays.fill(X_val_norm, null);
        }
        public void clearX_test_norm() {
            Arrays.fill(this.X_test_norm, null);
        }
        private void clearY_train_norm() {
            Arrays.fill(this.y_train_norm, null);
        }
        private void clearY_val_norm() {
            Arrays.fill(this.y_val_norm, null);
        }
        private void clearY_test_norm() {
            Arrays.fill(this.y_test_norm, null);
        }
    }

    public enum ModeData{
        SECUENCIAL,
        RAMDOM,
    }

}
