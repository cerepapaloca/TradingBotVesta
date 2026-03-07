package xyz.cereshost.vesta.core.utils;


import ai.djl.util.Pair;
import lombok.AccessLevel;
import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.Main;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.engine.VestaEngine;
import xyz.cereshost.vesta.core.io.IOdata;

import java.io.IOException;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiFunction;

@SuppressWarnings({"DataFlowIssue", "UnusedAssignment"})
@Getter
@Setter
public class TrainingData {

    private static final int INDEX_FOR_VALIDATION = 0;
    private static final int INDEX_FOR_TEST = 1;


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
        this.loadInRam = files.isEmpty();
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

    public Pair<float[][][], float[][]> getTestNormalize() {
        if (loadInRam){
            return testNormalize;
        }else {
            return EngineUtils.getSingleSplitWithLabels(getPairNormalizeFromDisk(files.get(INDEX_FOR_TEST)), splitParts, 0);
        }
    }

    public void closePosTraining(){
        trainNormalize = null;
        valNormalize = null;
        pair = null;
        pairsLoaded.clear();
        splitQueue.clear();
    }

    public void closeAll(){
        trainNormalize = null;
        valNormalize = null;
        testNormalize = null;
        pair = null;
        pairsLoaded.clear();
        splitQueue.clear();
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
        Vesta.info("Iniciando Normalizacion por cache (streaming)");
        XNormalizer xNormalizer = new XNormalizer();
        YNormalizer yNormalizer = new YNormalizer();

        AtomicInteger idx = new AtomicInteger();
        Queue<CompletableFuture<Pair<float[][][], float[][]>>> pairFutures = new LinkedList<>();
        for (Path path : trainingList) {
            pairFutures.add(CompletableFuture.supplyAsync(() -> {
                Pair<float[][][], float[][]> pair;
                try {

                    pair = IOdata.loadTrainingCache(path);
                    EngineUtils.cleanNaNValues(pair.getKey());
                    EngineUtils.cleanNaNValues(pair.getValue());

                    idx.getAndIncrement();
                    Vesta.info("(%d/%d) Datos cargado de disco", idx.get(), trainingList.size());
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
                return pair;
            }, VestaEngine.EXECUTOR_READ_CACHE_BUILD));
        }

        for (int i = 0; i < trainingList.size(); i++) {
            CompletableFuture<Pair<float[][][], float[][]>> pair = pairFutures.poll();
            try {
                Pair<float[][][], float[][]> p = pair.get();

                xNormalizer.partialFit(p.getKey());
                yNormalizer.partialFit(p.getValue());

                Arrays.fill(p.getKey(), null);
                Arrays.fill(p.getValue(), null);
                p = null;
                pair = null;
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }

        }

        Vesta.info("Normalizando X (streaming)");
        xNormalizer.finishFit();
        this.xNormalizer = xNormalizer;
        Vesta.info("Normalizando Y (streaming)");
        yNormalizer.finishFit();
        this.yNormalizer = yNormalizer;
        Vesta.info("Normalizando Terminada");
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

    private int maxLoaded = 1;
    private ArrayDeque<CompletableFuture<Pair<float[][][], float[][]>>> pairsLoaded = new ArrayDeque<>();
    private ArrayDeque<Pair<float[][][], float[][]>> splitQueue = new ArrayDeque<>();
    private int splitParts = 1;
    @Nullable
    private Random random = null;
    private AutoStopListener autoStopListener = null;
    private ModeData modeData = null;

    public void preLoad(int amount, ModeData mode, int splitParts){
        this.modeData = mode;
        maxLoaded = amount;
        this.splitParts = Math.max(1, splitParts);
        splitQueue.clear();
        if (Objects.requireNonNull(mode) == ModeData.RAMDOM) {
            this.random = new Random();
        }
        if (!loadInRam){
            List<Path> trainingList = files.subList(2, files.size());
            for (int i = 0; i < amount; i++){
                int j = i;
                pairsLoaded.add(CompletableFuture.supplyAsync(() ->
                    getPairNormalizeFromDisk(trainingList.get(j))
                ));
            }
        }
    }

    private int indexValidation = 0;

    @NotNull
    public Pair<float[][][], float[][]> nextValidationData() {
        if (loadInRam){
            return valNormalize;
        }else {
            //indexValidation++;
            if (valNormalize == null) valNormalize = getPairNormalizeFromDisk(files.get(INDEX_FOR_VALIDATION));
            return EngineUtils.getSingleSplitWithLabels(valNormalize, splitParts, indexValidation % splitParts);
        }
    }

    private int indexTrading = 0;

    public Pair<float[][][], float[][]> nextTrainingData(){
        if (modeData == null){
            throw new IllegalStateException("ModeData is null");
        }

        if (!splitQueue.isEmpty()) {
            return splitQueue.pollFirst();
        }

        switch (modeData){
            case RAMDOM -> indexTrading = Math.abs(random.nextInt());
            case SECUENCIAL -> indexTrading++;
        }
        Pair<float[][][], float[][]> result;
        if (loadInRam){
            result = EngineUtils.getSingleSplitWithLabels(trainNormalize.getKey(), trainNormalize.getValue(), splitParts, indexTrading % splitParts);
        }else {
            try {
                result = pairsLoaded.pollFirst().get();
                List<Path> trainingList = files.subList(2, files.size());
                while (pairsLoaded.size() < maxLoaded) {
                    pairsLoaded.add(CompletableFuture.supplyAsync(() ->
                            getPairNormalizeFromDisk(trainingList.get(indexTrading % trainingList.size())), VestaEngine.EXECUTOR_TRAINING)
                    );
                }
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException(e);
            }
        }

        if (splitParts <= 1) {
            return result;
        }else {
            List<Pair<float[][][], float[][]>> splits = new ArrayList<>(splitParts);
            for (int i = 0; i < splitParts; i++) {
                splits.add(EngineUtils.getSingleSplitWithLabels(result, splitParts, i));
            }
            if (modeData == ModeData.RAMDOM) {
                if (random == null) {
                    random = new Random();
                }
                Collections.shuffle(splits, random);
            }
            splitQueue.addAll(splits);
            return splitQueue.pollFirst();
        }
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


