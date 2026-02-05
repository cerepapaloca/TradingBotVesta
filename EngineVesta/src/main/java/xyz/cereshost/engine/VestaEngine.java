package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.basicmodelzoo.basic.Mlp;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.recurrent.GRU;
import ai.djl.pytorch.engine.PtModel;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.Main;
import xyz.cereshost.blocks.BlockUtils;
import xyz.cereshost.blocks.Softmax;
import xyz.cereshost.blocks.TemporalTransformerBlock;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.XNormalizer;
import xyz.cereshost.builder.YNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.io.IOdata;
import xyz.cereshost.metrics.BinaryDirectionEvaluator;
import xyz.cereshost.metrics.DirectionAccuracyEvaluator;
import xyz.cereshost.metrics.MAEEvaluator;
import xyz.cereshost.metrics.MetricsListener;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiFunction;

public class VestaEngine {

    public static final int LOOK_BACK = 60;
    public static final int EPOCH = 4;
    public static final int EPOCH_SUB = 15;
    public static final int SPLIT_DATASET = 1;

    @Getter
    private static NDManager rootManager;

    public static final ExecutorService EXECUTOR = Executors.newCachedThreadPool();
    public static final ExecutorService EXECUTOR_BUILD = Executors.newScheduledThreadPool(6);
    public static final ExecutorService EXECUTOR_TRAINING = Executors.newScheduledThreadPool(8);
    /**
     * Entrena un modelo con múltiples símbolos combinados
     */
    @SuppressWarnings("UnusedAssignment")
    public static @Nullable TrainingTestsResults trainingModel(@NotNull List<String> symbols) throws TranslateException, IOException, InterruptedException, ExecutionException {
        ai.djl.engine.Engine torch = ai.djl.engine.Engine.getEngine("PyTorch");
        if (torch == null) {
            Vesta.error("PyTorch no está disponible. Engines disponibles:");
            for (String engine : ai.djl.engine.Engine.getAllEngines()) {
                Vesta.error("  - " + engine);
            }
            throw new RuntimeException("PyTorch engine no encontrado");
        }

        Device device = Engine.getInstance().getDevices()[0];
        Vesta.info("Usando dispositivo: " + device);
        Vesta.info("Entrenando con " + symbols.size() + " símbolos: " + symbols);
        CompletableFuture<Market> futureMarket = new CompletableFuture<>();

        TrainingConfig config = new DefaultTrainingConfig(new VestaLoss("WeightedL2"))
                .optOptimizer(Optimizer.adamW()
                        .optLearningRateTracker(Tracker.cosine()
                                .setBaseValue(0.000_000_3f)
                                .optFinalValue(0.000_000_05f)
                                .setMaxUpdates((int) (EPOCH*EPOCH_SUB*((double)Main.MAX_MONTH_TRAINING /SPLIT_DATASET)))
                                .build())
//                        .optLearningRateTracker(Tracker.fixed(0.003f))
                        .optWeightDecays(0.0f)
                        .optClipGrad(1f)
                        .build())
                .optDevices(Engine.getInstance().getDevices())
                .addEvaluator(new MAEEvaluator())
                .addEvaluator(new BinaryDirectionEvaluator())
                .addEvaluator(new DirectionAccuracyEvaluator())
                .optExecutorService(EXECUTOR_TRAINING)
                .addTrainingListeners(TrainingListener.Defaults.logging())
                .addTrainingListeners(new MetricsListener());

        try (PtModel model = (PtModel) Model.newInstance(Main.NAME_MODEL, device, "PyTorch")) {
            PtNDManager manager = (PtNDManager) model.getNDManager();

            Pair<float[][][], float[][]> combined = BuilderData.fullBuild(symbols,  Main.MAX_MONTH_TRAINING, 1);

            float[][][] xCombined = combined.getKey();
            float[][] yCombined = combined.getValue();

            ChartUtils.showTPSLDistribution("Datos Combinados", yCombined, "Todos");
            ChartUtils.showDirectionDistribution("Datos Combinados", yCombined, "Todos");

            Vesta.info("Datos combinados:");
            Vesta.info("  Total de muestras: " + xCombined.length);
            Vesta.info("  Lookback: " + xCombined[0].length);
            Vesta.info("  Características: " + xCombined[0][0].length);

            // Preparar dimensiones
            long samples = xCombined.length;
            int lookback = xCombined[0].length;
            int features = xCombined[0][0].length;

            // SPLIT (antes de normalizar) -> 70% train, 15% val, 15% test
            long testSize = (long) (samples * 0.15);
            long valSize = (long) Math.min(samples *0.15, 5_000); // para que sume exactamente samples
            long trainSize = samples - valSize - testSize;

            Vesta.info("Split sizes: train=" + trainSize + " val=" + valSize + " test=" + testSize);
            // Borra por que no se va a usar más
            Vesta.MARKETS.clear();
            // Helper local para slice 3D arrays (copia el primer eje [start, end))
            BiFunction<long[], long[], float[][][]> slice3D = getSlice3D(xCombined);
            splitSample split = getSplitSample(slice3D, trainSize, valSize, samples, yCombined);
            combined = null;
            xCombined = null;
            yCombined = null;
            Normalize result = getNormalize(split);
            split = null;
            // Verificar NaN sólo en arrays normalizados (por si acaso)
            EngineUtils.cleanNaNValues(result.X_train_norm());
            EngineUtils.cleanNaNValues(result.X_val_norm());
            EngineUtils.cleanNaNValues(result.X_test_norm());

            // Construir modelo (usa tu método existente)

            model.setBlock(getSequentialBlock());
            // Configuración de entrenamiento (igual a tu código)

            int batchSize = 256;// 256;
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(batchSize, LOOK_BACK, features));


            // Entrenar
            int iteration =  Main.MAX_MONTH_TRAINING/SPLIT_DATASET;
            Vesta.info("Iniciando entrenamiento con " + EPOCH*EPOCH_SUB*iteration + " epochs...");
            rootManager = manager;
            ChunkDataset sampleTraining = computeDataset(EngineUtils.getSingleSplitWithLabels(result.X_train_norm(), result.y_train_norm(), iteration, 0), batchSize, manager);
            ChunkDataset sampleVal = computeDataset(new Pair<>(result.X_train_norm(), result.y_train_norm()), batchSize, manager);

            for (int epoch = 0; epoch < EPOCH; epoch++) {
                for (int idx = 0; idx < iteration; idx++) {
                    final int finalIdx = idx;
                    CompletableFuture<ChunkDataset> sampleTrainingNext = CompletableFuture.supplyAsync(() ->
                            computeDataset(EngineUtils.getSingleSplitWithLabels(result.X_train_norm(), result.y_train_norm(), iteration, (finalIdx + 1) % iteration), batchSize, manager), EXECUTOR_TRAINING);

//                    CompletableFuture<ChunkDataset> sampleValNext = CompletableFuture.supplyAsync(() ->
//                            computeDataset(EngineUtils.getSingleSplitWithLabels(result.X_train_norm(), result.y_train_norm(), iteration, (finalIdx + 1) % iteration), batchSize, manager), EXECUTOR_TRAINING);
                    EasyTrain.fit(trainer, EPOCH_SUB, sampleTraining.dataset(), sampleVal.dataset());
                    NDArray xT = sampleTraining.x();
                    NDArray yT = sampleTraining.y();
//                    NDArray xV = sampleVal.x();
//                    NDArray yV = sampleVal.y();
                    EXECUTOR_TRAINING.submit(() -> {
                        xT.close();
                        yT.close();
                        EngineUtils.clearCacheFloats();
//                        xV.close();
//                        yV.close();
                    });
                    if(!sampleTrainingNext.isDone() /*|| !sampleValNext.isDone()*/) Vesta.warning("Mes no listo procesando...");
                    sampleTraining = sampleTrainingNext.get();
                    //sampleVal = sampleValNext.get();
                }
            }
            sampleVal.x().close();
            sampleVal.y().close();
            sampleTraining.x().close();
            sampleTraining.y().close();

            // Guardar modelo (igual que antes)
            IOdata.saveModel(model);
            IOdata.saveYNormalizer(result.yNormalizer());
            IOdata.saveXNormalizer(result.xNormalizer());

            NDArray X_test  = EngineUtils.concat3DArrayToNDArray(result.X_test_norm(), manager, 1024);
            NDArray y_test  = manager.create(result.y_test_norm());

            // Evaluar en conjunto de test si hay muestras
            if (testSize > 0) {
                Vesta.info("\nEvaluando modelo con Backtest Walk-Forward (15% data)...");

                // 1. Crear instancia temporal de PredictionEngine con los datos recién entrenados
                // Nota: Necesitamos un PredictionEngine para usar el método 'predictForBacktest'
                // Como el modelo (PtModel) ya está en memoria, podemos pasarlo directamente.

                // Necesitamos pasar un Model 'genérico', PtModel hereda de Model
                PredictionEngine predEngine = new PredictionEngine(
                        result.xNormalizer(),
                        result.yNormalizer(),
                        model,
                        lookback,
                        features
                );

                // 2. Ejecutar Backtest para cada símbolo (o solo el primero si combinaste)
                // Como entrenaste combinando símbolos, lo ideal es probar en uno representativo o iterar.
                // Aquí probamos con el primer símbolo de la lista para obtener el ROI
                String testSymbol = symbols.get(0);
                Market market = IOdata.loadMarkets(Main.DATA_SOURCE_FOR_BACK_TEST, symbols.get(0), 1).limit(7);
                EngineUtils.ResultsEvaluate evaluate = EngineUtils.evaluateModel(trainer, X_test, y_test, result.yNormalizer(), 1024);

                BackTestEngine.BackTestResult simResult;
                if (market != null) {
                    simResult = new BackTestEngine(market, predEngine).run();
                } else {
                    Vesta.error("No se encontró mercado para backtest: " + testSymbol);
                    simResult = null;
                }
                manager.close();
                return new TrainingTestsResults(evaluate, simResult);
            }else {
                return null;
            }
        }
    }

    private static @NotNull BiFunction<long[], long[], float[][][]> getSlice3D(float[][][] xCombined) {
        final float[][][] finalXCombined = xCombined;
        BiFunction<long[], long[], float[][][]> slice3D = (long[] range, long[] dummy) -> {
            long start = range[0];
            long end = range[1];
            int len = (int) (end - start);
            float[][][] out = new float[len][][];
            for (long i = start; i < end; i++) {
                out[(int) (i - start)] = finalXCombined[(int) i];
            }
            return out;
        };
        return slice3D;
    }

    private static @NotNull splitSample getSplitSample(BiFunction<long[], long[], float[][][]> slice3D, long trainSize, long valSize, long samples, float[][] yCombined) throws InterruptedException, ExecutionException {
        // Crear splits en arrays Java antes de normalizar
        CompletableFuture<float[][][]> X_train_arr = CompletableFuture.supplyAsync(() -> slice3D.apply(new long[]{0, trainSize}, null), EXECUTOR);
        CompletableFuture<float[][][]> X_val_arr =   CompletableFuture.supplyAsync(() -> slice3D.apply(new long[]{trainSize, trainSize + valSize}, null), EXECUTOR);
        CompletableFuture<float[][][]> X_test_arr =  CompletableFuture.supplyAsync(() -> slice3D.apply(new long[]{trainSize + valSize, samples}, null), EXECUTOR);

        CompletableFuture<float[][]> y_train_arr = CompletableFuture.supplyAsync(() -> java.util.Arrays.copyOfRange(yCombined, 0, (int) trainSize), EXECUTOR);
        CompletableFuture<float[][]> y_val_arr =   CompletableFuture.supplyAsync(() -> java.util.Arrays.copyOfRange(yCombined,(int) trainSize, (int) (trainSize + valSize)), EXECUTOR);
        CompletableFuture<float[][]> y_test_arr =  CompletableFuture.supplyAsync(() -> java.util.Arrays.copyOfRange(yCombined,(int)  (trainSize + valSize),(int) samples), EXECUTOR);

        return new splitSample(X_train_arr.get(), X_val_arr.get(), X_test_arr.get(), y_train_arr.get(), y_val_arr.get(), y_test_arr.get());
    }

    private record splitSample(float[][][] X_train_arr, float[][][] X_val_arr, float[][][] X_test_arr, float[][] y_train_arr, float[][] y_val_arr, float[][] y_test_arr) {
    }

    private static @NotNull Normalize getNormalize(splitSample split) throws InterruptedException, ExecutionException {

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

        EXECUTOR.submit(() -> X_train_norm.complete(xNormalizer.transform(X_train_arr)));
        EXECUTOR.submit(() -> X_val_norm.complete(xNormalizer.transform(X_val_arr)));
        EXECUTOR.submit(() -> X_test_norm.complete(xNormalizer.transform(X_test_arr)));

        CompletableFuture<float[][]> y_train_norm = new CompletableFuture<>();
        CompletableFuture<float[][]> y_val_norm = new CompletableFuture<>();
        CompletableFuture<float[][]> y_test_norm = new CompletableFuture<>();

        EXECUTOR.submit(() -> y_train_norm.complete(yNormalizer.transform(y_train_arr)));
        EXECUTOR.submit(() -> y_val_norm.complete(yNormalizer.transform(y_val_arr)));
        EXECUTOR.submit(() -> y_test_norm.complete(yNormalizer.transform(y_test_arr)));
        return new Normalize(xNormalizer, yNormalizer, X_train_norm.get(), X_val_norm.get(), X_test_norm.get(), y_train_norm.get(), y_val_norm.get(), y_test_norm.get());
    }

    private record Normalize(XNormalizer xNormalizer, YNormalizer yNormalizer, float[][][] X_train_norm, float[][][] X_val_norm, float[][][] X_test_norm, float[][] y_train_norm, float[][] y_val_norm, float[][] y_test_norm) {
    }

    public record TrainingTestsResults(EngineUtils.ResultsEvaluate evaluate, BackTestEngine.BackTestResult backtest) {}

    @SuppressWarnings("DuplicatedCode")
    public static @NotNull SequentialBlock getSequentialBlock() {
        SequentialBlock mainBlock = new SequentialBlock();

        float deltaFloat = 0.7f; // Controla qué tan suave es el centro
        TTLHeader(mainBlock);
        // Branch aggregator: recibirá outputs de TP, SL y DIRECCION (que a su vez es la concatenacion de 3 sub-brazos)
        ParallelBlock branches = new ParallelBlock(list -> {
            NDArray tp = list.get(0).singletonOrThrow();
            NDArray sl = list.get(1).singletonOrThrow();
            NDArray dir = list.get(2).singletonOrThrow();
            return new NDList(
                    NDArrays.concat(new NDList(tp, sl, dir), 1) // axis = 1
            );
        });

        // TP (igual que antes)
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(128).build())
                .add(Dropout.builder().optRate(0.1f).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x.mul(x).add(EngineUtils.floatToNDArray(deltaFloat*deltaFloat, ndArrays.getManager())).sqrt().sub(EngineUtils.floatToNDArray(deltaFloat, ndArrays.getManager())));
                }))
        );

        // SL (igual que antes)
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(128).build())
                .add(Dropout.builder().optRate(0.1f).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x.mul(x).add(EngineUtils.floatToNDArray(deltaFloat*deltaFloat, ndArrays.getManager())).sqrt().sub(EngineUtils.floatToNDArray(deltaFloat, ndArrays.getManager())));
                }))
        );

        // -------------------------
        // DIRECCION: 3 sub-brazos pequeños -> concat -> smoothing -> softmax
        // -------------------------
        // Sub-parallel: cada sub-brazo toma la misma entrada (el embedding de 64/32) y produce 1 escalar.
        ParallelBlock dirSub = new ParallelBlock(subList -> {
            // subList contiene los outputs de cada sub-brazo (cada uno será [B,1])
            NDArray a = subList.get(0).singletonOrThrow();
            NDArray b = subList.get(1).singletonOrThrow();
            NDArray c = subList.get(2).singletonOrThrow();
            // concatenamos a lo largo de la dimensión de features para obtener [B,3]
            return new NDList(NDArrays.concat(new NDList(a, b, c), 1));
        });

        // Sub-brazo 1 (Long)
        dirSub.add(new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x); // ya es escalar por muestra
                }))
        );

        // Sub-brazo 2 (Neutral)
        dirSub.add(new SequentialBlock()
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x);
                }))
        );

        // Sub-brazo 3 (Short)
        dirSub.add(new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x);
                }))
        );

        // Ahora el bloque de Dirección que aplica unos Dense previos, luego el dirSub (que concatena los 3 escalares),
        // luego el "smoothing" que tenías, y por último softmax para obtener probabilidades mutuamente excluyentes.
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(128).build())
                .add(Dropout.builder().optRate(0.2f).build())
                .add(dirSub)
                .add(Softmax.builder().temperature(1).build())
        );

        mainBlock.add(branches);
        return mainBlock;
    }

    private static void RNNHeader(SequentialBlock mainBlock) {
        mainBlock.add(GRU.builder()
                        .setStateSize(256)
                        .setNumLayers(3)
                        .optReturnState(false)
                        .optHasBiases(true) // recomendado
                        .optBidirectional(false)
                        .optBatchFirst(true)
                        .optDropRate(0.3f)
                        .build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray seq = ndArrays.singletonOrThrow();  // [B, T, H]
                    NDArray last = seq.get(":, -1, :");              // [B, H]
                    NDArray mean = seq.mean(new int[]{1});            // [B, H]

                    NDArray combined = NDArrays.concat(
                            new NDList(last, mean),
                            1 // concat en features
                    ); // [B, 2H]
                    return new NDList(combined);
                }))
//                .add(LayerNorm.builder().build())
//                .add(Dropout.builder().optRate(0.08f).build())
                .add(Linear.builder().setUnits(128).build());
    }

    private static void TTLHeader(SequentialBlock mainBlock) {
        mainBlock.add(TemporalTransformerBlock.builder()
                        .setModelDim(128)
                        .setNumHeads(4)
                        .setFeedForwardDim(4*128)
                        .setDropoutRate(0.2f)
                        .setMaxSequenceLength(LOOK_BACK)
                        .setOftenClearCache(250)
                        .build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray seq = ndArrays.singletonOrThrow();  // [B, T, H]
                    NDArray last = seq.get(":, -1, :");              // [B, H]
                    NDArray mean = seq.mean(new int[]{1});            // [B, H]

                    NDArray combined = NDArrays.concat(
                            new NDList(last, mean),
                            1 // concat en features
                    ); // [B, 2H]
                    return new NDList(combined);
                }))
                .add(Linear.builder().setUnits(128).build());
    }

    private static void MLPHeader(SequentialBlock mainBlock) {
        mainBlock.add(new Mlp(BuilderData.getFeatures()*LOOK_BACK, 512, new int[]{512, 512}));
        mainBlock.add(Linear.builder().setUnits(512).build())
                .add(Activation.leakyReluBlock(0.1f)) // LeakyReLU es mejor para evitar neuronas muertas
                .add(BatchNorm.builder().build())     // Estabiliza el aprendizaje
                .add(Dropout.builder().optRate(0.2f).build()) // Evita memorización
                .add(Linear.builder().setUnits(256).build());

        // Capa 2: Compresión
        mainBlock.add(Linear.builder().setUnits(256).build())
                .add(Activation.leakyReluBlock(0.1f))
                .add(BatchNorm.builder().build())
                .add(Dropout.builder().optRate(0.1f).build());

        // Capa 3: Refinamiento antes de las cabezas
        mainBlock.add(Linear.builder().setUnits(128).build())
                .add(Linear.builder().setUnits(128).build())
                .add(Dropout.builder().optRate(0.1f).build())
                .add(Activation.leakyReluBlock(0.1f));
    }

    private static ChunkDataset computeDataset(Pair<float[][][], float[][]> pair, int batchSize, NDManager manager) {
        NDArray X_train = EngineUtils.concat3DArrayToNDArray(pair.getKey(), manager, 1024);
        NDArray y_train = manager.create(pair.getValue());
        return new ChunkDataset(X_train, y_train, new ArrayDataset.Builder()
                .setData(X_train)
                .optLabels(y_train)
                .setSampling(batchSize, false)
                .build());
    }

    private record ChunkDataset(NDArray x, NDArray y, ArrayDataset dataset) {}
}
