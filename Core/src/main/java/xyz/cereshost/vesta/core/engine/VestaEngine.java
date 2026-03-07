package xyz.cereshost.vesta.core.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.pytorch.engine.PtModel;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.initializer.Initializer;
import ai.djl.training.listener.EpochTrainingListener;
import ai.djl.training.listener.EvaluatorTrainingListener;
import ai.djl.training.listener.LoggingTrainingListener;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.core.Main;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.blocks.TemporalTransformerBlock;
import xyz.cereshost.vesta.core.io.IOMarket;
import xyz.cereshost.vesta.core.io.IOdata;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.core.metrics.MAEEvaluator;
import xyz.cereshost.vesta.core.metrics.MaxDiffEvaluator;
import xyz.cereshost.vesta.core.metrics.MetricsListener;
import xyz.cereshost.vesta.core.metrics.MinDiffEvaluator;
import xyz.cereshost.vesta.core.utils.BuilderData;
import xyz.cereshost.vesta.core.utils.EngineUtils;
import xyz.cereshost.vesta.core.utils.TrainingData;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VestaEngine {

    public static final int LOOK_BACK = 60*4;
    public static final int EPOCH = 32;
    public static final int AUXILIAR_EPOCH = 1;
    public static final int BACH_SIZE = 1;
    public static final int SPLIT_DATA = 128;//64*4;

    @Getter @Setter
    private static NDManager rootManager;

    public static final ExecutorService EXECUTOR = Executors.newCachedThreadPool();
    public static final ExecutorService EXECUTOR_BUILD = Executors.newScheduledThreadPool(6);
    public static final ExecutorService EXECUTOR_AUXILIAR_BUILD = Executors.newScheduledThreadPool(10);
    public static final ExecutorService EXECUTOR_READ_CACHE_BUILD = Executors.newScheduledThreadPool(2);
    public static final ExecutorService EXECUTOR_WRITE_CACHE_BUILD = Executors.newScheduledThreadPool(6);
    public static final ExecutorService EXECUTOR_TRAINING = Executors.newScheduledThreadPool(8);

    /**
     * Entrena un modelo con múltiples símbolos combinados
     */
    @SuppressWarnings("UnusedAssignment")
    public static @NotNull TrainingTestsResults trainingModel(@NotNull List<String> symbols) throws IOException, InterruptedException, ExecutionException {
        Engine torch = Engine.getEngine("PyTorch");
        JniUtils.setGraphExecutorOptimize(false);

        if (torch == null) {
            Vesta.error("PyTorch no está disponible. Engines disponibles:");
            for (String engine : Engine.getAllEngines()) {
                Vesta.error("  - " + engine);
            }
            throw new RuntimeException("PyTorch engine no encontrado");
        }
        Device device = Engine.getInstance().getDevices()[0];
        Vesta.info("Usando dispositivo: " + device);
        Vesta.info("Entrenando con " + symbols.size() + " símbolos: " + symbols);

        try (PtModel model = (PtModel) Model.newInstance(Main.NAME_MODEL, device, "PyTorch")) {
            PtNDManager manager = (PtNDManager) model.getNDManager();

            final TrainingData data;
            if (IOdata.isBuiltData()){
                data = IOdata.getBuiltData();
            }else {
                data = BuilderData.buildTrainingData(symbols,  Main.MAX_MONTH_TRAINING, 1);
                IOdata.saveCacheProperties(data.getCacheProperties(symbols));

            }
            data.prepareNormalize();

            IOdata.saveYNormalizer(data.getYNormalizer());
            IOdata.saveXNormalizer(data.getXNormalizer());

            Vesta.info("Datos combinados:");
            Vesta.info("  Total de muestras: " + data.getSamplesSize());
            Vesta.info("  Lookback: " + data.getLookback());
            Vesta.info("  Características: " + data.getFeatures());

            System.gc();

            Vesta.info("Split sizes: train=" + data.getTrainSize() + " val=" + data.getValSize() + " test=" + data.getTestSize());
            // Borra por que no se va a usar más
            Vesta.MARKETS.clear();

            model.setBlock(getSequentialBlock());

            // Configuración de entrenamiento
            System.out.println();
            TrainingConfig config = new DefaultTrainingConfig(new VestaLoss())
                    .optOptimizer(Optimizer.adam()
//                            .optLearningRateTracker(Tracker.cosine()
//                                    .setBaseValue(.000_1f)
//                                    .optFinalValue(.000_000_001f)
//                                    .setMaxUpdates(
//                                            (int) (
//                                                    (
//                                                            (double) data.getTrainSize() / ((double) (BACH_SIZE * EPOCH * Main.MAX_MONTH_TRAINING * AUXILIAR_EPOCH * SPLIT_DATA/2)) *
//                                                            EPOCH* AUXILIAR_EPOCH * Main.MAX_MONTH_TRAINING
//                                                    )*0.80
//                                            )
//                                    )
//                                    .build())
                            .optWeightDecays(0.0001f)
                            .optLearningRateTracker(Tracker.cyclical()
                                    .optBaseValue(.000_000_01f)
                                    .optMaxValue(.0001f)
                                    .optStepSizeDown(3_000)
                                    .optStepSizeUp(3_000)
                                    .build())
                            .optClipGrad(2f)
//                            .optLearningRateTracker(Tracker.fixed(.004f))
                            .build())
                    .optDevices(Engine.getInstance().getDevices())
                    .addEvaluator(new MAEEvaluator())
                    .addEvaluator(new MaxDiffEvaluator())
                    .addEvaluator(new MinDiffEvaluator())
                    .optInitializer(Initializer.ZEROS, Parameter.Type.BETA)
                    .optExecutorService(EXECUTOR_TRAINING)
//                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(
                            new EpochTrainingListener(),
                            new EvaluatorTrainingListener(),
                            new LoggingTrainingListener()
                    )
                    .addTrainingListeners(new MetricsListener());
//                    .addTrainingListeners(new AutoStopListener());

            int batchSize = BACH_SIZE;
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(batchSize, LOOK_BACK, data.getFeatures()));

            // Entrenar
            int maxMonthTraining = Main.MAX_MONTH_TRAINING;
            long totalParams = 0;
            for (Parameter p : model.getBlock().getParameters().values()) {
                totalParams += p.getArray().getShape().size();
            }
            Vesta.info("🧠 Total de parámetros: %,d", totalParams);
            Vesta.info("Iniciando entrenamiento con " + EPOCH* AUXILIAR_EPOCH *maxMonthTraining + " epochs...");
            rootManager = manager;
            NDManager managerTraining = manager.newSubManager();
            System.gc();
            data.preLoad(2, TrainingData.ModeData.SECUENCIAL, SPLIT_DATA);
            ChunkDataset sampleTraining = computeDataset(data.nextTrainingData(), batchSize, managerTraining);
            ChunkDataset sampleVal = computeDataset(data.nextValidationData(), 256, managerTraining);


            try{
                for (int epoch = 0; epoch < EPOCH; epoch++) {
                    for (int idx = 0; idx < maxMonthTraining; idx++) {
                        CompletableFuture<ChunkDataset> sampleTrainingNext = CompletableFuture.supplyAsync(() ->
                                computeDataset(data.nextTrainingData(), batchSize, managerTraining), EXECUTOR_AUXILIAR_BUILD);
                        CompletableFuture<ChunkDataset> sampleValNext = CompletableFuture.supplyAsync(() ->
                                computeDataset(data.nextValidationData(), batchSize, managerTraining), EXECUTOR_AUXILIAR_BUILD);
                        System.out.println(sampleTraining.dataset.size());
//                    roiBackTest = (float) new BackTestEngine(market, predEngine, new BetaStrategy()).run(allCandles).roiPercent();

                        EasyTrain.fit(trainer, AUXILIAR_EPOCH, sampleTraining.dataset(), sampleVal.dataset());
                        NDArray xT = sampleTraining.x();
                        NDArray yT = sampleTraining.y();
                        NDArray xV = sampleVal.x();
                        NDArray yV = sampleVal.y();

                        EXECUTOR_TRAINING.submit(() -> {
                            xT.close();
                            yT.close();
                            xV.close();
                            yV.close();
                            EngineUtils.clearCacheFloats();
                        });
                        if(!sampleTrainingNext.isDone() || !sampleValNext.isDone()) Vesta.warning("Mes no listo procesando...");
                        sampleTraining = sampleTrainingNext.get();
                        sampleTrainingNext = null;
                        sampleVal = sampleValNext.get();
                        sampleValNext = null;
                        if (stop) break;
                        //sampleVal = sampleValNext.get();
                    }
                    if (stop) break;
                }
            }catch (Exception e){
                e.printStackTrace();
            }
            stop = true;
            managerTraining.close();
            data.closePosTraining();
            System.gc();
            // Guardar modelo (igual que antes)
            IOdata.saveModel(model);

            Pair<float[][][], float[][]> pairTest = data.getTestNormalize();

            NDArray X_test  = EngineUtils.concat3DArrayToNDArray(pairTest.getKey(), manager, 1024);
            NDArray y_test  = manager.create(pairTest.getValue());
            // Evaluar en conjunto de test si hay muestras
            Vesta.info("\nEvaluando modelo con Backtest Walk-Forward (15% data)...");
            data.closeAll();

            EngineUtils.ResultsEvaluate evaluate = EngineUtils.evaluateModel(trainer, X_test, y_test, data.getYNormalizer(), 1024);
            PredictionEngine predEngine = new PredictionEngine(
                    data.getXNormalizer(),
                    data.getYNormalizer(),
                    model,
                    data.getLookback(),
                    data.getFeatures()
            );
            Market market = new Market(symbols.getFirst());
            for (int day = 12; day >= 1; day--) {
                market.concat(IOMarket.loadMarkets(Main.DATA_SOURCE_FOR_BACK_TEST, symbols.getFirst(), day));
            }

            BackTestEngine.BackTestResult simResult;
            simResult = new BackTestEngine(market, predEngine).run();
            manager.close();
            return new TrainingTestsResults(evaluate, simResult);
        }
    }
    public record TrainingTestsResults(EngineUtils.ResultsEvaluate evaluate, BackTestEngine.BackTestResult backtest) {}

    @SuppressWarnings("DuplicatedCode")
    public static @NotNull SequentialBlock getSequentialBlock() {
        SequentialBlock mainBlock = new SequentialBlock();

        TTLHeader(mainBlock);
        // Salidas en orden: [upMove, downMove, 0, 0, 0]
        ParallelBlock branches = new ParallelBlock(list -> {
            NDArray out0 = list.get(0).singletonOrThrow();
            NDArray out1 = list.get(1).singletonOrThrow();
            NDArray out2 = list.get(2).singletonOrThrow();
            NDArray out3 = list.get(3).singletonOrThrow();
            NDArray out4 = list.get(4).singletonOrThrow();
            return new NDList(
                    NDArrays.concat(new NDList(out0, out1, out2, out3, out4), 1)
            );
        });

        branches.add(getMagnitud());   // output 0: upMove
        branches.add(getMagnitud());   // output 1: downMove
        branches.add(getZeroHead());   // output 2: sin uso
        branches.add(getZeroHead());   // output 3: sin uso
        branches.add(getZeroHead());   // output 4: sin uso

        mainBlock.add(branches);
        return mainBlock;
    }

    private static SequentialBlock getMagnitud() {
//        ParallelBlock branches = new ParallelBlock(list -> {
//            NDArray linearBlock = list.get(0).singletonOrThrow();
//            NDArray cosBLock = list.get(1).singletonOrThrow();
//            return new NDList(
//                    NDArrays.concat(new NDList(linearBlock, cosBLock), 1) // axis = 1
//            );
//        });
//
//        branches.add(Linear.builder().setUnits(64).build())
//                .add(Linear.builder().setUnits(64).build())
//                .add(Linear.builder().setUnits(64).build())
//                .add(Linear.builder().setUnits(32).build());
//
//        branches.add(Linear.builder().setUnits(64).build())
//                .add(CosLinear.builder().build())
//                .add(Linear.builder().setUnits(64).build())
//                .add(CosLinear.builder().build())
//                .add(Linear.builder().setUnits(32).build());

        return new SequentialBlock()
                .add(Linear.builder().setUnits(1024).build())
                .add(Linear.builder().setUnits(1024).build())
                .add(Linear.builder().setUnits(1024).build())
                .add(Linear.builder().setUnits(1024).build())
                .add(Linear.builder().setUnits(1024).build())
                .add(Linear.builder().setUnits(512).build())
                .add(Linear.builder().setUnits(512).build())
                .add(Linear.builder().setUnits(1).build());
    }

    private static SequentialBlock getZeroHead() {
        return new SequentialBlock()
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x.getManager().zeros(new Shape(x.getShape().get(0), 1)));
                }));
    }

    private static void TTLHeader(SequentialBlock mainBlock) {
        mainBlock.add(TemporalTransformerBlock.builder()
                        .setModelDim(1024)
                        .setNumHeads(8)
                        .setFeedForwardDim(8*1024)
                        .setDropoutRate(0.0f)
                        .setMaxSequenceLength(LOOK_BACK)
                        .setOftenClearCache(20)
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
                .add(Linear.builder().setUnits(1024).build());
    }



    private static ChunkDataset computeDataset(Pair<float[][][], float[][]> pairNormalize, int batchSize, NDManager manager) {
        NDArray X_train = EngineUtils.concat3DArrayToNDArray(pairNormalize.getKey(), manager, 1024);
        NDArray y_train = manager.create(pairNormalize.getValue());
        Arrays.fill(pairNormalize.getValue(), null);
        Arrays.fill(pairNormalize.getKey(), null);
        return new ChunkDataset(X_train, y_train, new ArrayDataset.Builder()
                .setData(X_train)
                .optLabels(y_train)
                .setSampling(batchSize, false)
                .build());
    }

    private static boolean stop = false;

    public static void stopTraining(){
        Vesta.info("⛔ Deteniendo el entrenamiento");
        stop = true;
    }

    private record ChunkDataset(NDArray x, NDArray y, ArrayDataset dataset) {}
}
