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
import ai.djl.nn.*;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.BatchNorm;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.recurrent.GRU;
import ai.djl.nn.transformer.TransformerEncoderBlock;
import ai.djl.pytorch.engine.PtModel;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.timeseries.translator.TransformerTranslator;
import ai.djl.timeseries.translator.TransformerTranslatorFactory;
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
import com.ibm.icu.number.FormattedNumber;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.Main;
import xyz.cereshost.blocks.CosLinear;
import xyz.cereshost.blocks.TemporalTransformerBlock;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.io.IOMarket;
import xyz.cereshost.io.IOdata;
import xyz.cereshost.metrics.MAEEvaluator;
import xyz.cereshost.metrics.MaxDiffEvaluator;
import xyz.cereshost.metrics.MetricsListener;
import xyz.cereshost.metrics.MinDiffEvaluator;
import xyz.cereshost.utils.AutoStopListener;
import xyz.cereshost.utils.BuilderData;
import xyz.cereshost.utils.EngineUtils;
import xyz.cereshost.utils.TrainingData;

import java.io.IOException;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class VestaEngine {

    public static final int LOOK_BACK = 120;
    public static final int EPOCH = 4;
    public static final int EPOCH_SUB = 1;
    public static final int BACH_SIZE = 128;

    @Getter @Setter
    private static NDManager rootManager;

    public static final ExecutorService EXECUTOR = Executors.newCachedThreadPool();
    public static final ExecutorService EXECUTOR_BUILD = Executors.newScheduledThreadPool(1);
    public static final ExecutorService EXECUTOR_AUXILIAR_BUILD = Executors.newScheduledThreadPool(12);
    public static final ExecutorService EXECUTOR_WRITE_CACHE_BUILD = Executors.newScheduledThreadPool(4);
    public static final ExecutorService EXECUTOR_TRAINING = Executors.newScheduledThreadPool(8);
    private static final float DIRECTION_EQUALIZATION_STRENGTH = 0.35f;
    /**
     * Entrena un modelo con múltiples símbolos combinados
     */
    @SuppressWarnings("UnusedAssignment")
    public static @NotNull TrainingTestsResults trainingModel(@NotNull List<String> symbols) throws TranslateException, IOException, InterruptedException, ExecutionException {
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

        try (PtModel model = (PtModel) Model.newInstance(Main.NAME_MODEL, device, "PyTorch")) {
            PtNDManager manager = (PtNDManager) model.getNDManager();

            final TrainingData data;
            if (IOdata.isBuiltData()){
                data = IOdata.getBuiltData();
            }else {
                data = BuilderData.buildTrainingData(symbols,  Main.MAX_MONTH_TRAINING, 1);
                IOdata.saveCacheProperties(data.getCacheProperties(symbols));
            }


            Vesta.info("Datos combinados:");
            Vesta.info("  Total de muestras: " + data.getSamplesSize());
            Vesta.info("  Lookback: " + data.getLookback());
            Vesta.info("  Características: " + data.getFeatures());

            data.prepareNormalize();
            System.gc();

            Vesta.info("Split sizes: train=" + data.getTrainSize() + " val=" + data.getValSize() + " test=" + data.getTestSize());
            // Borra por que no se va a usar más
            Vesta.MARKETS.clear();

            // Construir modelo (usa tu método existente)
            model.setBlock(getSequentialBlock());

            // Configuración de entrenamiento
            TrainingConfig config = new DefaultTrainingConfig(new VestaLoss())
                    .optOptimizer(Optimizer.adamW()
                            .optLearningRateTracker(Tracker.cosine()
                                    .setBaseValue(0.000_1f)
                                    .optFinalValue(0.000_000_5f)
                                    .setMaxUpdates((int) (EPOCH*EPOCH_SUB*((double)Main.MAX_MONTH_TRAINING)))
                                    .build())
                            .optWeightDecays(0.001f)
//                            .optLearningRateTracker(Tracker.cyclical()
//                                    .optBaseValue(0.000_000_5f)
//                                    .optMaxValue(0.000_01f)
//                                    .optStepSizeDown(BACH_SIZE*4)
//                                    .optStepSizeUp(BACH_SIZE*4)
//                                    .build())
                            .optClipGrad(2f)
                            .build())
                    .optDevices(Engine.getInstance().getDevices())
                    .addEvaluator(new MAEEvaluator())
                    .addEvaluator(new MaxDiffEvaluator())
                    .addEvaluator(new MinDiffEvaluator())
//                .addEvaluator(new BinaryDirectionEvaluator())
//                .addEvaluator(new DirectionAccuracyEvaluator())
                    .optExecutorService(EXECUTOR_TRAINING)
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(new MetricsListener())
                    .addTrainingListeners(new AutoStopListener())
                    .addTrainingListeners(data)
                    ;

            int batchSize = BACH_SIZE;

            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(batchSize, LOOK_BACK, data.getFeatures()));


            // Entrenar
            int maxMonthTraining = Main.MAX_MONTH_TRAINING;
            long totalParams = 0;

            for (Parameter p : model.getBlock().getParameters().values()) {
                totalParams += p.getArray().size();
            }
            Vesta.info("Total de parámetros: %,d (neuronas)", totalParams);
            Vesta.info("Iniciando entrenamiento con " + EPOCH*EPOCH_SUB*maxMonthTraining + " epochs...");
            rootManager = manager;
            System.gc();
            data.preLoad(2, TrainingData.ModeData.RAMDOM);
            ChunkDataset sampleTraining = computeDataset(data.nextData(), batchSize, manager);
            ChunkDataset sampleVal = computeDataset(data.getValNormalize(), batchSize, manager);
            for (int epoch = 0; epoch < EPOCH; epoch++) {
                for (int idx = 0; idx < maxMonthTraining; idx++) {
                    CompletableFuture<ChunkDataset> sampleTrainingNext = CompletableFuture.supplyAsync(() ->
                            computeDataset(data.nextData(), batchSize, manager), EXECUTOR_TRAINING);
                    EasyTrain.fit(trainer, EPOCH_SUB, sampleTraining.dataset(), sampleVal.dataset());
                    NDArray xT = sampleTraining.x();
                    NDArray yT = sampleTraining.y();
                    EXECUTOR_TRAINING.submit(() -> {
                        xT.close();
                        yT.close();
                        EngineUtils.clearCacheFloats();
                    });
                    if(!sampleTrainingNext.isDone() /*|| !sampleValNext.isDone()*/) Vesta.warning("Mes no listo procesando...");
                    sampleTraining = sampleTrainingNext.get();
                    sampleTrainingNext = null;
                    if (stop) break;
                    //sampleVal = sampleValNext.get();
                }
                if (stop) break;
            }
            sampleVal.x().close();
            sampleVal.y().close();
            sampleTraining.x().close();
            sampleTraining.y().close();
            System.gc();
            // Guardar modelo (igual que antes)
            IOdata.saveModel(model);
            IOdata.saveYNormalizer(data.getYNormalizer());
            IOdata.saveXNormalizer(data.getXNormalizer());

            Pair<float[][][], float[][]> pairTest = data.getTestNormalize();
            NDArray X_test  = EngineUtils.concat3DArrayToNDArray(pairTest.getKey(), manager, 1024);
            NDArray y_test  = manager.create(pairTest.getValue());
            // Evaluar en conjunto de test si hay muestras
            Vesta.info("\nEvaluando modelo con Backtest Walk-Forward (15% data)...");

            // 1. Crear instancia temporal de PredictionEngine con los datos recién entrenados
            // Nota: Necesitamos un PredictionEngine para usar el método 'predictForBacktest'
            // Como el modelo (PtModel) ya está en memoria, podemos pasarlo directamente.

            // Necesitamos pasar un Model 'genérico', PtModel hereda de Model
            PredictionEngine predEngine = new PredictionEngine(
                    data.getXNormalizer(),
                    data.getYNormalizer(),
                    model,
                    data.getLookback(),
                    data.getFeatures()
            );

            // 2. Ejecutar Backtest para cada símbolo (o solo el primero si combinaste)
            // Como entrenaste combinando símbolos, lo ideal es probar en uno representativo o iterar.
            // Aquí probamos con el primer símbolo de la lista para obtener el ROI
            String testSymbol = symbols.get(0);
            Market market = IOMarket.loadMarkets(Main.DATA_SOURCE_FOR_BACK_TEST, symbols.get(0), 1).limit(7);
            EngineUtils.ResultsEvaluate evaluate = EngineUtils.evaluateModel(trainer, X_test, y_test, data.getYNormalizer(), 1024);

            BackTestEngine.BackTestResult simResult;
            if (market != null) {
                simResult = new BackTestEngine(market, predEngine).run();
            } else {
                Vesta.error("No se encontró mercado para backtest: " + testSymbol);
                simResult = null;
            }
            manager.close();
            return new TrainingTestsResults(evaluate, simResult);
        }
    }
    public record TrainingTestsResults(EngineUtils.ResultsEvaluate evaluate, BackTestEngine.BackTestResult backtest) {}

    @SuppressWarnings("DuplicatedCode")
    public static @NotNull SequentialBlock getSequentialBlock() {
        SequentialBlock mainBlock = new SequentialBlock();

        float deltaFloat = 0.5f; // Controla qué tan suave es el centro
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

        // Max
        branches.add(getMagnitud(deltaFloat));

        // Min
        branches.add(getMagnitud(deltaFloat));

        // Dirección
        ParallelBlock dirSub = new ParallelBlock(subList -> {
            // subList contiene los outputs de cada sub-brazo (cada uno será [B,1])
            NDArray a = subList.get(0).singletonOrThrow();
            NDArray b = subList.get(1).singletonOrThrow();
            NDArray c = subList.get(2).singletonOrThrow();
            // concatenamos a lo largo de la dimensión de features para obtener [B,3]
            return new NDList(NDArrays.concat(new NDList(a, b, c), 1));
        });

        // Long
//        dirSub.add(new SequentialBlock()
//                .add(Linear.builder().setUnits(64).build())
//                .add(Dropout.builder().optRate(0.3f).build())
//                .add(Linear.builder().setUnits(32).build())
//                .add(Linear.builder().setUnits(1).build())
//                .add(new LambdaBlock(ndArrays -> {
//                    NDArray x = ndArrays.singletonOrThrow();
//                    return new NDList(x); // ya es escalar por muestra
//                }))
//        );
//
//        // Neutral
//        dirSub.add(new SequentialBlock()
//                .add(Linear.builder().setUnits(32).build())
//                .add(Dropout.builder().optRate(0.2f).build())
//                .add(Linear.builder().setUnits(32).build())
//                .add(Linear.builder().setUnits(1).build())
//                .add(new LambdaBlock(ndArrays -> {
//                    NDArray x = ndArrays.singletonOrThrow();
//                    return new NDList(x);
//                }))
//        );
//
//        // Short
//        dirSub.add(new SequentialBlock()
//                .add(Linear.builder().setUnits(64).build())
//                .add(Dropout.builder().optRate(0.3f).build())
//                .add(Linear.builder().setUnits(32).build())
//                .add(Linear.builder().setUnits(1).build())
//                .add(new LambdaBlock(ndArrays -> {
//                    NDArray x = ndArrays.singletonOrThrow();
//                    return new NDList(x);
//                }))
//        );

        // Dirección (únion)
        branches.add(new SequentialBlock()
                        .add(Linear.builder().setUnits(3).build())
//                .add(Dropout.builder().optRate(0.3f).build())
//                .add(Linear.builder().setUnits(128).build())
//                .add(LayerNorm.builder().build())
//                .add(dirSub)
//                .add(new LambdaBlock(ndArrays -> {
//                    NDArray concatenated = ndArrays.singletonOrThrow(); // [B,3]
//                    // Mantener dimensión [B,1]
//                    NDArray longLogit = concatenated.get(":, 0:1");
//                    NDArray neutralLogit = concatenated.get(":, 1:2");
//                    NDArray shortLogit = concatenated.get(":, 2:3");
//
//                    NDArray diff = longLogit.sub(shortLogit);
//
//                    NDArray adjust = diff.mul(
//                            EngineUtils.floatToNDArray(
//                                    DIRECTION_EQUALIZATION_STRENGTH,
//                                    concatenated.getManager()
//                            )
//                    );
//
//                    NDArray longBalanced = longLogit.sub(adjust);
//                    NDArray shortBalanced = shortLogit.add(adjust);
//
//                    // Ahora TODOS son [B,1], concat en eje 1 es válido
//                    return new NDList(NDArrays.concat( new NDList(longBalanced, neutralLogit, shortBalanced),1));
//                }))
//                .add(new LambdaBlock(ndArrays -> {
//                    NDArray x = ndArrays.singletonOrThrow();
//                    return new NDList(x.mul(x).add(EngineUtils.floatToNDArray(deltaFloat*deltaFloat, ndArrays.getManager())).sqrt().sub(EngineUtils.floatToNDArray(deltaFloat, ndArrays.getManager())));
//                }))
//                .add(Softmax.builder().temperature(1).build())

        );

        mainBlock.add(branches);
        return mainBlock;
    }

    private static SequentialBlock getMagnitud(float deltaFloat) {
        ParallelBlock branches = new ParallelBlock(list -> {
            NDArray linearBlock = list.get(0).singletonOrThrow();
            NDArray cosBLock = list.get(1).singletonOrThrow();
            return new NDList(
                    NDArrays.concat(new NDList(linearBlock, cosBLock), 1) // axis = 1
            );
        });

        branches.add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(32).build());

        branches.add(Linear.builder().setUnits(64).build())
                .add(CosLinear.builder().build())
                .add(Linear.builder().setUnits(64).build())
                .add(CosLinear.builder().build())
                .add(Linear.builder().setUnits(32).build());

        return new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(branches)
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(1).build());
//                .add(new LambdaBlock(ndArrays -> {
//                    NDArray x = ndArrays.singletonOrThrow();
//                    return new NDList(x.mul(x).add(EngineUtils.floatToNDArray(deltaFloat * deltaFloat, ndArrays.getManager())).sqrt().sub(EngineUtils.floatToNDArray(deltaFloat, ndArrays.getManager())));
//                }));
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
                        .setModelDim(256)
                        .setNumHeads(4)
                        .setFeedForwardDim(4*256)
                        .setDropoutRate(0.4f)
                        .setMaxSequenceLength(LOOK_BACK)
                        .setOftenClearCache(40)
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
                .add(Linear.builder().setUnits(128).build())
                .add(Dropout.builder().optRate(0.05f).build())
                .add(Linear.builder().setUnits(128).build());
    }

    private static void MLPHeader(SequentialBlock mainBlock) {
        mainBlock.add(new Mlp(BuilderData.FEATURES*LOOK_BACK, 512, new int[]{512, 512}));
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

    private static ChunkDataset computeDataset(Pair<float[][][], float[][]> pairNormalize, int batchSize, NDManager manager) {
        NDArray X_train = EngineUtils.concat3DArrayToNDArray(pairNormalize.getKey(), manager, 1024);
        NDArray y_train = manager.create(pairNormalize.getValue());
        Arrays.fill(pairNormalize.getValue(), null);
        Arrays.fill(pairNormalize.getKey(), null);
        return new ChunkDataset(X_train, y_train, new ArrayDataset.Builder()
                .setData(X_train)
                .optLabels(y_train)
                .setSampling(batchSize, true)
                .build());
    }

    private static boolean stop = false;

    public static void stopTraining(){
        Vesta.info("⛔ Deteniendo el entrenamiento");
        stop = true;
    }

    private record ChunkDataset(NDArray x, NDArray y, ArrayDataset dataset) {}
}
