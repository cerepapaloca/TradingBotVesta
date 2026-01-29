package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.LambdaBlock;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.pytorch.engine.PtModel;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.MAEEvaluator;
import xyz.cereshost.Main;
import xyz.cereshost.MetricsListener;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;

public class VestaEngine {

    public static final int LOOK_BACK = 45;
    public static final int EPOCH = 10;

    public static final ExecutorService EXECUTOR = Executors.newCachedThreadPool();
    public static final ExecutorService EXECUTOR_TRAINING = Executors.newScheduledThreadPool(8);
    /**
     * Entrena un modelo con múltiples símbolos combinados
     */
    public static TrainingTestsResults trainingModel(@NotNull List<String> symbols) throws TranslateException, IOException, InterruptedException {

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

        Pair<float[][][], float[][]> combined = BuilderData.fullBuild(symbols);

        //Pair<float[][][], float[]> deduped = EngineUtils.clearData(combined.getKey(), combined.getValue());
        float[][][] xCombined = combined.getKey();
        float[][] yCombined = combined.getValue();

        //EngineUtils.shuffleData(xCombined, yCombined);

        ChartUtils.CandleChartUtils.showTPSLDistribution("Datos Combinados", yCombined, "Todos");
        ChartUtils.CandleChartUtils.showDirectionDistribution("Datos Combinados", yCombined, "Todos");


        Vesta.info("Datos combinados:");
        Vesta.info("  Total de muestras: " + xCombined.length);
        Vesta.info("  Lookback: " + xCombined[0].length);
        Vesta.info("  Características: " + xCombined[0][0].length);

        // Preparar dimensiones
        long samples = xCombined.length;
        int lookback = xCombined[0].length;
        int features = xCombined[0][0].length;

        // SPLIT (antes de normalizar) -> 70% train, 15% val, 15% test
        long trainSize = (int) (samples * 0.7);
        long valSize = (int) (samples * 0.15);
        long testSize = samples - trainSize - valSize; // para que sume exactamente samples

        Vesta.info("Split sizes: train=" + trainSize + " val=" + valSize + " test=" + testSize);
        // Borra por que no se va a usar más
        Vesta.MARKETS.clear();
        // Helper local para slice 3D arrays (copia el primer eje [start, end))
        java.util.function.BiFunction<long[], long[], float[][][]> slice3D = (long[] range, long[] dummy) -> {
            long start = range[0];
            long end = range[1];
            int len = (int) (end - start);
            float[][][] out = new float[len][][];
            for (long i = start; i < end; i++) {
                out[(int) (i - start)] = xCombined[(int) i];
            }
            return out;
        };
        splitSample split = getSplitSample(slice3D, trainSize, valSize, samples, yCombined);
        Normalize result = getNormalize(split.X_train_arr(), split.X_val_arr(), split.X_test_arr(), split.y_train_arr(), split.y_val_arr(), split.y_test_arr());
        // Verificar NaN sólo en arrays normalizados (por si acaso)
        EngineUtils.cleanNaNValues(result.X_train_norm().get());
        EngineUtils.cleanNaNValues(result.X_val_norm().get());
        EngineUtils.cleanNaNValues(result.X_test_norm().get());

        try (PtModel model = (PtModel) Model.newInstance(Main.NAME_MODEL, device, "PyTorch")) {
            PtNDManager manager = (PtNDManager) model.getNDManager();
            // Aplanar 3D -> 1D para crear NDArray con Shape(samples, lookback, features)
            float[] XtrainFlat = EngineUtils.flatten3DArray(result.X_train_norm().get());
            float[] XvalFlat   = EngineUtils.flatten3DArray(result.X_val_norm().get());
            float[] XtestFlat  = EngineUtils.flatten3DArray(result.X_test_norm().get());

            NDArray X_train = manager.create(XtrainFlat, new Shape(trainSize, lookback, features));
            NDArray X_val   = manager.create(XvalFlat,   new Shape(valSize,   lookback, features));
            NDArray X_test  = manager.create(XtestFlat,  new Shape(testSize,  lookback, features));

            // y -> shape (N, 1)
            NDArray y_train = manager.create(result.y_train_norm().get());
            NDArray y_val   = manager.create(result.y_val_norm().get());
            NDArray y_test  = manager.create(result.y_test_norm().get());

            Vesta.info("\nDatos finales preparados:");
            Vesta.info("  X_train shape: " + X_train.getShape());
            Vesta.info("  y_train shape: " + y_train.getShape());
            Vesta.info("  X_val shape: " + X_val.getShape());
            Vesta.info("  X_test shape: " + X_test.getShape());

            // Construir modelo (usa tu método existente)

            model.setBlock(getSequentialBlock());
            // Configuración de entrenamiento (igual a tu código)
            MetricsListener metrics = new MetricsListener();
            TrainingConfig config = new DefaultTrainingConfig(new VestaLoss("WeightedL2", 3, 3))
                    .optOptimizer(Optimizer.adamW()
                            .optLearningRateTracker(Tracker.cosine()
                                    .setBaseValue(0.000_01f)
                                    .optFinalValue(0.000_000_5f)
                                    .setMaxUpdates((int) (EPOCH*0.75))
                                    .build())
                            //.optLearningRateTracker(Tracker.fixed(0.0005f))
                            .optWeightDecays(0.01f)
                            .optClipGrad(2.8f)
                            .build())
                    .optDevices(Engine.getInstance().getDevices())
                    .addEvaluator(new MAEEvaluator())
                    .optExecutorService(EXECUTOR_TRAINING)
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(metrics);

            // Crear datasets con los NDArray ya normalizados (shuffle sólo en train)
            int batchSize = 64;
            Dataset trainDataset = new ArrayDataset.Builder()
                    .setData(X_train)
                    .optLabels(y_train)
                    .setSampling(batchSize, true)
                    .build();
            Dataset valDataset = new ArrayDataset.Builder()
                    .setData(X_val)
                    .optLabels(y_val)
                    .setSampling(batchSize, false)
                    .build();
            trainDataset.prepare();
            valDataset.prepare();
            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(symbols.size(), LOOK_BACK, features));

            // Limpiar RAM
            System.gc();

            // Entrenar
            Vesta.info("Iniciando entrenamiento con " + EPOCH + " epochs...");
            EasyTrain.fit(trainer, EPOCH, trainDataset, valDataset);

            // Guardar modelo (igual que antes)
            IOdata.saveModel(model);
            IOdata.saveYNormalizer(result.yNormalizer());
            IOdata.saveXNormalizer(result.xNormalizer());

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
                        LOOK_BACK,
                        features
                );

                // 2. Ejecutar Backtest para cada símbolo (o solo el primero si combinaste)
                // Como entrenaste combinando símbolos, lo ideal es probar en uno representativo o iterar.
                // Aquí probamos con el primer símbolo de la lista para obtener el ROI
                String testSymbol = symbols.get(0);
                EngineUtils.ResultsEvaluate evaluate = EngineUtils.evaluateModel(trainer, X_test, y_test, result.yNormalizer());


                Market market = IOdata.loadMarkets(Main.DATA_SOURCE_FOR_BACK_TEST, testSymbol);

                BackTestEngine.BackTestResult simResult;

                if (market != null) {
                    simResult = new BackTestEngine(market, predEngine).run();
                } else {
                    Vesta.error("No se encontró mercado para backtest: " + testSymbol);
                    simResult = null;
                }
                return new TrainingTestsResults(evaluate, simResult);
            }else {
                return null;
            }
        }
    }

    private static @NotNull splitSample getSplitSample(BiFunction<long[], long[], float[][][]> slice3D, long trainSize, long valSize, long samples, float[][] yCombined) throws InterruptedException {
        CountDownLatch latch1 = new CountDownLatch(6);
        // Crear splits en arrays Java antes de normalizar
        AtomicReference<float[][][]> X_train_arr = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            X_train_arr.set(slice3D.apply(new long[]{0, trainSize}, null));
            latch1.countDown();
        });
        AtomicReference<float[][][]> X_val_arr = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            X_val_arr.set(slice3D.apply(new long[]{trainSize, trainSize + valSize}, null));
            latch1.countDown();
        });
        AtomicReference<float[][][]> X_test_arr = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            X_test_arr.set(slice3D.apply(new long[]{trainSize + valSize, samples}, null));
            latch1.countDown();
        });

        AtomicReference<float[][]> y_train_arr = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            y_train_arr.set(java.util.Arrays.copyOfRange(yCombined, 0, (int) trainSize));
            latch1.countDown();
        });
        AtomicReference<float[][]> y_val_arr = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            y_val_arr.set(java.util.Arrays.copyOfRange(yCombined,(int) trainSize, (int) (trainSize + valSize)));
            latch1.countDown();
        });
        AtomicReference<float[][]> y_test_arr = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            y_test_arr.set(java.util.Arrays.copyOfRange(yCombined,(int)  (trainSize + valSize),(int) samples));
            latch1.countDown();
        });
        latch1.await();
        splitSample split = new splitSample(X_train_arr, X_val_arr, X_test_arr, y_train_arr, y_val_arr, y_test_arr);
        return split;
    }

    private record splitSample(AtomicReference<float[][][]> X_train_arr, AtomicReference<float[][][]> X_val_arr, AtomicReference<float[][][]> X_test_arr, AtomicReference<float[][]> y_train_arr, AtomicReference<float[][]> y_val_arr, AtomicReference<float[][]> y_test_arr) {
    }

    private static @NotNull Normalize getNormalize(AtomicReference<float[][][]> X_train_arr,
                                                   AtomicReference<float[][][]> X_val_arr,
                                                   AtomicReference<float[][][]> X_test_arr,
                                                   AtomicReference<float[][]> y_train_arr,
                                                   AtomicReference<float[][]> y_val_arr,
                                                   AtomicReference<float[][]> y_test_arr
    ) throws InterruptedException {
        // Normalizadores: FIT sólo con TRAIN
        RobustNormalizer xNormalizer = new RobustNormalizer();
        xNormalizer.fit(X_train_arr.get()); // fit con train solamente

        MultiSymbolNormalizer yNormalizer = new MultiSymbolNormalizer();
        yNormalizer.fit(y_train_arr.get()); // fit con train solamente

        CountDownLatch latch2 = new CountDownLatch(6);
        // Transformar train/val/test
        AtomicReference<float[][][]> X_train_norm = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            X_train_norm.set(xNormalizer.transform(X_train_arr.get()));
            latch2.countDown();
        });
        AtomicReference<float[][][]> X_val_norm = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            X_val_norm.set(xNormalizer.transform(X_val_arr.get()));
            latch2.countDown();
        });
        AtomicReference<float[][][]> X_test_norm = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            X_test_norm.set(xNormalizer.transform(X_test_arr.get()));
            latch2.countDown();
        });

        AtomicReference<float[][]> y_train_norm = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            y_train_norm.set(yNormalizer.transform(y_train_arr.get()));
            latch2.countDown();
        });
        AtomicReference<float[][]> y_val_norm = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            y_val_norm.set(yNormalizer.transform(y_val_arr.get()));
            latch2.countDown();
        });
        AtomicReference<float[][]> y_test_norm = new AtomicReference<>(null);
        EXECUTOR.submit(() -> {
            y_test_norm.set(yNormalizer.transform(y_test_arr.get()));
            latch2.countDown();
        });
        latch2.await();
        return new Normalize(xNormalizer, yNormalizer, X_train_norm, X_val_norm, X_test_norm, y_train_norm, y_val_norm, y_test_norm);
    }

    private record Normalize(RobustNormalizer xNormalizer, MultiSymbolNormalizer yNormalizer, AtomicReference<float[][][]> X_train_norm, AtomicReference<float[][][]> X_val_norm, AtomicReference<float[][][]> X_test_norm, AtomicReference<float[][]> y_train_norm, AtomicReference<float[][]> y_val_norm, AtomicReference<float[][]> y_test_norm) {
    }

    public record TrainingTestsResults(EngineUtils.ResultsEvaluate evaluate, BackTestEngine.BackTestResult backtest) {}

    public static SequentialBlock getSequentialBlock() {
        SequentialBlock mainBlock = new SequentialBlock();
        float delta = 0.7f; // Controla qué tan suave es el centro
        float deltaSq = delta * delta;

        mainBlock.add(GRU.builder()
                        .setStateSize(64)
                        .setNumLayers(2)
                        .optReturnState(false)
                        .optBatchFirst(true)
                        .optDropRate(0.3f)
                        .build())
                .add(ndList -> new NDList(ndList.singletonOrThrow().get(":, -1, :")))
                .add(Linear.builder().setUnits(64).build());
        ParallelBlock branches = new ParallelBlock(list -> {
            NDArray tp = list.get(0).singletonOrThrow();
            NDArray sl = list.get(1).singletonOrThrow();
            NDArray dir = list.get(2).singletonOrThrow();
            return new NDList(
                    NDArrays.concat(new NDList(tp, sl, dir), 1) // ✅ axis = 1
            );
        });

        // TP
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x.pow(2).add(deltaSq).sqrt().sub(delta));
                }))
        );

        // SL
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(1).build())
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x.pow(2).add(deltaSq).sqrt().sub(delta));
                }))
        );

        // Dirección
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Linear.builder().setUnits(3).build()) // Long Neutro Short
                .add(new LambdaBlock(ndArrays -> {
                    NDArray x = ndArrays.singletonOrThrow();
                    return new NDList(x.pow(2).add(deltaSq).sqrt().sub(delta));
                }))
                .add(new LambdaBlock(ndArrays -> new NDList(softmax(ndArrays.singletonOrThrow()))))
        );


        mainBlock.add(branches);
        return mainBlock;
    }


    public static NDArray softmax(NDArray input) {
        // Para estabilidad numérica: restar el máximo
        NDArray max = input.max(new int[]{-1}, true);
        NDArray shifted = input.sub(max);

        // Calcular exponencial
        NDArray exp = shifted.exp();

        // Calcular suma a lo largo del último eje
        NDArray sum = exp.sum(new int[]{-1}, true);

        // Dividir exponencial por la suma
        return exp.div(sum);
    }



}
