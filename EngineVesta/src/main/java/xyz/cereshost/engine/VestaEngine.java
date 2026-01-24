package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.ParallelBlock;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.recurrent.GRU;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.pytorch.engine.PtModel;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.training.*;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.*;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.List;

public class VestaEngine {

    public static final int LOOK_BACK = 30;
    public static final int EPOCH = 250;

    /**
     * Entrena un modelo con múltiples símbolos combinados
     */
    public static TrainingTestsResults trainingModel(@NotNull List<String> symbols) throws TranslateException, IOException, InterruptedException {
        IOdata.loadMarkets(Main.DATA_SOURCE_FOR_TRAINING_MODEL, symbols);
        ai.djl.engine.Engine torch = ai.djl.engine.Engine.getEngine("PyTorch");
        if (torch == null) {
            Vesta.error("PyTorch no está disponible. Engines disponibles:");
            for (String engine : ai.djl.engine.Engine.getAllEngines()) {
                Vesta.error("  - " + engine);
            }
            throw new RuntimeException("PyTorch engine no encontrado");
        }

        Device device = Device.gpu();
        Vesta.info("Usando dispositivo: " + device);
        Vesta.info("Entrenando con " + symbols.size() + " símbolos: " + symbols);

        Pair<float[][][], float[][]> combined = BuilderData.fullBuild(symbols);

        //Pair<float[][][], float[]> deduped = EngineUtils.clearData(combined.getKey(), combined.getValue());
        float[][][] xCombined = combined.getKey();
        float[][] yCombined = combined.getValue();

        //EngineUtils.shuffleData(xCombined, yCombined);

        ChartUtils.CandleChartUtils.showDataDistribution("Datos Combinados", yCombined, "Todos");


        Vesta.info("Datos combinados:");
        Vesta.info("  Total de muestras: " + xCombined.length);
        Vesta.info("  Lookback: " + xCombined[0].length);
        Vesta.info("  Características: " + xCombined[0][0].length);

        // Preparar dimensiones
        int samples = xCombined.length;
        int lookback = xCombined[0].length;
        int features = xCombined[0][0].length;

        // SPLIT (antes de normalizar) -> 70% train, 15% val, 15% test
        int trainSize = (int) (samples * 0.7);
        int valSize = (int) (samples * 0.15);
        int testSize = samples - trainSize - valSize; // para que sume exactamente samples

        Vesta.info("Split sizes: train=" + trainSize + " val=" + valSize + " test=" + testSize);

        // Helper local para slice 3D arrays (copia el primer eje [start, end))
        java.util.function.BiFunction<int[], int[], float[][][]> slice3D = (int[] range, int[] dummy) -> {
            int start = range[0];
            int end = range[1];
            int len = end - start;
            float[][][] out = new float[len][][];
            for (int i = start; i < end; i++) {
                out[i - start] = xCombined[i];
            }
            return out;
        };

        // Crear splits en arrays Java antes de normalizar
        float[][][] X_train_arr = slice3D.apply(new int[]{0, trainSize}, null);
        float[][][] X_val_arr   = slice3D.apply(new int[]{trainSize, trainSize + valSize}, null);
        float[][][] X_test_arr  = slice3D.apply(new int[]{trainSize + valSize, samples}, null);

        float[][] y_train_arr = java.util.Arrays.copyOfRange(yCombined, 0, trainSize);
        float[][] y_val_arr   = java.util.Arrays.copyOfRange(yCombined, trainSize, trainSize + valSize);
        float[][] y_test_arr  = java.util.Arrays.copyOfRange(yCombined, trainSize + valSize, samples);

        // Normalizadores: FIT sólo con TRAIN
        RobustNormalizer xNormalizer = new RobustNormalizer();
        xNormalizer.fit(X_train_arr); // fit con train solamente

        MultiSymbolNormalizer yNormalizer = new MultiSymbolNormalizer();
        yNormalizer.fit(y_train_arr); // fit con train solamente

        // Transformar train/val/test
        float[][][] X_train_norm = xNormalizer.transform(X_train_arr);
        float[][][] X_val_norm   = xNormalizer.transform(X_val_arr);
        float[][][] X_test_norm  = xNormalizer.transform(X_test_arr);

        float[][] y_train_norm = yNormalizer.transform(y_train_arr);
        float[][] y_val_norm   = yNormalizer.transform(y_val_arr);
        float[][] y_test_norm  = yNormalizer.transform(y_test_arr);

        // Verificar NaN sólo en arrays normalizados (por si acaso)
        EngineUtils.cleanNaNValues(X_train_norm);
        EngineUtils.cleanNaNValues(X_val_norm);
        EngineUtils.cleanNaNValues(X_test_norm);

        try (PtModel model = (PtModel) Model.newInstance(Main.NAME_MODEL, device, "PyTorch")) {
            PtNDManager manager = (PtNDManager) model.getNDManager();
            // Aplanar 3D -> 1D para crear NDArray con Shape(samples, lookback, features)
            float[] XtrainFlat = EngineUtils.flatten3DArray(X_train_norm);
            float[] XvalFlat   = EngineUtils.flatten3DArray(X_val_norm);
            float[] XtestFlat  = EngineUtils.flatten3DArray(X_test_norm);

            NDArray X_train = manager.create(XtrainFlat, new Shape(trainSize, lookback, features));
            NDArray X_val   = manager.create(XvalFlat,   new Shape(valSize,   lookback, features));
            NDArray X_test  = manager.create(XtestFlat,  new Shape(testSize,  lookback, features));

            // y -> shape (N, 1)
            NDArray y_train = manager.create(y_train_norm);
            NDArray y_val   = manager.create(y_val_norm);
            NDArray y_test  = manager.create(y_test_norm);

            Vesta.info("\nDatos finales preparados:");
            Vesta.info("  X_train shape: " + X_train.getShape());
            Vesta.info("  y_train shape: " + y_train.getShape());
            Vesta.info("  X_val shape: " + X_val.getShape());
            Vesta.info("  X_test shape: " + X_test.getShape());

            // Construir modelo (usa tu método existente)

            model.setBlock(getSequentialBlock());
            // Configuración de entrenamiento (igual a tu código)
            MetricsListener metrics = new MetricsListener();
            TrainingConfig config = new DefaultTrainingConfig(new WeightedDirectionLoss("WeightedL2", 5.0f))
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.cosine()
                                    .setBaseValue(0.0001f)
                                    .optFinalValue(0.00001f)
                                    .setMaxUpdates(EPOCH)
                                    .build())
                            .optLearningRateTracker(Tracker.fixed(0.0005f))
                            .optWeightDecays(0.0f)
                            .optClipGrad(2.8f)
                            .build())
                    .addEvaluator(new MAEEvaluator())
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(metrics);

            // Crear datasets con los NDArray ya normalizados (shuffle sólo en train)
            int batchSize = 32*4*2;
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

            // Entrenar
            Vesta.info("Iniciando entrenamiento con " + EPOCH + " epochs...");
            EasyTrain.fit(trainer, EPOCH, trainDataset, valDataset);

            // Gráficas
            ChartUtils.plot("Training Loss/MAE " + String.join(", ", symbols), "epochs",  List.of(new ChartUtils.DataPlot("Loss", metrics.getLoss()), new ChartUtils.DataPlot("MAE", metrics.getMae())));

            // Guardar modelo (igual que antes)
            IOdata.saveModel(model);
            IOdata.saveYNormalizer(yNormalizer);
            IOdata.saveXNormalizer(xNormalizer);

            // Evaluar en conjunto de test si hay muestras
            if (testSize > 0) {
                Vesta.info("\nEvaluando modelo con Backtest Walk-Forward (15% data)...");

                // 1. Crear instancia temporal de PredictionEngine con los datos recién entrenados
                // Nota: Necesitamos un PredictionEngine para usar el método 'predictForBacktest'
                // Como el modelo (PtModel) ya está en memoria, podemos pasarlo directamente.

                // Necesitamos pasar un Model 'genérico', PtModel hereda de Model
                PredictionEngine predEngine = new PredictionEngine(
                        xNormalizer,
                        yNormalizer,
                        model,
                        LOOK_BACK,
                        features
                );

                // 2. Ejecutar Backtest para cada símbolo (o solo el primero si combinaste)
                // Como entrenaste combinando símbolos, lo ideal es probar en uno representativo o iterar.
                // Aquí probamos con el primer símbolo de la lista para obtener el ROI
                String testSymbol = symbols.get(0);
                EngineUtils.ResultsEvaluate evaluate = EngineUtils.evaluateModel(trainer, X_test, y_test, yNormalizer);
                Vesta.MARKETS.clear();
                IOdata.loadMarkets(Main.DATA_SOURCE_FOR_BACK_TEST, symbols);
                Market market = Vesta.MARKETS.get(testSymbol);

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

    public record TrainingTestsResults(EngineUtils.ResultsEvaluate evaluate, BackTestEngine.BackTestResult backtest) {}

    public static SequentialBlock getSequentialBlock() {
        SequentialBlock mainBlock = new SequentialBlock();

        // 1. BACKBONE (Capas comunes que extraen patrones de las velas)
        mainBlock.add(GRU.builder()
                        .setStateSize(256)
                        .setNumLayers(2)
                        .optReturnState(false)
                        .optBatchFirst(true)
                        .optDropRate(0.2f)
                        .build())
                .add(LSTM.builder()
                        .setStateSize(128)
                        .setNumLayers(1)
                        .optReturnState(false)
                        .optBatchFirst(true)
                        .optDropRate(0.2f)
                        .build())
                .add(ndList -> new NDList(ndList.singletonOrThrow().get(":, -1, :"))) // Tomar último estado
                .add(Linear.builder().setUnits(128).build())
                .add(Activation::relu);

        // 2. RAMIFICACIÓN (Aquí dividimos el camino)
        // El ParallelBlock enviará los 128 datos a ambas ramas y luego las juntará
        ParallelBlock branches = new ParallelBlock(list -> {
            // Esta función indica cómo concatenar los resultados de las dos ramas
            NDArray tp = list.get(0).singletonOrThrow();
            NDArray sl = list.get(1).singletonOrThrow();
            NDArray dir = list.get(2).singletonOrThrow();
            return new NDList(
                    NDArrays.concat(new NDList(tp, sl, dir), 1) // ✅ axis = 1
            );
        });

        // TP
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(1).build())
        );

        // SL
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(1).build())
        );

        // Dirección
        branches.add(new SequentialBlock()
                .add(Linear.builder().setUnits(64).build())
                .add(Linear.builder().setUnits(32).build())
                .add(Activation::relu)
                .add(Linear.builder().setUnits(1).build())
                .add(Activation::tanh)
        );

        // Añadimos las ramas al bloque principal
        mainBlock.add(branches);
        return mainBlock;
    }

}
