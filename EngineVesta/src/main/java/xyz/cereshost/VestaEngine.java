package xyz.cereshost;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Dataset;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.List;

public class VestaEngine {

    public static final int LOOK_BACK = 35;
    public static final int EPOCH = 10;//300

    /**
     * Entrena un modelo con múltiples símbolos combinados
     */
    public static void trainingModel(
            @NotNull List<String> symbols) throws TranslateException, IOException {

        ai.djl.engine.Engine TensorFlow = ai.djl.engine.Engine.getEngine("PyTorch");
        if (TensorFlow == null) {
            Vesta.error("PyTorch no está disponible. Engines disponibles:");
            for (String engine : ai.djl.engine.Engine.getAllEngines()) {
                System.err.println("  - " + engine);
            }
            throw new RuntimeException("PyTorch engine no encontrado");
        }

        Device device = Device.gpu();
        Vesta.info("Usando dispositivo: " + device);
        Vesta.info("Entrenando con " + symbols.size() + " símbolos: " + symbols);

        Pair<float[][][], float[]> combined = BuilderData.fullBuild(symbols);


        Pair<float[][][], float[]> deduped = EngineUtils.clearData(combined.getKey(), combined.getValue());
        float[][][] xCombined = deduped.getKey();
        float[] yCombined = deduped.getValue();


        ChartUtils.CandleChartUtils.showDataDistribution("Datos Combinados", xCombined, yCombined, "Todos");


        Vesta.info("\nDatos combinados:");
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

        float[] y_train_arr = java.util.Arrays.copyOfRange(yCombined, 0, trainSize);
        float[] y_val_arr   = java.util.Arrays.copyOfRange(yCombined, trainSize, trainSize + valSize);
        float[] y_test_arr  = java.util.Arrays.copyOfRange(yCombined, trainSize + valSize, samples);

        // Normalizadores: FIT sólo con TRAIN
        RobustNormalizer xNormalizer = new RobustNormalizer();
        xNormalizer.fit(X_train_arr); // fit con train solamente

        MultiSymbolNormalizer yNormalizer = new MultiSymbolNormalizer();
        yNormalizer.fit(y_train_arr); // fit con train solamente

        // Transformar train/val/test
        float[][][] X_train_norm = xNormalizer.transform(X_train_arr);
        float[][][] X_val_norm   = xNormalizer.transform(X_val_arr);
        float[][][] X_test_norm  = xNormalizer.transform(X_test_arr);

        float[] y_train_norm = yNormalizer.transform(y_train_arr);
        float[] y_val_norm   = yNormalizer.transform(y_val_arr);
        float[] y_test_norm  = yNormalizer.transform(y_test_arr);

        // Verificar NaN sólo en arrays normalizados (por si acaso)
        EngineUtils.cleanNaNValues(X_train_norm);
        EngineUtils.cleanNaNValues(X_val_norm);
        EngineUtils.cleanNaNValues(X_test_norm);

        try (NDManager manager = NDManager.newBaseManager(device)) {

            // Aplanar 3D -> 1D para crear NDArray con Shape(samples, lookback, features)
            float[] XtrainFlat = EngineUtils.flatten3DArray(X_train_norm);
            float[] XvalFlat   = EngineUtils.flatten3DArray(X_val_norm);
            float[] XtestFlat  = EngineUtils.flatten3DArray(X_test_norm);

            NDArray X_train = manager.create(XtrainFlat, new Shape(trainSize, lookback, features));
            NDArray X_val   = manager.create(XvalFlat,   new Shape(valSize,   lookback, features));
            NDArray X_test  = manager.create(XtestFlat,  new Shape(testSize,  lookback, features));

            // y -> shape (N, 1)
            NDArray y_train = manager.create(y_train_norm, new Shape(trainSize, 1));
            NDArray y_val   = manager.create(y_val_norm,   new Shape(valSize,   1));
            NDArray y_test  = manager.create(y_test_norm,  new Shape(testSize,  1));

            Vesta.info("\nDatos finales preparados:");
            Vesta.info("  X_train shape: " + X_train.getShape());
            Vesta.info("  y_train shape: " + y_train.getShape());
            Vesta.info("  X_val shape: " + X_val.getShape());
            Vesta.info("  X_test shape: " + X_test.getShape());

            // Construir modelo (usa tu método existente)
            Model model = Model.newInstance(Main.NAME_MODEL, device, "PyTorch");
            model.setBlock(getSequentialBlock());

            // Configuración de entrenamiento (igual a tu código)
            MetricsListener metrics = new MetricsListener();
            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.cosine()
                                    .setBaseValue(0.0001f)
                                    .optFinalValue(0.00001f)
                                    .setMaxUpdates(EPOCH)
                                    .build())
                            .optWeightDecays(0.001f)
                            .optClipGrad(2.5f)
                            .build())
                    .addEvaluator(new MAEEvaluator())
                    .optDevices(new Device[]{device})
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(metrics);

            // Crear datasets con los NDArray ya normalizados (shuffle sólo en train)
            int batchSize = 128;
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

            Trainer trainer = model.newTrainer(config);
            trainer.initialize(new Shape(1, LOOK_BACK, features));

            // Entrenar
            Vesta.info("\nIniciando entrenamiento con " + EPOCH + " epochs...");
            EasyTrain.fit(trainer, EPOCH, trainDataset, valDataset);

            // Gráficas
            ChartUtils.plot("Training Loss/MAE", "epochs",  List.of(new ChartUtils.DataPlot("Loss", metrics.getLoss()), new ChartUtils.DataPlot("MAE", metrics.getMae())));

            // Guardar modelo (igual que antes)
            IOdata.saveModel(model);
            IOdata.saveYNormalizer(yNormalizer);
            IOdata.saveXNormalizer(xNormalizer);

            // Evaluar en conjunto de test si hay muestras
            if (testSize > 0) {
                Vesta.info("\nEvaluando en conjunto de test...");
                EngineUtils.evaluateModel(trainer, X_test, y_test, yNormalizer);
            }

            // Cerrar trainer y modelo si es necesario
            trainer.close();
            model.close();
        }
    }

    public static SequentialBlock getSequentialBlock() {
        return new SequentialBlock()
                .add(LSTM.builder()
                        .setStateSize(1024)//2048 / 1024
                        .setNumLayers(4)
                        .optReturnState(false)
                        .optBatchFirst(true)
                        .optDropRate(0.5f)
                        .build())
                .add(ndList -> {
                    NDArray x = ndList.singletonOrThrow();
                    // batchFirst = true → eje 1 es seqLen
                    NDArray last = x.get(":, -1, :");
                    return new NDList(last);
                })
                .add(Dropout.builder().optRate(0.02f).build())
                .add(Linear.builder().setUnits(128).build())
                .add(Activation.reluBlock())
                .add(Dropout.builder().optRate(0.02f).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(32).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(16).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(1).build());
    }
}
