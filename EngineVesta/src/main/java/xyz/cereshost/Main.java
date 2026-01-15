package xyz.cereshost;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
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
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

import static xyz.cereshost.Engine.checkEngines;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    private static final int LOOK_BACK = 10;
    public static final int EPOCH = 500;

    public static void main(String[] args) throws IOException, TranslateException {
        IOdata.loadAll();

        checkEngines();

        Model model = training("BTCUSDT");
    }

    private static @NotNull Model training(@NotNull String symbol) throws TranslateException, IOException {
        // Primero verificar qué engines están disponibles
        Engine TensorFlow = Engine.getEngine("PyTorch");
        if (TensorFlow == null) {
            System.err.println("PyTorch no está disponible. Engines disponibles:");
            for (String engine : Engine.getAllEngines()) {
                System.err.println("  - " + engine);
            }
            throw new RuntimeException("PyTorch engine no encontrado");
        }

        Device device = Device.gpu();
        System.out.println("Usando dispositivo: " + device);

        try (NDManager manager = NDManager.newBaseManager(device)) {
            List<Candle> candles = DatasetBuilder.to1mCandles(Utils.MARKETS.get(symbol));

            if (candles.size() <= LOOK_BACK + 1) {
                throw new IllegalStateException(
                        "Not enough candles: " + candles.size() +
                                " for lookback=" + LOOK_BACK
                );
            }

            Pair<float[][][], float[][]> pair = DatasetBuilder.build(candles, LOOK_BACK);
            float[][][] Xraw = pair.getKey();
            float[][] yraw = pair.getValue();

            // 1. Normalizar X robustamente
            RobustNormalizer xNormalizer = new RobustNormalizer();
            xNormalizer.fit(Xraw);
            float[][][] Xnorm = xNormalizer.transform(Xraw);

            // 2. Normalizar y con estadísticas simples
            // Calcular retornos porcentuales en lugar de precios absolutos
            for (int i = 0; i < yraw.length; i++) {
                // Convertir a retorno porcentual
                // yraw[i][0] = open, yraw[i][1] = close
                // Normalizar a [-1, 1] aproximado
                yraw[i][0] = yraw[i][0] / 1000f; // Ajusta este divisor según tu rango de precios
                yraw[i][1] = yraw[i][1] / 1000f;
            }

            // 3. Verificar que no haya NaN
            for (int i = 0; i < Xnorm.length; i++) {
                for (int j = 0; j < Xnorm[0].length; j++) {
                    for (int k = 0; k < Xnorm[0][0].length; k++) {
                        if (Float.isNaN(Xnorm[i][j][k]) || Float.isInfinite(Xnorm[i][j][k])) {
                            Xnorm[i][j][k] = 0f;
                        }
                    }
                }
            }

            Normalizer normalizer = new Normalizer();
            normalizer.fit(Xraw, yraw);
            float[][] ynorm = normalizer.transformY(yraw);

            // Usar Xnorm y ynorm en lugar de Xraw y yraw
            int samples = Xnorm.length;
            int lookback = Xnorm[0].length;
            int features = Xnorm[0][0].length;

            // Crear arrays planos desde Xnorm
            float[] Xflat = new float[samples * lookback * features];
            int idx = 0;

            for (float[][] floats : Xnorm) {
                for (int j = 0; j < lookback; j++) {
                    for (int k = 0; k < features; k++) {
                        Xflat[idx++] = floats[j][k];
                    }
                }
            }

            NDArray x = manager.create(Xflat, new Shape(samples, lookback, features));
            NDArray y = manager.create(ynorm); // Usar y normalizado

            System.out.println("Datos preparados:");
            System.out.println("  X shape: " + x.getShape());
            System.out.println("  y shape: " + y.getShape());
            System.out.println("  Candles: " + candles.size());
            System.out.println("  Samples: " + samples);
            System.out.println("  Lookback: " + lookback);
            System.out.println("  Features: " + features);

            // Construir el modelo LSTM
            SequentialBlock block = new SequentialBlock()
                    .add(LSTM.builder()
                            .setStateSize(256)
                            .setNumLayers(8)
                            .optReturnState(false)
                            .optBatchFirst(true)
                            .optDropRate(0.3f)
                            .build())
                    .add(Blocks.batchFlattenBlock())
                    .add(Dropout.builder().optRate(0.2f).build())
                    .add(Linear.builder().setUnits(32).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(16).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(4).build());
            // Crear modelo especificando PyTorch EXPLÍCITAMENTE
            Model model = Model.newInstance("vesta-lstm", device, "PyTorch");
            model.setBlock(block);

            MetricsListener metrics = new MetricsListener();
            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.fixed(0.0001f))
                            .optWeightDecays(0.001f)
                            .optClipGrad(1.0f)  // ¡IMPORTANTE!
                            .build())
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.cosine()
                                    .setBaseValue(0.001f)  // Learning rate inicial
                                    .optFinalValue(0.0001f) // Learning rate final
                                    .setMaxUpdates(EPOCH)     // Número total de epochs
                                    .build())
                            .optWeightDecays(0.001f)
                            .build())
                    .addEvaluator(new MAEEvaluator())
                    .optDevices(new Device[]{device})
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(metrics);

            // split train/val (ejemplo 80/20)
            int split = (int) (samples * 0.8);
            NDArray X_train = x.get(new NDIndex("0:" + split));
            NDArray y_train = y.get(new NDIndex("0:" + split));

            NDArray X_val = x.get(new NDIndex(split + ":" + samples));
            NDArray y_val = y.get(new NDIndex(split + ":" + samples));

            int batchSize = 32;
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
            System.out.println("Iniciando entrenamiento...");
            EasyTrain.fit(trainer, EPOCH, trainDataset, valDataset);

            ChartUtils.plot(
                    "Training Loss",
                    "Loss",
                    metrics.getLoss()
            );

            ChartUtils.plot(
                    "Training MAE",
                    "MAE",
                    metrics.getMae()
            );


            // Guardar
            Path modelDir = Paths.get("models");
            Files.createDirectories(modelDir);
            model.save(modelDir, symbol);

            System.out.println("Modelo guardado en: " + modelDir.resolve(symbol).toAbsolutePath());

            return model;
        }
    }
}