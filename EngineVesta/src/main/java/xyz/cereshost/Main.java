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
    public static final int EPOCH = 5_000;

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

            int samples = Xraw.length;
            int lookback = Xraw[0].length;
            int features = Xraw[0][0].length;

            // Crear arrays
            float[] Xflat = new float[samples * lookback * features];
            int idx = 0;

            for (float[][] floats : Xraw) {
                for (int j = 0; j < lookback; j++) {
                    for (int k = 0; k < features; k++) {
                        Xflat[idx++] = floats[j][k];
                    }
                }
            }

            NDArray x = manager.create(Xflat, new Shape(samples, lookback, features));
            NDArray y = manager.create(yraw);

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
                            .setStateSize(128*4)
                            .setNumLayers(12)
                            .optReturnState(false)
                            .optBatchFirst(true)
                            .optDropRate(0.1f)
                            .build())
                    .add(Blocks.batchFlattenBlock())
                    .add(Linear.builder().setUnits(64).build())
                    .add(Activation.reluBlock())
                    .add(Linear.builder().setUnits(2).build());

            // Crear modelo especificando PyTorch EXPLÍCITAMENTE
            Model model = Model.newInstance("vesta-lstm", device, "PyTorch");
            model.setBlock(block);

            MetricsListener metrics = new MetricsListener();
            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Optimizer.adam()
                            .optWeightDecays(0.001f)
                            .build())
                    .addEvaluator(new Accuracy())
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