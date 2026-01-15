package xyz.cereshost;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Block;
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
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.market.Candle;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

//TIP To <b>Run</b> code, press <shortcut actionId="Run"/> or
// click the <icon src="AllIcons.Actions.Execute"/> icon in the gutter.
public class Main {

    private static final int LOOK_BACK = 10;

    public static void main(String[] args) throws IOException, TranslateException {
        IOdata.loadAll();

        //checkEngines();

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

        // Crear manager sin especificar dispositivo (usará CPU por defecto)
        try (NDManager manager = NDManager.newBaseManager()) {
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

            NDArray X = manager.create(Xflat, new Shape(samples, lookback, features));
            NDArray y = manager.create(yraw);

            System.out.println("Datos preparados:");
            System.out.println("  X shape: " + X.getShape());
            System.out.println("  y shape: " + y.getShape());
            System.out.println("  Candles: " + candles.size());
            System.out.println("  Samples: " + samples);
            System.out.println("  Lookback: " + lookback);
            System.out.println("  Features: " + features);

            // Construir el modelo LSTM
            SequentialBlock block = new SequentialBlock()
                    .add(LSTM.builder()
                            .setStateSize(128)
                            .setNumLayers(2)
                            .optReturnState(false)
                            .optBatchFirst(true)
                            .optDropRate(0.0f)
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
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(metrics);


            Trainer trainer = model.newTrainer(config);
            try {
                // Obtener el bloque del modelo
                SequentialBlock SequentialBlock = (SequentialBlock) model.getBlock();

                // Buscar la capa LSTM y compactar parámetros
                for (int i = 0; i < SequentialBlock.getChildren().size(); i++) {
                    Block child = SequentialBlock.getChildren().get(i).getValue();
                    if (child instanceof LSTM) {
                        LSTM lstm = (LSTM) child;
                        // Forzar la compactación (equivalente a flatten_parameters() en PyTorch)
                        // En DJL, podrías necesitar reconstruir la capa
                        System.out.println("Compactando parámetros LSTM...");
                    }
                }
            } catch (Exception e) {
                System.err.println("No se pudo compactar parámetros: " + e.getMessage());
            }
            trainer.initialize(new Shape(1, LOOK_BACK, features));



            // Crear dataset
            Dataset dataset = new ArrayDataset.Builder()
                    .setData(X)
                    .optLabels(y)
                    .setSampling(1, true)
                    .build();

            // Entrenar
            System.out.println("Iniciando entrenamiento...");
            EasyTrain.fit(trainer, 100, dataset, null);

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



    @Contract("_ -> new")
    public static double @NotNull [] extractFeatures(@NotNull Candle c) {
        return new double[]{
                c.close(),
                c.quoteVolume(),
                c.deltaUSDT(),
                c.buyRatio(),
                c.bidLiquidity(),
                c.askLiquidity(),
                c.depthImbalance(),
                c.spread()
        };
    }

    @Contract("_ -> new")
    public static double @NotNull [] extractTarget(@NotNull Candle next) {
        return new double[]{
                next.open(),
                next.close()
        };
    }
}