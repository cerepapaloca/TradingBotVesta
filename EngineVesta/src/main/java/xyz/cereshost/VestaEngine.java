package xyz.cereshost;

import ai.djl.Device;
import ai.djl.Model;
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
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.DatasetBuilder;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class VestaEngine {

    private static final int LOOK_BACK = 35;
    public static final int EPOCH = 300;

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

        try (NDManager manager = NDManager.newBaseManager(device)) {
            // Combinar datos de todos los símbolos
            List<float[][][]> allX = new ArrayList<>();
            List<float[]> allY = new ArrayList<>(); // Cambiado: float[][] -> float[]

            for (String symbol : symbols) {
                try {
                    Vesta.info("\nProcesando símbolo: " + symbol);
                    List<Candle> candles = DatasetBuilder.to1mCandles(Vesta.MARKETS.get(symbol));

                    if (candles.size() <= LOOK_BACK + 1) {
                        Vesta.error("Símbolo " + symbol + " no tiene suficientes velas: " + candles.size());
                        continue;
                    }

                    Pair<float[][][], float[]> pair = DatasetBuilder.build(candles, LOOK_BACK); // Cambiado
                    float[][][] Xraw = pair.getKey();
                    float[] yraw = pair.getValue(); // Cambiado

                    if (Xraw.length > 0) {
                        // Añadir símbolo como característica adicional
                        float[][][] XwithSymbol = EngineUtils.addSymbolFeature(Xraw, symbol, symbols);
                        allX.add(XwithSymbol);
                        allY.add(yraw); // Cambiado
                        Vesta.info("Añadidas " + Xraw.length + " muestras");
                    }
                } catch (Exception e) {
                    Vesta.error("Error procesando símbolo " + symbol + ": " + e.getMessage());
                }
            }

            if (allX.isEmpty()) {
                throw new RuntimeException("No hay datos suficientes de ningún símbolo");
            }

            // Combinar todos los datos
            Pair<float[][][], float[]> combined = EngineUtils.combineDatasets(allX, allY); // Cambiado
            float[][][] Xcombined = combined.getKey();
            float[] ycombined = combined.getValue(); // Cambiado

            Vesta.info("\nDatos combinados:");
            Vesta.info("  Total de muestras: " + Xcombined.length);
            Vesta.info("  Lookback: " + Xcombined[0].length);
            Vesta.info("  Características: " + Xcombined[0][0].length);

            // Normalizar datos combinados
            RobustNormalizer xNormalizer = new RobustNormalizer();
            xNormalizer.fit(Xcombined);
            float[][][] Xnorm = xNormalizer.transform(Xcombined);

            // Normalizar y
            MultiSymbolNormalizer yNormalizer = new MultiSymbolNormalizer(); // Cambiado: ahora usa float[]
            yNormalizer.fit(ycombined); // Cambiado
            float[] ynorm = yNormalizer.transform(ycombined); // Cambiado

            // Verificar que no haya NaN
            EngineUtils.cleanNaNValues(Xnorm);

            // Preparar datos para DJL
            int samples = Xnorm.length;
            int lookback = Xnorm[0].length;
            int features = Xnorm[0][0].length;

            float[] Xflat = EngineUtils.flatten3DArray(Xnorm);
            NDArray x = manager.create(Xflat, new Shape(samples, lookback, features));

            // Cambiado: crear y con shape (samples, 1) en lugar de (samples, 2)
            NDArray y = manager.create(ynorm, new Shape(samples, 1));

            Vesta.info("\nDatos finales preparados:");
            Vesta.info("  X shape: " + x.getShape());
            Vesta.info("  y shape: " + y.getShape());
            Vesta.info("  Samples: " + samples);
            Vesta.info("  Features totales: " + features);

            // Construir modelo (aumentar capacidad para múltiples símbolos)
            SequentialBlock block = getSequentialBlock();

            Model model = Model.newInstance("VestaIA", device, "PyTorch");
            model.setBlock(block);

            // Configuración de entrenamiento
            MetricsListener metrics = new MetricsListener();
            TrainingConfig config = new DefaultTrainingConfig(Loss.l2Loss())
                    .optOptimizer(Optimizer.adam()
                            .optLearningRateTracker(Tracker.cosine()
                                    .setBaseValue(0.001f)
                                    .optFinalValue(0.0001f)
                                    .setMaxUpdates(EPOCH)
                                    .build())
                            .optWeightDecays(0.001f)
                            .optClipGrad(1.0f)
                            .build())
                    .addEvaluator(new MAEEvaluator())
                    .optDevices(new Device[]{device})
                    .addTrainingListeners(TrainingListener.Defaults.logging())
                    .addTrainingListeners(metrics);

            // Split train/val (70/15/15 para train/val/test)
            int trainSize = (int) (samples * 0.7);
            int valSize = (int) (samples * 0.15);

            NDArray X_train = x.get(new NDIndex("0:" + trainSize));
            NDArray y_train = y.get(new NDIndex("0:" + trainSize));

            NDArray X_val = x.get(new NDIndex(trainSize + ":" + (trainSize + valSize)));
            NDArray y_val = y.get(new NDIndex(trainSize + ":" + (trainSize + valSize)));

            NDArray X_test = x.get(new NDIndex((trainSize + valSize) + ":" + samples));
            NDArray y_test = y.get(new NDIndex((trainSize + valSize) + ":" + samples));

            int batchSize = 32;  // Aumentado para múltiples símbolos
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
            ChartUtils.plot("Training Loss", "Loss", metrics.getLoss());
            ChartUtils.plot("Training MAE", "MAE", metrics.getMae());

            // Guardar modelo
            IOdata.saveModel(model);

            // Evaluar en conjunto de test
            if (X_test.size() > 0) {
                Vesta.info("\nEvaluando en conjunto de test...");
                EngineUtils.evaluateModel(trainer, X_test, y_test, yNormalizer);
            }
        }
    }

    private static SequentialBlock getSequentialBlock() {
        return new SequentialBlock()
                .add(LSTM.builder()
                        .setStateSize(1024)  // Aumentado para capturar patrones múltiples
                        .setNumLayers(4)
                        .optReturnState(false)
                        .optBatchFirst(true)
                        .optDropRate(0.5f)
                        .build())
                .add(Blocks.batchFlattenBlock())
                .add(Dropout.builder().optRate(0.03f).build())
                .add(Linear.builder().setUnits(128).build())
                .add(Activation.reluBlock())
                .add(Dropout.builder().optRate(0.02f).build())
                .add(Linear.builder().setUnits(64).build())
                .add(Activation.reluBlock())
                .add(Activation.tanhBlock())
                .add(Linear.builder().setUnits(32).build())
                .add(Activation.reluBlock())
                .add(Linear.builder().setUnits(1).build());
    }
}
