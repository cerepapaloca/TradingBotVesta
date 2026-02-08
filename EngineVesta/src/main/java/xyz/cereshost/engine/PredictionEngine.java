package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.utils.BuilderData;
import xyz.cereshost.utils.YNormalizer;
import xyz.cereshost.utils.XNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.io.IOdata;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.utils.EngineUtils;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;

@Getter
public class PredictionEngine {

    public static final double THRESHOLD_PRICE = 0.002;
    public static final double THRESHOLD_RELATIVE = 0.08;

    private final Model model;
    private final XNormalizer xNormalizer;
    private final YNormalizer yNormalizer;
    private final int lookBack;
    private final int features;
    private final Device device;

    public PredictionEngine(XNormalizer xNormalizer, YNormalizer yNormalizer, Model model, int lookBack, int features) {
        this.device = Device.gpu();
        this.model = model;
        this.xNormalizer = xNormalizer;
        this.yNormalizer = yNormalizer;
        this.lookBack = lookBack;
        this.features = features;
    }

    /**
     * Hace la inferencia en el modelo para dos salidas (TP y SL).
     * Devuelve los valores RAW desnormalizados (Log Returns para TP y SL).
     */
    public float[] predictRaw(float[][][] inputSequence) {
        try (NDManager manager = NDManager.newBaseManager(device)) {
            //debugInputData(inputSequence);
            int batchSize = inputSequence.length;
            int sequenceLength = inputSequence[0].length;
            int actualFeatures = inputSequence[0][0].length;
            //debugInputData(inputSequence);
            // Validación de dimensiones
            if (actualFeatures != features) {
                Vesta.warning("⚠️ Advertencia de dimensiones: El modelo espera " + features +
                        " features, pero recibió " + actualFeatures);
            }

            // 1. Normalizar entrada (RobustScaling)
            float[][][] normalizedInput = xNormalizer.transform(inputSequence);

            // 2. Aplanar para DJL
            float[] flatInput = EngineUtils.flatten3DArray(normalizedInput);
            NDArray inputArray = manager.create(flatInput, new Shape(batchSize, sequenceLength, actualFeatures));

            // 3. Forward Pass
            var block = model.getBlock();
            var parameterStore = new ai.djl.training.ParameterStore(manager, false);
            NDList output = block.forward(parameterStore, new NDList(inputArray), false);

            NDArray prediction = output.singletonOrThrow();
            float[] normalizedOutput = prediction.toFloatArray();

            // Verificar la forma de la salida
            long[] shape = prediction.getShape().getShape();
            if (shape[shape.length - 1] != 5) {
                throw new RuntimeException("El modelo debe tener 3 salidas (TP, SL, Dirección). Forma actual: " + prediction.getShape());
            }

            // Reorganizar el array plano en [batch_size, 2]
            int batch = (int) shape[0];
            float[][] output2D = new float[batch][5];

            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < 5; j++) {
                    output2D[i][j] = normalizedOutput[i * 5 + j];
                }
            }

            // Des normalizar salidas
            float[][] denormalized = yNormalizer.inverseTransform(output2D);
            return denormalized[0];
        } catch (Exception e) {
            Vesta.error("Error en predictRaw: " + e.getMessage());
            e.printStackTrace();
            return new float[]{0f, 0f, 0f, 0f, 0f};
        }
    }

    public PredictionResult predictNextPriceDetail(List<Candle> candles, String symbol) {
        candles.sort(Comparator.comparingLong(Candle::openTime));

        if (candles.size() < lookBack + 1) {
            throw new RuntimeException("Historial insuficiente. Se necesitan " + (lookBack + 1) + " velas.");
        }

        List<Candle> subList = candles.subList((int) (candles.size() - (lookBack + 1)), candles.size());

        // Construir entrada
        float[][][] X = new float[1][Math.toIntExact(lookBack)][Math.toIntExact(features - 2)];
        for (int j = 0; j < lookBack; j++) {
            X[0][j] = BuilderData.extractFeatures(subList.get(j + 1), subList.get(j));
        }
        float[][][] XWithSymbol = BuilderData.addSymbolFeature(X, symbol);

        // Inferencia
        float[] rawPredictions = predictRaw(XWithSymbol); // Output del modelo

        float maxLog = rawPredictions[0];
        float minLog = rawPredictions[1];

        float currentPrice = (float) subList.get(subList.size() - 1).close();
        float maxPrice = (float) (currentPrice * Math.exp(maxLog));
        float minPrice = (float) (currentPrice * Math.exp(minLog));

        EngineUtils.DirectionFilterResult decision = EngineUtils.filterDirectionByExtremes(maxLog, minLog);
        int direction = decision.direction();

        float tpLogReturn;
        float slLogReturn;
        float tpPrice;
        float slPrice;

        if (direction > 0) {
            tpLogReturn = decision.maxMove();
            slLogReturn = decision.minMove();
            tpPrice = maxPrice;
            slPrice = minPrice;
        } else if (direction < 0) {
            tpLogReturn = decision.minMove();
            slLogReturn = decision.maxMove();
            tpPrice = minPrice;
            slPrice = maxPrice;
        } else {
            tpLogReturn = 0f;
            slLogReturn = 0f;
            tpPrice = currentPrice;
            slPrice = currentPrice;
        }

        return new PredictionResult(
                currentPrice, tpPrice, slPrice, tpLogReturn, slLogReturn, decision.confidence(), direction
        );
    }

    public record PredictionResult(
            double currentPrice,
            double tpPrice,       // Precio de Take Profit
            double slPrice,       // Precio de Stop Loss
            double tpLogReturn,   // Log return para TP (positivo)
            double slLogReturn,   // Log return para SL (positivo)
            double confident,
            int direction   // -1 Short, 0 Neutral, 1 Long
    ) {
        public Trading.DireccionOperation directionOperation() {
            if (direction > 0) {
                return Trading.DireccionOperation.LONG;
            }
            if (direction < 0) {
                return Trading.DireccionOperation.SHORT;
            }
            return Trading.DireccionOperation.NEUTRAL;
        }
        public double getTpDistance() {
            return Math.abs(tpPrice - currentPrice);
        }

        public double getSlDistance() {
            return Math.abs(slPrice - currentPrice);
        }

        public double getTpPercent() {
            return (float) ((Math.exp(tpLogReturn) - 1.0) * 100.0);
        }

        public double getSlPercent() {
            return (float) ((Math.exp(slLogReturn) - 1.0) * 100.0);
        }

        public double getRatio(){
            return getTpPercent() / getSlPercent();
        }

        public boolean isProfitableSetup() {
            return getRatio() <= 1f;
        }
    }

    public static float computeDirection(float[] probs) {
        float rawDirection = probs[0] - probs[2];

        // 2. Confianza: Solo nos importa si NO es Neutral.
        // Si neutral es 0.9, confidence es 0.1.
        float confidence = 1.0f - probs[1];

        // 3. Resultado directo: Sin umbrales, sin "if", sin "force to 0".
        // Esto permite que el scatter plot muestre nubes reales en lugar de líneas.
        return rawDirection * confidence;
    }

    @Contract("_ -> new")
    public static @NotNull PredictionEngine loadPredictionEngine(String modelName) throws IOException {

        Model model = IOdata.loadModel();
        Pair<XNormalizer, YNormalizer> normalizers = IOdata.loadNormalizers();

        int lookBack = VestaEngine.LOOK_BACK;
        // Ajuste automático de features
        int features = normalizers.getKey().getMedians().length;

        Vesta.info("✅ Sistema completo cargado:");
        Vesta.info("  Modelo: " + modelName);
        Vesta.info("  Lookback: " + lookBack);
        Vesta.info("  Features detectadas: " + features);

        return new PredictionEngine(normalizers.getKey(), normalizers.getValue(), model, lookBack, features);
    }
}
