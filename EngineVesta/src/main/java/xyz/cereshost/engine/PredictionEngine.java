package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import lombok.AllArgsConstructor;
import lombok.Data;
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
import xyz.cereshost.utils.PredictionUtils;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;


@Getter
public class PredictionEngine {

    public static final double THRESHOLD_PRICE = 0.002; // No se usa
    public static final double THRESHOLD_RELATIVE = 0.5;

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
     * Hace la inferencia en el modelo.
     * Devuelve los valores desnormalizados para el output 0 (magnitud) y raw para el output 2 (ratio).
     * Formato: [magnitud, 0, ratio, 0, 0]
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
                throw new RuntimeException("El modelo debe tener 5 salidas. Forma actual: " + prediction.getShape());
            }

            // Reorganizar el array plano en [batch_size, 5]
            int batch = (int) shape[0];
            float[][] output2D = new float[batch][5];

            for (int i = 0; i < batch; i++) {
                int base = i * 5;
                output2D[i][0] = normalizedOutput[base];
                output2D[i][1] = 0f;
                output2D[i][2] = normalizedOutput[base + 2];
                output2D[i][3] = 0f;
                output2D[i][4] = 0f;
            }

            // Des normalizar solo output 0 (magnitud)
            float[][] denormalized = yNormalizer.inverseTransform(output2D);
            float mag = denormalized[0][0];
            float ratio = output2D[0][2];
            return new float[]{mag, 0f, ratio, 0f, 0f};
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

        List<Candle> subList = candles.subList((candles.size() - (lookBack + 1)), candles.size());

        // Construir entrada
        float[][][] X = new float[1][Math.toIntExact(lookBack)][Math.toIntExact(features - 2)];
        for (int j = 0; j < lookBack; j++) {
            X[0][j] = BuilderData.extractFeatures(subList.get(j + 1), subList.get(j));
        }
        float[][][] XWithSymbol = BuilderData.addSymbolFeature(X, symbol);

        // Inferencia
        float[] rawPredictions = predictRaw(XWithSymbol); // Output del modelo

        float totalMoveRatio = rawPredictions[0];
        float ratioRaw = rawPredictions[2];

        float currentPrice = (float) subList.getLast().close();
        if (currentPrice <= 0f) {
            return new PredictionResult(currentPrice, currentPrice, currentPrice, 0f, 0f, 0f, 0);
        }

        float ratio = PredictionUtils.clampRatio(ratioRaw);
        float totalMove = Math.max(0f, totalMoveRatio) * currentPrice;

        float[] moves = PredictionUtils.splitMoves(totalMove, ratio);
        float upMove = moves[0];
        float downMove = moves[1];

        float maxDownMove = currentPrice * 0.999f;
        if (downMove > maxDownMove) {
            downMove = maxDownMove;
        }

        int direction = PredictionUtils.directionFromRatioRaw(ratio);

        float tpLogReturn = 0f;
        float slLogReturn = 0f;
        float tpPrice = currentPrice;
        float slPrice = currentPrice;

        if (direction > 0) {
            tpPrice = currentPrice + upMove;
            slPrice = currentPrice - downMove;
            tpLogReturn = (float) Math.log(tpPrice / currentPrice);
            slLogReturn = (float) Math.log(currentPrice / slPrice);
        } else if (direction < 0) {
            tpPrice = currentPrice - downMove;
            slPrice = currentPrice + upMove;
            tpLogReturn = (float) Math.log(currentPrice / tpPrice);
            slLogReturn = (float) Math.log(slPrice / currentPrice);
        }
        float confidence = Float.isFinite(ratio) ? Math.abs(ratio) : 0f;
        return new PredictionResult(
                currentPrice, tpPrice, slPrice, tpLogReturn, slLogReturn, confidence, direction
        );
    }

    @Data
    @AllArgsConstructor
    public static class PredictionResult {
        private final double currentPrice;
        private final double tpPrice;       // Precio de Take Profit
        private final double slPrice;       // Precio de Stop Loss
        private final double tpLogReturn;   // Log return para TP (positivo)
        private final double slLogReturn;   // Log return para SL (positivo)
        private final double confident;
        private final int direction;   // -1 Short, 0 Neutral, 1 Long

        public Trading.DireccionOperation directionOperation() {
            return EngineUtils.directionToOperation(direction);
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

    private static int directionFromRatioRaw(float ratio) {
        if (!Float.isFinite(ratio)) return 0;
        if (ratio > 0f) return 1;  // ratio positivo => domina el máximo => LONG
        if (ratio < 0f) return -1; // ratio negativo => domina el mínimo => SHORT
        return 0;
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
