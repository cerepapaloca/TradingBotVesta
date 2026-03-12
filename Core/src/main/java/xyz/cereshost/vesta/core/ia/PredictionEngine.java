package xyz.cereshost.vesta.core.ia;

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
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.utils.BuilderData;
import xyz.cereshost.vesta.core.ia.utils.YNormalizer;
import xyz.cereshost.vesta.core.ia.utils.XNormalizer;
import xyz.cereshost.vesta.core.io.IOdata;
import xyz.cereshost.vesta.core.trading.TradingManager;
import xyz.cereshost.vesta.core.ia.utils.EngineUtils;
import xyz.cereshost.vesta.core.ia.utils.PredictionUtils;

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
     * Devuelve los valores desnormalizados para los outputs 0 (upMove) y 1 (downMove).
     * Formato: [upMove, downMove, 0, 0, 0]
     */
    public float[] predictRaw(float[][][] inputSequence) {
        NDManager manager = model.getNDManager();
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
            output2D[i][1] = normalizedOutput[base + 1];
            output2D[i][2] = 0f;
            output2D[i][3] = 0f;
            output2D[i][4] = 0f;
        }

        // Des normalizar outputs 0 y 1 (upMove/downMove)
        float[][] denormalized = yNormalizer.inverseTransform(output2D);
        float upMove = denormalized[0][0] * 100.0f;
        float downMove = denormalized[0][1] * 100.0f;
//            manager.close();
        return new float[]{upMove , downMove, 0f, 0f, 0f};
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

        // Inferencia
        float[] rawPredictions = predictRaw(X); // Output del modelo

        float upMovePercent = rawPredictions[0];
        float downMovePercent = rawPredictions[1];

        float currentPrice = (float) subList.getLast().close();
        if (currentPrice <= 0f) {
            return new PredictionResult(currentPrice, currentPrice, currentPrice, 0f, 0f, 0f, 0);
        }

        float upMove = Math.max(0f, upMovePercent) * currentPrice / 100f;
        float downMove = Math.max(0f, downMovePercent) * currentPrice / 100f;

        float maxDownMove = currentPrice * 0.999f;
        if (downMove > maxDownMove) {
            downMove = maxDownMove;
        }

        float ratio = ratioFromMoves(upMove, downMove);
        int direction = PredictionUtils.directionFromRatioRaw(ratio);

        float tpReturn = 0f;
        float slReturn = 0f;
        float tpPrice = currentPrice;
        float slPrice = currentPrice;

        if (direction > 0) {
            tpPrice = currentPrice + upMove;
            slPrice = currentPrice - downMove;
            tpReturn = (tpPrice - currentPrice) / currentPrice;
            slReturn = (currentPrice - slPrice) / currentPrice;
        } else if (direction < 0) {
            tpPrice = currentPrice - downMove;
            slPrice = currentPrice + upMove;
            tpReturn = (currentPrice - tpPrice) / currentPrice;
            slReturn = (slPrice - currentPrice) / currentPrice;
        }
        float confidence = Float.isFinite(ratio) ? Math.abs(ratio) : 0f;
        return new PredictionResult(
                currentPrice, tpPrice, slPrice, tpReturn, slReturn, confidence, direction
        );
    }

    private static float ratioFromMoves(float upMove, float downMove) {
        float maxMove = Math.max(upMove, downMove);
        if (!Float.isFinite(maxMove) || maxMove <= 0f) return 0f;
        float ratio = (upMove - downMove) / maxMove;
        return PredictionUtils.clampRatio(ratio);
    }

    @Data
    @AllArgsConstructor
    public static class PredictionResult {
        private final double currentPrice;
        private final double tpPrice;       // Precio de Take Profit
        private final double slPrice;       // Precio de Stop Loss
        private final double tpReturn;   // Return ratio para TP (positivo)
        private final double slReturn;   // Return ratio para SL (positivo)
        private final double confident;
        private final int direction;   // -1 Short, 0 Neutral, 1 Long

        public DireccionOperation directionOperation() {
            return EngineUtils.directionToOperation(direction);
        }
        public double getTpDistance() {
            return Math.abs(tpPrice - currentPrice);
        }

        public double getSlDistance() {
            return Math.abs(slPrice - currentPrice);
        }

        public double getTpPercent() {
            return tpReturn * 100.0;
        }

        public double getSlPercent() {
            return slReturn * 100.0;
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
