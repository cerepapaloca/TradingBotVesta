package xyz.cereshost;

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
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Getter
public class PredictionEngine {

    private final Model model;
    private final RobustNormalizer xNormalizer;
    private final MultiSymbolNormalizer yNormalizer;
    private final int lookBack;
    private final int features;
    private final Device device;

    public PredictionEngine(RobustNormalizer xNormalizer, MultiSymbolNormalizer yNormalizer, Model model, int lookBack, int features) {
        this.device = Device.gpu();
        this.model = model;
        this.xNormalizer = xNormalizer;
        this.yNormalizer = yNormalizer;
        this.lookBack = lookBack;
        this.features = features;
    }

    /**
     * Hace la inferencia en el modelo.
     * Devuelve el valor RAW desnormalizado (en este caso, el Log Return).
     */
    public float predictRaw(float[][][] inputSequence) {
        try (NDManager manager = NDManager.newBaseManager(device)) {
            int batchSize = inputSequence.length;
            int sequenceLength = inputSequence[0].length;
            int actualFeatures = inputSequence[0][0].length;

            // Validación de dimensiones
            if (actualFeatures != features) {
                Vesta.waring("⚠️ Advertencia de dimensiones: El modelo espera " + features +
                        " features, pero recibió " + actualFeatures);
            }

            // 1. Normalizar entrada (RobustScaling)
            float[][][] normalizedInput = xNormalizer.transform(inputSequence);

            // 2. Aplanar para DJL
            float[] flatInput = EngineUtils.flatten3DArray(normalizedInput);
            NDArray inputArray = manager.create(flatInput, new Shape(batchSize, sequenceLength, actualFeatures));

            // 3. Forward Pass
            var block = model.getBlock();
            var parameterStore = new ai.djl.training.ParameterStore(manager, false); // False = modo inferencia
            NDList output = block.forward(parameterStore, new NDList(inputArray), false);

            // 4. Obtener resultado
            NDArray prediction = output.singletonOrThrow();
            float normalizedOutput = prediction.toFloatArray()[0];

            // 5. Desnormalizar salida (Convierte de escala normalizada a Log Return real)
            // Nota: inverseTransform espera un array, devolvemos el primer elemento
            float[] denormalized = yNormalizer.inverseTransform(new float[]{normalizedOutput});

            return denormalized[0];
        }
    }

    /**
     * Predice el precio absoluto del siguiente intervalo.
     * Retorna un objeto con detalles para mejor visualización.
     */
    public PredictionDetail predictNextPriceDetail(Market market) {
        // 1. Obtener velas recientes
        List<Candle> candles = BuilderData.to1mCandles(market);

        if (candles.size() < lookBack) {
            throw new RuntimeException("Insuficientes datos para predecir. Se requieren " + lookBack + " velas.");
        }

        // 2. Extraer la última ventana de 'lookBack' velas
        // Tomamos desde (size - lookback) hasta el final
        List<Candle> windowCandles = candles.subList(candles.size() - lookBack, candles.size());

        // 3. Construir array de features (X) manualmente
        // No usamos fullBuild aquí para evitar procesar todo el historial
        float[][] seq = new float[lookBack][];
        for (int i = 0; i < lookBack; i++) {
            double[] feats = BuilderData.extractFeatures(windowCandles.get(i));
            // Convertir double[] a float[]
            float[] floatFeats = new float[feats.length];
            for (int k = 0; k < feats.length; k++) floatFeats[k] = (float) feats[k];
            seq[i] = floatFeats;
        }

        // 4. Añadir features del símbolo (OneHot) tal como en el entrenamiento
        // Envolvemos en array 3D [1, lookback, features]
        float[][][] rawInput = new float[][][]{seq};

        // IMPORTANTE: Debemos pasar la lista de símbolos usada en entrenamiento para que el OneHot coincida.
        // Si solo entrenaste con BTCUSDT, la lista es List.of("BTCUSDT").
        List<String> trainingSymbols = List.of("BTCUSDT"); // TODO: Pasar esto dinámicamente si usas más símbolos
        float[][][] inputWithSymbol = EngineUtils.addSymbolFeature(rawInput, market.getSymbol(), trainingSymbols);

        // 5. Obtener predicción (Log Return)
        float predictedLogReturn = predictRaw(inputWithSymbol);

        // 6. Reconstruir precio absoluto
        // Precio Futuro = Precio Actual * e^(log_return)
        double currentPrice = candles.get(candles.size() - 1).close();
        double predictedPrice = currentPrice * Math.exp(predictedLogReturn);

        return new PredictionDetail(
                (float) currentPrice,
                (float) predictedPrice,
                predictedLogReturn
        );
    }

    @Contract("_ -> new")
    public static @NotNull PredictionResult loadFullModel(String modelName) throws IOException {
        Device device = Device.gpu();

        Model model = IOdata.loadModel();
        Pair<RobustNormalizer, MultiSymbolNormalizer> normalizers = IOdata.loadNormalizers();

        int lookBack = VestaEngine.LOOK_BACK;
        // Ajuste automático de features:
        // Si entrenaste con Symbol Features, el normalizador sabe la dimensión correcta.
        // xNormalizer.getFeatureMedians().length nos da la dimensión exacta esperada.
        int features = normalizers.getKey().getFeatureMedians().length;

        Vesta.info("✅ Sistema completo cargado:");
        Vesta.info("  Modelo: " + modelName);
        Vesta.info("  Lookback: " + lookBack);
        Vesta.info("  Features detectadas: " + features);

        return new PredictionResult(model, normalizers.getKey(), normalizers.getValue(), lookBack, features, device);
    }

    // --- Clases de soporte ---

    public record PredictionDetail(float currentPrice, float predictedPrice, float logReturn) {
        public float getAbsChange() { return predictedPrice - currentPrice; }
        public float getPercentChange() { return (predictedPrice - currentPrice) / currentPrice * 100f; }
    }

    @Getter
    public static class PredictionResult {
        public final Model model;
        public final RobustNormalizer xNormalizer;
        public final MultiSymbolNormalizer yNormalizer;
        public final int lookBack;
        public final int features;
        public final Device device;

        public PredictionResult(Model model, RobustNormalizer xNormalizer,
                                MultiSymbolNormalizer yNormalizer, int lookBack,
                                int features, Device device) {
            this.model = model;
            this.xNormalizer = xNormalizer;
            this.yNormalizer = yNormalizer;
            this.lookBack = lookBack;
            this.features = features;
            this.device = device;
        }

        public void close() {
            if (model != null) model.close();
        }
    }

    // --- Main para probar ---
    public static void makePrediction(String symbol) {
        try {
            PredictionEngine.PredictionResult fullSystem = PredictionEngine.loadFullModel("VestaIA");
            // Usamos las features detectadas automáticamente en loadFullModel
            PredictionEngine engine = new PredictionEngine(
                    fullSystem.xNormalizer,
                    fullSystem.yNormalizer,
                    fullSystem.model,
                    fullSystem.lookBack,
                    fullSystem.features
            );

            try {
                Market market = Vesta.MARKETS.get(symbol);
                if (market == null) {
                    Vesta.error("Mercado no encontrado: " + symbol);
                    return;
                }

                // Realizar predicción detallada
                PredictionDetail result = engine.predictNextPriceDetail(market);

                // Mostrar resultados
                Vesta.info("\n🔮 Análisis de Predicción para " + symbol + ":");
                Vesta.info("  Precio Actual (Close):   $%.2f", result.currentPrice());
                Vesta.info("  Precio Predicho (t+1):   $%.2f", result.predictedPrice());

                String directionIcon = result.getPercentChange() > 0 ? "🚀 Subida" : "🔻 Bajada";
                String color = result.getPercentChange() > 0 ? "\u001B[32m" : "\u001B[31m"; // Verde o Rojo en consola
                String reset = "\u001B[0m";

                Vesta.info("  Tendencia: " + directionIcon);
                Vesta.info("  Variación Esperada:      %s%+.2f$ (%+.3f%%)%s",
                        color, result.getAbsChange(), result.getPercentChange(), reset);

            } finally {
                fullSystem.close();
            }
        } catch (Exception e) {
            Vesta.error("Error en predicción:");
            e.printStackTrace();
        }
    }
}