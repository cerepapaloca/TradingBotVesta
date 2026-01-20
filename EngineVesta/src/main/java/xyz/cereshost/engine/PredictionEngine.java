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
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Comparator;
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
    public PredictionResultSimple predictNextPriceDetail(List<Candle> candles) {

        candles.sort(Comparator.comparingLong(Candle::openTime));

        // Necesitamos lookBack + 1 velas para tener la referencia del cierre anterior (relativo)
        if (candles.size() < lookBack + 1) {
            throw new RuntimeException("Historial insuficiente para predecir. Se necesitan " + (lookBack + 1) + " velas.");
        }

        // Tomar las últimas velas necesarias
        List<Candle> subList = candles.subList(candles.size() - (lookBack + 1), candles.size());

        // El modelo espera [1][lookBack][17]
        float[][][] X = new float[1][lookBack][17];

        for (int j = 0; j < lookBack; j++) {
            X[0][j] = BuilderData.extractFeatures(subList.get(j + 1), subList.get(j));
        }

        // Ejecutar la predicción raw (esto ya aplicará el RobustNormalizer de 17 features)
        float predictedLogReturn = predictRaw(X);

        // Des-normalizar para obtener el Log Return real
        float expectedLogReturn = yNormalizer.inverseTransform(new float[]{predictedLogReturn})[0];

        // CALCULAR PRECIO FINAL: Precio = UltimoCierre * exp(LogReturn)
        float currentPrice = (float) subList.get(subList.size() - 1).close();
        float predictedPrice = (float) (currentPrice * Math.exp(expectedLogReturn));

        return new PredictionResultSimple(currentPrice, predictedPrice,0 , expectedLogReturn);
    }

    public record PredictionResultSimple(float currentPrice, float predictedPrice, float percentChange, float logReturn) {
        public float getAbsChange() {
            return predictedPrice - currentPrice;
        }
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

    public record PredictionResult(Model model, RobustNormalizer xNormalizer, MultiSymbolNormalizer yNormalizer, int lookBack, int features, Device device) {

        public void close() {
                if (model != null) model.close();
            }
        }

    // --- Main para probar ---
    public static void makePrediction(String symbol) {
        try {
            long lastUpdate = IOdata.loadMarkets(false, symbol);
            Vesta.info("🖥️ Iniciando prediction para " + symbol);
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
                PredictionResultSimple result = engine.predictNextPriceDetail(BuilderData.to1mCandles(market));
                DecimalFormat df = new DecimalFormat("###,##0.00###$");
                DecimalFormat df2 = new DecimalFormat("###,##0.00###%");
                // Mostrar resultados
                Vesta.info("\n🔮 Análisis de Predicción para " + symbol + ":");
                Vesta.info("  Delay en la predicción de los datos %s",  new DecimalFormat("###,##0.00s").format((System.currentTimeMillis() - lastUpdate)/1000));
                Vesta.info("  Precio Actual (Close):   %s", df.format(result.currentPrice()));
                Vesta.info("  Precio Predicho (t+1):   %s", df.format(result.predictedPrice()));


                String color = result.getAbsChange() > 0 ? "\u001B[32m" : "\u001B[31m"; // Verde o Rojo en consola
                String reset = "\u001B[0m";

                Vesta.info("  Variación Esperada:    %s %s (%s)%s",
                        color, df.format(result.getAbsChange()), df2.format(result.logReturn()), reset);

            } finally {
                fullSystem.close();
            }
        } catch (Exception e) {
            Vesta.error("Error en predicción:");
            e.printStackTrace();
        }
    }
}