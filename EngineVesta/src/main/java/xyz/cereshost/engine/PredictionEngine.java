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
import xyz.cereshost.DataSource;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Arrays;
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
     * Diagnóstico completo para depurar problemas NaN
     */
    private void debugInputData(float[][][] inputSequence) {
        Vesta.info("=== DIAGNÓSTICO DE ENTRADA ===");
        Vesta.info("Dimensiones: [%d, %d, %d]",
                inputSequence.length, inputSequence[0].length, inputSequence[0][0].length);

        // Estadísticas por característica
        int featureCount = inputSequence[0][0].length;
        float[] mins = new float[featureCount];
        float[] maxs = new float[featureCount];
        float[] means = new float[featureCount];
        int[] nanCounts = new int[featureCount];

        for (int i = 0; i < featureCount; i++) {
            mins[i] = Float.MAX_VALUE;
            maxs[i] = -Float.MAX_VALUE;
        }

        for (float[][] floats : inputSequence) {
            for (float[] aFloat : floats) {
                for (int k = 0; k < featureCount; k++) {
                    float val = aFloat[k];
                    if (Float.isNaN(val) || Float.isInfinite(val)) {
                        nanCounts[k]++;
                    } else {
                        mins[k] = Math.min(mins[k], val);
                        maxs[k] = Math.max(maxs[k], val);
                        means[k] += val;
                    }
                }
            }
        }

        for (int k = 0; k < featureCount; k++) {
            means[k] /= (inputSequence.length * inputSequence[0].length - nanCounts[k]);
            Vesta.info("Feature %d: Min=%.6f, Max=%.6f, Mean=%.6f, NaN/Inf=%d",
                    k, mins[k], maxs[k], means[k], nanCounts[k]);
        }
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
            var parameterStore = new ai.djl.training.ParameterStore(manager, false);
            NDList output = block.forward(parameterStore, new NDList(inputArray), false);

            // 4. Obtener resultado (ahora con 2 salidas)
            NDArray prediction = output.singletonOrThrow();
            // El modelo debe tener salida [batch_size, 2]
            float[] normalizedOutput = prediction.toFloatArray();

            // Verificar la forma de la salida
            long[] shape = prediction.getShape().getShape();
            if (shape[shape.length - 1] != 2) {
                throw new RuntimeException("El modelo debe tener 2 salidas (TP y SL). Forma actual: " + prediction.getShape());
            }

            // Reorganizar el array plano en [batch_size, 2]
            int batch = (int) shape[0];
            float[][] output2D = new float[batch][2];

            for (int i = 0; i < batch; i++) {
                for (int j = 0; j < 2; j++) {
                    output2D[i][j] = normalizedOutput[i * 2 + j];
                }
            }

            // 5. Desnormalizar salidas (TP y SL)
            float[][] denormalized = yNormalizer.inverseTransform(output2D);
            // Para batch size = 1, retornar la primera fila
            return denormalized[0];
        } catch (Exception e) {
            Vesta.error("Error en predictRaw: " + e.getMessage());
            e.printStackTrace();
            return new float[]{0f, 0f};
        }
    }

    /**
     * Predice TP y SL para el siguiente intervalo.
     * Retorna un objeto con detalles para mejor visualización.
     */
    public PredictionResultTP_SL predictNextPriceDetail(List<Candle> candles, String symbol) {
        candles.sort(Comparator.comparingLong(Candle::openTime));

        // Necesitamos lookBack + 1 velas para tener la referencia del cierre anterior (relativo)
        if (candles.size() < lookBack + 1) {
            throw new RuntimeException("Historial insuficiente para predecir. Se necesitan " + (lookBack + 1) + " velas.");
        }

        // Tomar las últimas velas necesarias
        List<Candle> subList = candles.subList(candles.size() - (lookBack + 1), candles.size());

        // Construir la secuencia de entrada
        float[][][] X = new float[1][lookBack][features - 2]; // -2 porque addSymbolFeature añadirá 2 features

        for (int j = 0; j < lookBack; j++) {
            X[0][j] = BuilderData.extractFeatures(subList.get(j + 1), subList.get(j));
        }

        // Añadir características del símbolo
        float[][][] XWithSymbol = BuilderData.addSymbolFeature(X, symbol);

        // Ejecutar la predicción para obtener TP y SL
        float[] predictions = predictRaw(XWithSymbol);

        if (predictions.length != 2) {
            throw new RuntimeException("Se esperaban 2 predicciones (TP y SL), pero se obtuvieron " + predictions.length);
        }

        float tpLogReturn = predictions[0]; // TP en log return (positivo)
        float slLogReturn = predictions[1]; // SL en log return (positivo)

        // Obtener precio actual
        float currentPrice = (float) subList.get(subList.size() - 1).close();

        // Determinar dirección basada en el ratio TP/SL
        // Si TP > SL, sugiere operación LONG (ganancia potencial mayor que pérdida)
        // Si SL > TP, sugiere operación SHORT (pérdida potencial mayor que ganancia)
        String direction = tpLogReturn > slLogReturn ? "LONG" : "SHORT";

        // Calcular precios absolutos basados en la dirección
        float tpPrice, slPrice;

        if ("LONG".equals(direction)) {
            // Para LONG: TP por encima del precio actual, SL por debajo
            tpPrice = (float) (currentPrice * Math.exp(tpLogReturn));
            slPrice = (float) (currentPrice * Math.exp(-slLogReturn));
        } else {
            // Para SHORT: TP por debajo del precio actual, SL por encima
            tpPrice = (float) (currentPrice * Math.exp(-tpLogReturn));
            slPrice = (float) (currentPrice * Math.exp(slLogReturn));
        }

        // Calcular ratio TP/SL
        float ratio = slLogReturn > 0 ? tpLogReturn / slLogReturn : 0;

        return new PredictionResultTP_SL(
                currentPrice,
                tpPrice,
                slPrice,
                tpLogReturn,
                slLogReturn,
                direction,
                ratio
        );
    }

    public record PredictionResultTP_SL(
            float currentPrice,
            float tpPrice,       // Precio de Take Profit
            float slPrice,       // Precio de Stop Loss
            float tpLogReturn,   // Log return para TP (positivo)
            float slLogReturn,   // Log return para SL (positivo)
            String direction,    // "LONG" o "SHORT"
            float tpSlRatio      // Ratio TP/SL (idealmente > 2)
    ) {
        public float getTpDistance() {
            return Math.abs(tpPrice - currentPrice);
        }

        public float getSlDistance() {
            return Math.abs(slPrice - currentPrice);
        }

        public float getTpPercent() {
            return (float) ((Math.exp(tpLogReturn) - 1.0) * 100.0);
        }

        public float getSlPercent() {
            return (float) ((Math.exp(slLogReturn) - 1.0) * 100.0);
        }

        public boolean isProfitableSetup() {
            return tpSlRatio >= 2.0f; // Ratio 2:1 mínimo para ser rentable
        }
    }

    @Contract("_ -> new")
    public static @NotNull PredictionEngine loadPredictionEngine(String modelName) throws IOException {

        Model model = IOdata.loadModel();
        Pair<RobustNormalizer, MultiSymbolNormalizer> normalizers = IOdata.loadNormalizers();

        int lookBack = VestaEngine.LOOK_BACK;
        // Ajuste automático de features
        int features = normalizers.getKey().getFeatureMedians().length;

        Vesta.info("✅ Sistema completo cargado:");
        Vesta.info("  Modelo: " + modelName);
        Vesta.info("  Lookback: " + lookBack);
        Vesta.info("  Features detectadas: " + features);

        return new PredictionEngine(normalizers.getKey(), normalizers.getValue(), model, lookBack, features);
    }

    // --- Main para probar ---
    public static void makePrediction(String symbol) {
        try {
            long lastUpdate = IOdata.loadMarkets(DataSource.BINANCE, symbol);
            Vesta.info("🖥️ Iniciando prediction para " + symbol);
            PredictionEngine engine = PredictionEngine.loadPredictionEngine("VestaIA");

            try {
                Market market = Vesta.MARKETS.get(symbol);
                if (market == null) {
                    Vesta.error("Mercado no encontrado: " + symbol);
                    return;
                }

                // Realizar predicción detallada
                PredictionResultTP_SL result = engine.predictNextPriceDetail(BuilderData.to1mCandles(market), market.getSymbol());

                DecimalFormat df = new DecimalFormat("###,##0.00###$");
                DecimalFormat dfPercent = new DecimalFormat("###,##0.00###%");
                DecimalFormat dfRatio = new DecimalFormat("###,##0.00");

                // Determinar color para consola
                String colorGreen = "\u001B[32m";
                String colorRed = "\u001B[31m";
                String colorYellow = "\u001B[33m";
                String reset = "\u001B[0m";

                String directionColor = "LONG".equals(result.direction()) ? colorGreen : colorRed;
                String profitableColor = result.isProfitableSetup() ? colorGreen : colorRed;

                // Mostrar resultados
                Vesta.info("\n🔮 Análisis de Predicción para " + symbol + ":");
                Vesta.info("  Delay en la predicción de los datos: %s",
                        new DecimalFormat("###,##0.00s").format((System.currentTimeMillis() - lastUpdate)/1000));
                Vesta.info("  Precio Actual:            %s", df.format(result.currentPrice()));

                Vesta.info("\n  📈 TAKE PROFIT (TP):");
                Vesta.info("    Precio TP:              %s", df.format(result.tpPrice()));
                Vesta.info("    Distancia TP:           %s (%s)",
                        df.format(result.getTpDistance()), dfPercent.format(result.getTpPercent()));
                Vesta.info("    Log Return TP:          %s", dfPercent.format(result.tpLogReturn()));

                Vesta.info("\n  📉 STOP LOSS (SL):");
                Vesta.info("    Precio SL:              %s", df.format(result.slPrice()));
                Vesta.info("    Distancia SL:           %s (%s)",
                        df.format(result.getSlDistance()), dfPercent.format(result.getSlPercent()));
                Vesta.info("    Log Return SL:          %s", dfPercent.format(result.slLogReturn()));

                Vesta.info("\n  📊 ESTADÍSTICAS:");
                Vesta.info("    Dirección sugerida:     %s%s%s",
                        directionColor, result.direction(), reset);
                Vesta.info("    Ratio TP/SL:            %s%s:1%s",
                        profitableColor, dfRatio.format(result.tpSlRatio()), reset);
                Vesta.info("    Setup rentable:         %s%s%s",
                        profitableColor, result.isProfitableSetup() ? "SÍ" : "NO", reset);

                if (result.isProfitableSetup()) {
                    Vesta.info("\n  ✅ SETUP RECOMENDADO - Ratio favorable");
                } else {
                    Vesta.info("\n  ⚠️  SETUP NO RECOMENDADO - Ratio insuficiente");
                }

            } finally {
                engine.getModel().close();
            }
        } catch (Exception e) {
            Vesta.error("Error en predicción:");
            e.printStackTrace();
        }
    }

    /**
     * Predice múltiples pasos hacia adelante
     */
    public PredictionResultTP_SL[] predictMultipleSteps(List<Candle> candles, String symbol, int steps) {
        PredictionResultTP_SL[] results = new PredictionResultTP_SL[steps];
        List<Candle> currentCandles = new java.util.ArrayList<>(candles);

        for (int i = 0; i < steps; i++) {
            try {
                results[i] = predictNextPriceDetail(currentCandles, symbol);

                // Añadir una vela "simulada" para el siguiente paso (usando predicción como cierre)
                // Esto es una simplificación, en producción usarías datos reales
                if (i < steps - 1) {
                    Candle lastCandle = currentCandles.get(currentCandles.size() - 1);
                    Candle simulatedCandle = createSimulatedCandle(lastCandle, results[i]);
                    currentCandles.add(simulatedCandle);
                }
            } catch (Exception e) {
                Vesta.error("Error en predicción del paso " + (i+1) + ": " + e.getMessage());
                results[i] = null;
            }
        }

        return results;
    }

    private Candle createSimulatedCandle(Candle lastCandle, PredictionResultTP_SL prediction) {
        // Crear una vela simulada basada en la predicción
        // Esto es una simplificación para demostración
        return new Candle(
                lastCandle.openTime() + 60000L, // +1 minuto
                lastCandle.close(), // open = último close
                Math.max(prediction.tpPrice(), prediction.currentPrice()), // high
                Math.min(prediction.slPrice(), prediction.currentPrice()), // low
                "LONG".equals(prediction.direction()) ? prediction.tpPrice() : prediction.slPrice(), // close
                lastCandle.amountTrades(),
                lastCandle.volumeBase(),
                lastCandle.quoteVolume(),
                lastCandle.buyQuoteVolume(),
                lastCandle.sellQuoteVolume(),
                lastCandle.deltaUSDT(),
                lastCandle.buyRatio(),
                lastCandle.bidLiquidity(),
                lastCandle.askLiquidity(),
                lastCandle.depthImbalance(),
                lastCandle.midPrice(),
                lastCandle.spread(),
                lastCandle.rsi4(),
                lastCandle.rsi8(),
                lastCandle.rsi16(),
                lastCandle.macdVal(),
                lastCandle.macdSignal(),
                lastCandle.macdHist(),
                lastCandle.nvi()
        );
    }
}