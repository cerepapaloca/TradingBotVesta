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
import xyz.cereshost.trading.Trading;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Comparator;
import java.util.List;

@Getter
public class PredictionEngine {

    public static final double THRESHOLD_PRICE = 0.001;
    public static final double THRESHOLD_RELATIVE = 0.08;

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

        List<Candle> subList = candles.subList(candles.size() - (lookBack + 1), candles.size());

        // Construir entrada
        float[][][] X = new float[1][lookBack][features - 2];
        for (int j = 0; j < lookBack; j++) {
            X[0][j] = BuilderData.extractFeatures(subList.get(j + 1), subList.get(j));
        }
        float[][][] XWithSymbol = BuilderData.addSymbolFeature(X, symbol);

        // Inferencia
        float[] rawPredictions = predictRaw(XWithSymbol); // Output del modelo


        float upForce = rawPredictions[0];   // TP real
        float downForce = rawPredictions[1]; // SL real
        float probLong = rawPredictions[2]; // Long
        float probNeutral = rawPredictions[3]; // Neutro
        float probShort = rawPredictions[4]; // Short
        Trading.DireccionOperation direction;
        // 3. Lógica de Negocio
        float[] directionProbs = {probLong, probNeutral, probShort};
        float directionValue = computeDirection(directionProbs);

        boolean signalLong = directionValue > THRESHOLD_RELATIVE;
        boolean signalShort = directionValue < -THRESHOLD_RELATIVE;

        float currentPrice = (float) subList.get(subList.size() - 1).close();
        float tpLogReturn, slLogReturn;
        if (signalLong) {
            direction = Trading.DireccionOperation.LONG;
            tpLogReturn = upForce;
            slLogReturn = downForce;
        } else if (signalShort) {
            direction = Trading.DireccionOperation.SHORT;
            tpLogReturn = downForce;
            slLogReturn = upForce;
        } else {
            direction = Trading.DireccionOperation.NEUTRAL;
            tpLogReturn = 0;
            slLogReturn = 0;
        }

        // Cálculos de precios finales
        float tpPrice = (float) (currentPrice * Math.exp(direction.equals(Trading.DireccionOperation.SHORT) ? -tpLogReturn : tpLogReturn));
        float slPrice = (float) (currentPrice * Math.exp(direction.equals(Trading.DireccionOperation.SHORT) ? slLogReturn : -slLogReturn));

        // Fix visual para Neutral
        //if(direction.equals(DireccionOperation.NEUTRAL)) { tpPrice = currentPrice; slPrice = currentPrice; }

        return new PredictionResult(
                currentPrice, tpPrice, slPrice, tpLogReturn, slLogReturn, direction
        );
    }

    public record PredictionResult(
            double currentPrice,
            double tpPrice,       // Precio de Take Profit
            double slPrice,       // Precio de Stop Loss
            double tpLogReturn,   // Log return para TP (positivo)
            double slLogReturn,   // Log return para SL (positivo)
            Trading.DireccionOperation direction   // "LONG" o "SHORT"
    ) {
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

    public static float computeDirection(float[] probs, float temperature) {
        if (probs.length != 3) {
            throw new IllegalArgumentException("El array debe tener 3 probabilidades: [long, neutral, short]");
        }

        float probLong = probs[0];
        float probNeutral = probs[1];
        float probShort = probs[2];

        // 1. Calcular fuerza bruta: diferencia entre long y short
        float rawForce = probLong - probShort;

        // 2. Calcular factor de certidumbre basado en neutralidad y dispersión
        // - Si neutral es alta, certidumbre baja
        // - Si hay empate entre las 3, certidumbre muy baja
        float maxProb = Math.max(Math.max(probLong, probNeutral), probShort);
        float dispersion = 0f;

        // Calcular dispersión (entropía simple)
        if (maxProb > 0) {
            float[] normalized = new float[3];
            float sum = probLong + probNeutral + probShort;
            if (sum > 0) {
                normalized[0] = probLong / sum;
                normalized[1] = probNeutral / sum;
                normalized[2] = probShort / sum;

                // Entropía simple (no logarítmica para velocidad)
                dispersion = 0f;
                for (float p : normalized) {
                    if (p > 0) {
                        dispersion += p * (1 - p); // Varianza por categoría
                    }
                }
                dispersion /= 3.0f; // Normalizar a [0, 0.25]
            }
        }

        // 3. Certidumbre combina: baja neutralidad + baja dispersión
        float certainty = (1 - probNeutral) * (1 - dispersion * 4); // Escalar dispersion a [0,1]
        certainty = Math.max(0, Math.min(1, certainty));

        // 4. Aplicar función sigmoide ajustada por temperatura
        // Con temperatura=1: sigmoide estándar
        // Temperatura >1: más plana (conservadora)
        // Temperatura <1: más pronunciada (decisiva)
        float scaledForce = rawForce * temperature;

        // Sigmoide ajustada para mantener escala [-1, 1]
        float sigmoid;
        if (scaledForce >= 0) {
            sigmoid = 2 / (1 + (float)Math.exp(-scaledForce)) - 1;
        } else {
            sigmoid = -2 / (1 + (float)Math.exp(scaledForce)) + 1;
        }

        // 5. Combinar: dirección = sigmoide * certidumbre
        float direction = sigmoid * certainty;

        // 6. Umbral de neutralidad: si está muy empatado, forzar a 0
        float maxDiff = Math.abs(probLong - probNeutral);
        maxDiff = Math.max(maxDiff, Math.abs(probNeutral - probShort));
        maxDiff = Math.max(maxDiff, Math.abs(probLong - probShort));

        // Si la máxima diferencia es pequeña (<0.3), considerar empate
        if (maxDiff < 0.3f) {
            // Reducir gradualmente hacia 0 según qué tan empatado esté
            direction *= (maxDiff / 0.3f);
        }

        // 7. Para casos extremadamente empatados como [0.3, 0.2, 0.2]
        float totalDiff = Math.abs(probLong - probNeutral) +
                Math.abs(probNeutral - probShort) +
                Math.abs(probLong - probShort);

        if (totalDiff < 0.5f) {
            direction *= 0.1f; // Casi forzar a neutral
        }

        return Math.max(-1, Math.min(1, (float) Math.tanh(direction)*2));
    }

    public static float computeDirection(float[] probs) {
        return computeDirection(probs, 100f);
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
            Market market = IOdata.loadMarkets(DataSource.BINANCE, symbol);
            Vesta.info("🖥️ Iniciando prediction para " + symbol);
            PredictionEngine engine = PredictionEngine.loadPredictionEngine("VestaIA");

            try {

                if (market == null) {
                    Vesta.error("Mercado no encontrado: " + symbol);
                    return;
                }

                // Realizar predicción detallada
                PredictionResult result = engine.predictNextPriceDetail(BuilderData.to1mCandles(market), market.getSymbol());

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
                        profitableColor, dfRatio.format(result.getRatio()), reset);
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
}