package xyz.cereshost;

import ai.djl.Device;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.training.Trainer;
import ai.djl.util.Pair;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.common.Vesta;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

@UtilityClass
public class EngineUtils {

    public static void checkEngines() {
        Vesta.info("=== Verificando Engines DJL ===");

        // Listar todos los engines disponibles
        for (String engineName : ai.djl.engine.Engine.getAllEngines()) {
            Vesta.info("\nEngine: " + engineName);
            ai.djl.engine.Engine engine = ai.djl.engine.Engine.getEngine(engineName);
            if (engine != null) {
                Vesta.info("  Version: " + engine.getVersion());
                Vesta.info("  Dispositivos disponibles:");

                for (Device device : engine.getDevices()) {
                    Vesta.info("    - " + device +
                            " (GPU: " + device.isGpu() +
                            ", ID: " + device.getDeviceId() +
                            ", C: " + engine.hasCapability(StandardCapabilities.CUDA) + ")");
                }
            } else {
                Vesta.info("  No disponible");
            }
        }
    }

    /**
     * Mezcla los datos aleatoriamente
     */
    public void shuffleData(float[][][] X, float[] y) { // Cambiado
        Random rand = new Random(42); // Semilla para reproducibilidad
        for (int i = X.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            // Intercambiar X
            float[][] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            // Intercambiar y
            float tempY = y[i]; // Cambiado
            y[i] = y[j];
            y[j] = tempY;
        }
    }

    /**
     * Aplanar array 3D a 1D
     */
    public float @NotNull [] flatten3DArray(float[][] @NotNull [] array) {
        int samples = array.length;
        int lookback = array[0].length;
        int features = array[0][0].length;

        float[] flat = new float[samples * lookback * features];
        int idx = 0;

        for (float[][] sample : array) {
            for (int j = 0; j < lookback; j++) {
                System.arraycopy(sample[j], 0, flat, idx, features);
                idx += features;
            }
        }
        return flat;
    }

    /**
     * Limpiar valores NaN
     */
    public static void cleanNaNValues(float[][] @NotNull [] array) {
        for (float[][] sample : array) {
            for (float[] timestep : sample) {
                for (int k = 0; k < timestep.length; k++) {
                    if (Float.isNaN(timestep[k]) || Float.isInfinite(timestep[k])) {
                        timestep[k] = 0f;
                    }
                }
            }
        }
    }

    /**
     * Combina múltiples datasets en uno solo
     */
    @Contract("_, _ -> new")
    public static @NotNull Pair<float[][][], float[]> combineDatasets(@NotNull List<float[][][]> allX, List<float[]> allY) {

        int totalSamples = 0;
        for (float[][][] X : allX) {
            totalSamples += X.length;
        }

        // Verificar dimensiones consistentes
        int lookback = allX.get(0)[0].length;
        int features = allX.get(0)[0][0].length;

        float[][][] Xcombined = new float[totalSamples][lookback][features];
        float[] ycombined = new float[totalSamples]; // Cambiado

        int currentIndex = 0;
        for (int s = 0; s < allX.size(); s++) {
            float[][][] X = allX.get(s);
            float[] y = allY.get(s); // Cambiado

            for (int i = 0; i < X.length; i++) {
                Xcombined[currentIndex] = X[i];
                ycombined[currentIndex] = y[i]; // Cambiado
                currentIndex++;
            }
        }

        // Mezclar los datos para evitar sesgo por símbolo
        EngineUtils.shuffleData(Xcombined, ycombined); // Cambiado

        return new Pair<>(Xcombined, ycombined); // Cambiado
    }

    /**
     * Añade características del símbolo a los datos
     */
    public static float[][][] addSymbolFeature(float[][][] X, String symbol, List<String> allSymbols) {
        float[][][] XwithSymbol = new float[X.length][X[0].length][X[0][0].length + 2];

        // Codificación one-hot simplificada del símbolo
        int symbolIndex = allSymbols.indexOf(symbol);
        float symbolOneHot = symbolIndex / (float) allSymbols.size();
        float symbolNorm = (float) Math.log(symbolIndex + 1) / (float) Math.log(allSymbols.size() + 1);

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                // Copiar características originales
                for (int k = 0; k < X[0][0].length; k++) {
                    XwithSymbol[i][j][k] = X[i][j][k];
                }
                // Añadir características del símbolo
                XwithSymbol[i][j][X[0][0].length] = symbolOneHot;
                XwithSymbol[i][j][X[0][0].length + 1] = symbolNorm;
            }
        }
        return XwithSymbol;
    }

    /**
     * Evaluar modelo en conjunto de test
     */
    public static void evaluateModel(Trainer trainer, NDArray X_test, NDArray y_test,
                                      MultiSymbolNormalizer normalizer) {
        Vesta.info("  Test samples: " + X_test.getShape().get(0));

        // Hacer predicciones
        try {
            // Obtener el bloque del modelo
            var block = trainer.getModel().getBlock();
            var manager = trainer.getManager();

            // Procesar en lotes para evitar problemas de memoria
            int batchSize = 32;
            int totalSamples = (int) X_test.getShape().get(0);

            List<NDArray> batchPredictions = new ArrayList<>();

            for (int i = 0; i < totalSamples; i += batchSize) {
                int end = Math.min(i + batchSize, totalSamples);
                NDArray batchX = X_test.get(new NDIndex(i + ":" + end));

                // Crear ParameterStore para el forward pass
                // En modo evaluación, training = false
                var parameterStore = new ai.djl.training.ParameterStore(manager, false);

                // Forward pass para obtener predicciones
                NDList output = block.forward(parameterStore, new NDList(batchX), false);
                NDArray batchPred = output.singletonOrThrow();

                batchPredictions.add(batchPred);
            }

            // Concatenar todas las predicciones
            NDArray allPredictions = batchPredictions.get(0);
            for (int i = 1; i < batchPredictions.size(); i++) {
                allPredictions = allPredictions.concat(batchPredictions.get(i), 0);
            }

            // Calcular error directamente (MAE)
            NDArray error = allPredictions.sub(y_test).abs();
            float maeNormalized = error.mean().getFloat();

            // Desnormalizar predicciones y valores reales
            float[] predArray = allPredictions.toFloatArray();
            float[] trueArray = y_test.toFloatArray();

            float[] predPrices = normalizer.inverseTransform(predArray);
            float[] truePrices = normalizer.inverseTransform(trueArray);

            // Calcular MAE en precios reales
            double maeReal = 0.0;
            double mape = 0.0;
            int count = 0;

            for (int i = 0; i < predPrices.length; i++) {
                if (Float.isFinite(predPrices[i]) && Float.isFinite(truePrices[i]) && truePrices[i] != 0) {
                    double errorReal = Math.abs(predPrices[i] - truePrices[i]);
                    maeReal += errorReal;
                    mape += (errorReal / Math.abs(truePrices[i])) * 100;
                    count++;
                }
            }

            if (count > 0) {
                maeReal /= count;
                mape /= count;

                Vesta.info("\n📊 Resultados de evaluación:");
                Vesta.info("  MAE (normalizado): " + String.format("%.6f", maeNormalized));
                Vesta.info("  MAE (precio real): $" + String.format("%.4f", maeReal));
                Vesta.info("  MAPE (error porcentual): " + String.format("%.2f", mape) + "%");
                Vesta.info("  Predicciones válidas: " + count + "/" + predPrices.length);

                // Mostrar algunos ejemplos
                Vesta.info("\n🔍 Ejemplos de predicción:");
                int examples = Math.min(25, predPrices.length);
                for (int i = 0; i < examples; i++) {
                    Vesta.info(String.format("  Pred: $%.4f | Real: $%.4f | Error: $%.4f (%.2f%%)",
                            predPrices[i], truePrices[i],
                            Math.abs(predPrices[i] - truePrices[i]),
                            (Math.abs(predPrices[i] - truePrices[i]) / truePrices[i]) * 100));
                }
            }

            // También podemos usar el evaluador configurado para obtener métricas
            // usando el método updateAccumulator si queremos métricas por lote
            try {
                var evaluator = trainer.getEvaluators().get(0);
                evaluator.addAccumulator("test");
                evaluator.resetAccumulator("test");

                // Evaluar por lotes
                for (int i = 0; i < totalSamples; i += batchSize) {
                    int end = Math.min(i + batchSize, totalSamples);
                    NDArray batchX = X_test.get(new NDIndex(i + ":" + end));
                    NDArray batchY = y_test.get(new NDIndex(i + ":" + end));

                    // Obtener predicción para este lote
                    var parameterStore = new ai.djl.training.ParameterStore(manager, false);
                    NDList output = block.forward(parameterStore, new NDList(batchX), false);

                    evaluator.updateAccumulator("test", new NDList(batchY), output);
                }

                float testMae = evaluator.getAccumulator("test");
                Vesta.info("  MAE (usando evaluador): " + testMae);

            } catch (Exception e) {
                Vesta.waring("No se pudo calcular métricas usando evaluador: " + e.getMessage());
            }

        } catch (Exception e) {
            Vesta.error("Error durante la evaluación: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
