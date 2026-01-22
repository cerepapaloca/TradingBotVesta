package xyz.cereshost.engine;

import ai.djl.Device;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.Trainer;
import ai.djl.util.Pair;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.common.Vesta;

import java.util.*;

@UtilityClass
public class EngineUtils {

    public static void checkEngines() {
        Vesta.info("=== Verificando Engines DJL ===");

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
     * Mezcla los datos aleatoriamente (actualizado para float[][])
     */
    public void shuffleData(float[][][] X, float[][] y) {
        Random rand = new Random(42);
        for (int i = X.length - 1; i > 0; i--) {
            int j = rand.nextInt(i + 1);

            // Intercambiar X
            float[][] tempX = X[i];
            X[i] = X[j];
            X[j] = tempX;

            // Intercambiar y (ahora con dos columnas)
            float[] tempY = y[i];
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
     * Combina múltiples datasets en uno solo (actualizado para float[][])
     */
    @Contract("_, _ -> new")
    public static @NotNull Pair<float[][][], float[][]> combineDatasets(@NotNull List<float[][][]> allX, List<float[][]> allY) {
        int totalSamples = 0;
        for (float[][][] X : allX) {
            totalSamples += X.length;
        }

        int lookback = allX.get(0)[0].length;
        int features = allX.get(0)[0][0].length;

        float[][][] Xcombined = new float[totalSamples][lookback][features];
        float[][] ycombined = new float[totalSamples][2]; // Cambiado a 2 columnas

        int currentIndex = 0;
        for (int s = 0; s < allX.size(); s++) {
            float[][][] X = allX.get(s);
            float[][] y = allY.get(s); // Cambiado

            for (int i = 0; i < X.length; i++) {
                Xcombined[currentIndex] = X[i];
                ycombined[currentIndex][0] = y[i][0]; // TP
                ycombined[currentIndex][1] = y[i][1]; // SL
                currentIndex++;
            }
        }

        return new Pair<>(Xcombined, ycombined);
    }

    /**
     * Evalúa el modelo con dos salidas (TP y SL)
     */
    public static ResultsEvaluate evaluateModel(Trainer trainer, NDArray X_test, NDArray y_test, MultiSymbolNormalizer yNormalizer) {
        // 1. Obtener predicciones
        NDList predictions = trainer.evaluate(new NDList(X_test));
        NDArray yPred = predictions.singletonOrThrow();

        // 2. Convertir a arrays 2D
        long[] shape = y_test.getShape().getShape();
        int batchSize = (int) shape[0];
        int numOutputs = (int) shape[1];

        // Convertir NDArrays a float[][] usando la forma correcta
        float[] yTestFlat = y_test.toFloatArray();
        float[] yPredFlat = yPred.toFloatArray();

        float[][] yTest2D = new float[batchSize][numOutputs];
        float[][] yPred2D = new float[batchSize][numOutputs];

        for (int i = 0; i < batchSize; i++) {
            for (int j = 0; j < numOutputs; j++) {
                yTest2D[i][j] = yTestFlat[i * numOutputs + j];
                yPred2D[i][j] = yPredFlat[i * numOutputs + j];
            }
        }

        // 3. Des-normalizar
        float[][] yTestRaw = yNormalizer.inverseTransform(yTest2D);
        float[][] yPredRaw = yNormalizer.inverseTransform(yPred2D);

        // 4. Métricas para TP y SL por separado
        double totalMaeTP = 0;
        double totalMaeSL = 0;
        int hitsDirection = 0;
        int totalTrades = batchSize;

        List<ResultPrediccion> results = new ArrayList<>();

        for (int i = 0; i < batchSize; i++) {
            float realTP = yTestRaw[i][0];
            float realSL = yTestRaw[i][1];
            float predTP = yPredRaw[i][0];
            float predSL = yPredRaw[i][1];

            // Métricas de error
            totalMaeTP += Math.abs(realTP - predTP);
            totalMaeSL += Math.abs(realSL - predSL);

            // Acierto direccional (basado en el ratio TP/SL)
            // Si el ratio predicho sugiere un trade rentable (TP > SL) y el real también lo es
            boolean predProfitable = predTP > predSL;
            boolean realProfitable = realTP > realSL;

            if (predProfitable == realProfitable) {
                hitsDirection++;
            }

            // Guardar resultados para backtest
            results.add(new ResultPrediccion(predTP, predSL, realTP, realSL, i));
        }

        double avgMaeTP = totalMaeTP / batchSize;
        double avgMaeSL = totalMaeSL / batchSize;
        double hitRate = (double) hitsDirection / batchSize * 100.0;

        // 5. Backtest
        SimulationResult simResult = simulateBacktest(results);

        Vesta.info("=== Evaluación del Modelo ===");
        Vesta.info("MAE TP: %.6f", avgMaeTP);
        Vesta.info("MAE SL: %.6f", avgMaeSL);
        Vesta.info("Hit Rate Direccional: %.2f%%", hitRate);
        Vesta.info("ROI Backtest: %.2f%%", simResult.roiPercent());

        return new ResultsEvaluate(
                "VestaLSTM",
                avgMaeTP,
                avgMaeSL,
                hitRate,
                simResult,
                results
        );
    }

    public record ResultsEvaluate(
            String modelName,
            double avgMaeTP,
            double avgMaeSL,
            double hitRate,
            SimulationResult simulation,
            List<ResultPrediccion> resultPrediccions
    ) {}

    public record ResultPrediccion(float predTP, float predSL, float realTP, float realSL, long timestamp) {

        public float lsDiff() {
            return predSL - realSL;
        }

        public float tpDiff() {
            return predTP - realTP;
        }
    }

    /**
     * Limpia datos (actualizado para float[][])
     */
    public static Pair<float[][][], float[][]> clearData(float[][][] X, float[][] y) {
        final float EPS = 1e-9f;

        Map<String, Integer> seen = new HashMap<>();
        List<float[][]> uniqueX = new ArrayList<>();
        List<float[]> uniqueY = new ArrayList<>(); // Cambiado a float[]

        int removedDuplicates = 0;
        int removedBadValue = 0;

        for (int i = 0; i < X.length; i++) {
            float tp = y[i][0];
            float sl = y[i][1];

            // 1) filtrar NaN / Inf / valores inválidos
            if (Float.isNaN(tp) || Float.isInfinite(tp) ||
                    Float.isNaN(sl) || Float.isInfinite(sl)) {
                removedBadValue++;
                Vesta.waring("Eliminando muestra con TP/SL inválido en índice " + i);
                continue;
            }

            // 2) filtrar valores negativos o cero
            if (tp <= 0 || sl <= 0) {
                removedBadValue++;
                Vesta.waring("Eliminando muestra con TP/SL <= 0 en índice " + i);
                continue;
            }

            // 3) detectar duplicados
            String hash = Arrays.deepToString(X[i]);
            if (!seen.containsKey(hash)) {
                seen.put(hash, i);
                uniqueX.add(X[i]);
                uniqueY.add(new float[]{tp, sl});
            } else {
                removedDuplicates++;
                Vesta.waring("Eliminando duplicado en índice " + i);
            }
        }

        // Convertir de vuelta a arrays
        float[][][] Xunique = new float[uniqueX.size()][][];
        float[][] yunique = new float[uniqueY.size()][2];

        for (int i = 0; i < Xunique.length; i++) {
            Xunique[i] = uniqueX.get(i);
            yunique[i][0] = uniqueY.get(i)[0];
            yunique[i][1] = uniqueY.get(i)[1];
        }

        Vesta.info("Datos limpiados: " + Xunique.length + " muestras válidas");
        Vesta.info("Eliminados " + removedDuplicates + " duplicados");
        Vesta.info("Eliminadas " + removedBadValue + " muestras con valores inválidos");

        return new Pair<>(Xunique, yunique);
    }

    public record SimulationResult(
            double initialBalance,
            double finalBalance,
            double netPnL,
            double roiPercent,
            int totalTrades,
            int winTrades,
            int lossTrades,
            double maxDrawdown
    ) {}

    /**
     * Simula operaciones de trading usando TP y SL predichos
     * Ratio flexible basado en las predicciones
     */
    private static SimulationResult simulateBacktest(List<ResultPrediccion> predictions) {
        double balance = 1000.0;
        double initialBalance = balance;
        double fee = 0.0004; // 0.04% comisión

        int wins = 0;
        int losses = 0;
        int totalTrades = 0;
        double maxBalance = balance;
        double maxDrawdown = 0.0;

        // Umbral mínimo para operar
        double minThreshold = 0.0005;

        for (ResultPrediccion p : predictions) {
            float predTP = p.predTP(); // Porcentaje de TP predicho
            float predSL = p.predSL(); // Porcentaje de SL predicho
            float realTP = p.realTP(); // TP real
            float realSL = p.realSL(); // SL real

            // Si las predicciones son muy pequeñas, no operar
            if (predTP < minThreshold || predSL < minThreshold) {
                continue;
            }

            // Determinar dirección basada en el ratio TP/SL predicho
            // Si TP predicho > SL predicho, LONG (esperamos ganar más de lo que podemos perder)
            // Si SL predicho > TP predicho, SHORT (la pérdida potencial es mayor que la ganancia)
            boolean isLong = predTP > predSL;

            double pnlPercent = 0;

            if (isLong) {
                // LÓGICA LONG
                if (realTP >= predTP) {
                    // TP alcanzado
                    pnlPercent = predTP;
                } else if (realSL >= predSL) {
                    // SL alcanzado (nota: en long, SL es negativo)
                    pnlPercent = -predSL;
                } else {
                    // Cierre de mercado - usar el movimiento real (positivo o negativo)
                    pnlPercent = Math.min(realTP, -realSL);
                }
            } else {
                // LÓGICA SHORT
                if (realSL >= predSL) {
                    // TP alcanzado (en short, SL real es la ganancia)
                    pnlPercent = predSL;
                } else if (realTP >= predTP) {
                    // SL alcanzado (en short, TP real es la pérdida)
                    pnlPercent = -predTP;
                } else {
                    // Cierre de mercado
                    pnlPercent = Math.min(realSL, -realTP);
                }
            }

            // Aplicar PnL
            double tradeResult = balance * (pnlPercent - (fee * 2));

            // Actualizar contadores
            if (pnlPercent > 0) {
                wins++;
            } else if (pnlPercent < 0) {
                losses++;
            }

            totalTrades++;
            balance += tradeResult;

            // Calcular Drawdown
            if (balance > maxBalance) {
                maxBalance = balance;
            }
            double currentDD = (maxBalance - balance) / maxBalance;
            if (currentDD > maxDrawdown) {
                maxDrawdown = currentDD;
            }
        }

        // Estadísticas finales
        double netPnL = balance - initialBalance;
        double roiPercent = (netPnL / initialBalance) * 100.0;

        Vesta.info("=== Resultados Backtest ===");
        Vesta.info("Trades totales: %d", totalTrades);
        Vesta.info("Trades ganadores: %d (%.1f%%)", wins,
                totalTrades > 0 ? ((double) wins / totalTrades) * 100.0 : 0);
        Vesta.info("Balance inicial: $%.2f", initialBalance);
        Vesta.info("Balance final: $%.2f", balance);
        Vesta.info("PnL neto: $%.2f (%.2f%%)", netPnL, roiPercent);
        Vesta.info("Drawdown máximo: %.2f%%", maxDrawdown * 100.0);

        return new SimulationResult(
                initialBalance,
                balance,
                netPnL,
                roiPercent,
                totalTrades,
                wins,
                losses,
                maxDrawdown * 100.0
        );
    }

    /**
     * Analiza el rendimiento por tamaño de TP/SL
     */
    public static void analyzePerformanceBySize(List<EngineUtils.ResultPrediccion> predictions) {
        if (predictions == null || predictions.isEmpty()) return;

        // Crear buckets por tamaño de TP
        List<BackTestEngine.PerformanceBucket> buckets = new ArrayList<>();
        buckets.add(new BackTestEngine.PerformanceBucket(0.0, 0.001, "0.0-0.1%"));   // Muy pequeño
        buckets.add(new BackTestEngine.PerformanceBucket(0.001, 0.005, "0.1-0.5%")); // Pequeño
        buckets.add(new BackTestEngine.PerformanceBucket(0.005, 0.01, "0.5-1.0%"));  // Mediano
        buckets.add(new BackTestEngine.PerformanceBucket(0.01, 0.02, "1.0-2.0%"));   // Grande
        buckets.add(new BackTestEngine.PerformanceBucket(0.02, Double.MAX_VALUE, ">2.0%")); // Muy grande

        for (BackTestEngine.PerformanceBucket bucket : buckets) {
            bucket.analyze(predictions);
            bucket.printResults();
        }
    }
}