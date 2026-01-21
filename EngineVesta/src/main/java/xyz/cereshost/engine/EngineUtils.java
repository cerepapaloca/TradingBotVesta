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
import xyz.cereshost.ChartUtils;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Trade;

import java.util.*;

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
    public void shuffleData(float[][] @NotNull [] X, float[] y) { // Cambiado
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

        return new Pair<>(Xcombined, ycombined); // Cambiado
    }

    public static ResultsEvaluate evaluateModel(Trainer trainer, NDArray X_test, NDArray y_test, MultiSymbolNormalizer yNormalizer) {
        // 1. Obtener predicciones (y_test ya está en el dispositivo correcto)
        NDList predictions = trainer.evaluate(new NDList(X_test));
        NDArray yPred = predictions.singletonOrThrow();

        // 2. Traer a CPU para cálculo de métricas y des-normalización
        float[] yRealNorm = y_test.toFloatArray();
        float[] yPredNorm = yPred.toFloatArray();

        // 3. Des-normalizar los log-returns
        float[] yRealRaw = yNormalizer.inverseTransform(yRealNorm);
        float[] yPredRaw = yNormalizer.inverseTransform(yPredNorm);

        int hits = 0;
        int total = yRealRaw.length;
        double totalMae = 0;
        int TotalMargenError = 0;

        Vesta.info("=== Evaluación de Predicción (Muestra de 10) ===");
        List<ResultPrediccion> results = new ArrayList<>();
        for (int i = 0; i < total; i++) {
            float realReturn = yRealRaw[i];
            float predReturn = yPredRaw[i];

            // Métrica: Directional Accuracy (Hit Rate)
            // Verificamos si ambos tienen el mismo signo (ambos suben o ambos bajan)
            if (Math.signum(realReturn) == Math.signum(predReturn)) {
                hits++;
            }

            float diff = Math.abs(realReturn - predReturn);
            totalMae += diff;
            if (diff < 0.000_05) {
                TotalMargenError++;
            }

            // Mostrar solo los primeros 10 para no saturar la consola
            if (i < 10) {
                String dirReal = realReturn >= 0 ? "UP" : "DOWN";
                String dirPred = predReturn >= 0 ? "UP" : "DOWN";
                boolean success = dirReal.equals(dirPred);

                Vesta.info("[%d] Real: %.6f (%s) | Pred: %.6f (%s) | DiffAbs: %.6f | ¿Acierto Dir?: %b",
                        i, realReturn, dirReal, predReturn, dirPred, diff, success);
            }
            results.add(new ResultPrediccion(predReturn, realReturn, 1));

        }
        results.sort(Comparator.comparingDouble(s -> s.pred() - s.real()));

        double hitRate = (double) hits / total * 100;
        double avgMae = totalMae / total;
        double margenRate = (double) TotalMargenError / total *100;


        return new EngineUtils.ResultsEvaluate(
                "VestaLSTM",
                avgMae,
                hitRate,
                margenRate,
                results
        );
    }

    public record ResultsEvaluate(
            String modelName,
            double avgMae,
            double hitRate,
            double margenRate,
            List<ResultPrediccion> resultPrediccions
    ) {}

    public record ResultPrediccion(float pred, float real, long timestamp) {}

    public static Pair<float[][][], float[]> clearData(float[][][] X, float[] y) {
        final float EPS = 1e-9f;

        Map<String, Integer> seen = new HashMap<>();
        List<float[][]> uniqueX = new ArrayList<>();
        List<Float> uniqueY = new ArrayList<>();

        int removedDuplicates = 0;
        int removedZeroClose = 0;
        int removedBadValue = 0;

        for (int i = 0; i < X.length; i++) {
            float yi = y[i];

            // 1) filtrar NaN / Inf / valores inválidos
            if (Float.isNaN(yi) || Float.isInfinite(yi)) {
                removedBadValue++;
                Vesta.waring("Eliminando muestra con y inválido en índice " + i + " (y=" + yi + ")");
                continue;
            }

            // 2) filtrar cierres cero (o muy cercanos a 0)
            if (Math.abs(yi) < EPS) {
                removedZeroClose++;
                Vesta.waring("Eliminando muestra con cierre ~0 en índice " + i + " (y=" + yi + ")");
                continue;
            }

            // 3) detectar duplicados por hash de la ventana X[i]
            String hash = Arrays.deepToString(X[i]);
            if (!seen.containsKey(hash)) {
                seen.put(hash, i);
                uniqueX.add(X[i]);
                uniqueY.add(yi);
            } else {
                removedDuplicates++;
                Vesta.waring("Eliminando duplicado en índice " + i +
                        " (igual a índice " + seen.get(hash) + ")");
            }
        }

        // Convertir de vuelta a arrays
        float[][][] Xunique = new float[uniqueX.size()][][];
        float[] yunique = new float[uniqueY.size()];

        for (int i = 0; i < Xunique.length; i++) {
            Xunique[i] = uniqueX.get(i);
            yunique[i] = uniqueY.get(i);
        }

        Vesta.info("Eliminados " + (removedDuplicates) + " duplicados");
        Vesta.info("Eliminadas " + (removedZeroClose) + " muestras con cierre = 0 (o ~0)");
        Vesta.info("Eliminadas " + (removedBadValue) + " muestras con valores inválidos (NaN/Inf)");
        Vesta.info("Total resultante: " + Xunique.length + " muestras (de " + X.length + " originales)");

        return new Pair<>(Xunique, yunique);
    }


}
