package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.evaluator.Evaluator;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Evaluador sencillo para porcentaje de acierto en la dirección.
 * Espera que labels y predictions sean NDList donde el primer NDArray tiene forma [B, 5]
 * y las columnas son: [TP, SL, Long, Neutral, Short].
 *
 * Uso:
 *   DirectionAccuracyEvaluator eval = new DirectionAccuracyEvaluator();
 *   eval.updateAccumulator("direction", labelsNDList, predsNDList);
 *   float acc = eval.getAccumulator("accuracy");
 */
public class DirectionAccuracyEvaluator extends Evaluator {

    public DirectionAccuracyEvaluator() {
        super("dae");
    }

    // Usamos nuestras propias variables para no depender de la versión de DJL
    private final AtomicLong correctCount = new AtomicLong(0);
    private final AtomicLong totalCount = new AtomicLong(0);

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        // 1. PREDICCIONES: Extraer solo la parte de dirección (columnas 2-4)
        NDArray predAll = predictions.singletonOrThrow();
        NDArray predDirection = predAll.get(":, 2:5"); // Solo columnas de dirección
        // No fatal, pero puede ser una advertencia

        // 3. Obtener clase predicha (argmax en dirección)
        NDArray predClass = predDirection.argMax(1); // [B]

        // 4. ETIQUETAS: Extraer one-hot de dirección
        NDArray allLabels = labels.singletonOrThrow();
        NDArray trueDirection = allLabels.get(":, 2:5"); // Solo columnas de dirección
        NDArray trueClass = trueDirection.argMax(1); // [B]

        // 5. Comparar
        NDArray match = predClass.eq(trueClass);

        return match.toType(DataType.FLOAT32, false);
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        try (NDArray correctArray = evaluate(labels, predictions)) {
            // Verificar que el array no esté vacío
            if (correctArray.isEmpty()) {
                return;
            }

            // Sumar todos los aciertos (1 = acierto, 0 = error)
            float batchCorrect = correctArray.sum().getFloat();
            long batchTotal = correctArray.getShape().get(0);

            // Actualizar contadores
            correctCount.addAndGet((long) batchCorrect);
            totalCount.addAndGet(batchTotal);
        }
    }

    @Override
    public float getAccumulator(String key) {
        long total = totalCount.get();
        if (total == 0) {
            return 0f;
        }
        return (float) correctCount.get() / total * 100; // Multiplicar por 100 para porcentaje
    }

    @Override
    public void resetAccumulator(String key) {
        correctCount.set(0);
        totalCount.set(0);
    }

    @Override
    public void addAccumulator(String key) {
        // No se necesita implementación específica
    }
}
