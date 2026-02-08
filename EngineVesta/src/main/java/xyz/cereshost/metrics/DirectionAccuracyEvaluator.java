package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import xyz.cereshost.utils.EngineUtils;

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
public class DirectionAccuracyEvaluator extends BaseAccuracy {

    public DirectionAccuracyEvaluator() {
        super("3_dir");
    }

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
            NDArray pred = predictions.singletonOrThrow();
            NDArray truth = labels.singletonOrThrow();

            NDArray predClass = pred.argMax(1);
            NDArray trueClass = truth.argMax(1);

            // máscara de válidos (ejemplo: ignorar clase neutral = 2)
            NDArray validMask = trueClass.neq(EngineUtils.floatToNDArray(2f, correctArray.getManager()));

            // aciertos válidos
            NDArray hits = predClass.eq(trueClass).mul(validMask);

            // Sumar todos los aciertos (1 = acierto, 0 = error)
            //float batchCorrect = correctArray.sum().getFloat();
            NDArray batchCorrect = hits.sum()
                    .toType(DataType.FLOAT32, false);

            NDArray batchTotal = validMask.sum()
                    .toType(DataType.FLOAT32, false);

            computeResult(key, batchCorrect, batchTotal);

        }
    }
}
