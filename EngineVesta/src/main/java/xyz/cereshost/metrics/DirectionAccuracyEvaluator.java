package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.evaluator.Evaluator;

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

    @Override
    public NDArray evaluate(
            NDList labels,
            NDList predictions
    ) {
        // prediction: [B, 5]
        NDArray pred = predictions.singletonOrThrow();
        NDArray label = labels.singletonOrThrow();

        // Extraer solo dirección: columnas 2,3,4
        NDArray predDir = pred.get(":, 2:5");   // [B,3]
        NDArray trueDir = label.get(":, 2:5");  // [B,3]

        // Argmax → clase predicha y real
        NDArray predClass = predDir.argMax(1);  // [B]
        NDArray trueClass = trueDir.argMax(1);  // [B]

        // Comparar
        NDArray correct = predClass.eq(trueClass);

        // Accuracy
        return correct.toType(pred.getDataType(), false).mean();
    }

    @Override
    public void addAccumulator(String key) {

    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {

    }

    @Override
    public void resetAccumulator(String key) {

    }

    @Override
    public float getAccumulator(String key) {
        return 0;
    }
}
