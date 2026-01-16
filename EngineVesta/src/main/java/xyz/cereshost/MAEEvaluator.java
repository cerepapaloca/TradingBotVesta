package xyz.cereshost;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;

import java.util.concurrent.ConcurrentHashMap;
import java.util.Map;

public class MAEEvaluator extends Evaluator {

    // Guardamos: key -> (totalElements, totalAbsError)
    private final ConcurrentHashMap<String, Pair<Long, Double>> accum = new ConcurrentHashMap<>();

    public MAEEvaluator() {
        this("mae");
    }

    public MAEEvaluator(String name) {
        super(name);
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray label = labels.singletonOrThrow();
        NDArray pred = predictions.singletonOrThrow();

        // abs error por elemento y media sobre todos los elementos (batch * outputs)
        NDArray absError = label.sub(pred).abs();
        // escalar NDArray
        return absError.mean();
    }

    @Override
    public void addAccumulator(String key) {
        accum.put(key, new Pair<>(0L, 0.0));
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        NDArray label = labels.singletonOrThrow();
        NDArray pred = predictions.singletonOrThrow();

        // media de abs error en este batch (escalar)
        NDArray meanAbs = evaluate(labels, predictions);

        // número de elementos en este batch (por ejemplo batch * outputs)
        long numElements = label.size(); // NDArray.size() devuelve el número total de elementos

        double meanVal = meanAbs.toFloatArray()[0]; // o meanAbs.getDouble() si lo soporta

        // suma absoluta total en este batch = mean * numElements
        double sumAbsForBatch = meanVal * (double) numElements;

        accum.compute(key, (k, old) -> {
            if (old == null) {
                return new Pair<>(numElements, sumAbsForBatch);
            } else {
                long newCount = old.getKey() + numElements;
                double newSum = old.getValue() + sumAbsForBatch;
                return new Pair<>(newCount, newSum);
            }
        });
    }

    @Override
    public void resetAccumulator(String key) {
        accum.put(key, new Pair<>(0L, 0.0));
    }

    @Override
    public float getAccumulator(String key) {
        Pair<Long, Double> p = accum.get(key);
        if (p == null || p.getKey() == 0L) {
            return Float.NaN;
        }
        // media absoluta por elemento
        double average = p.getValue() / (double) p.getKey();
        return (float) average;
    }
}
