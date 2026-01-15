package xyz.cereshost;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;

import java.util.concurrent.ConcurrentHashMap;

public class MAEEvaluator extends Evaluator {

    private final ConcurrentHashMap<String, Pair<Long, Float>> maeAccumulators;

    public MAEEvaluator() {
        this("mae");
    }

    public MAEEvaluator(String name) {
        super(name);
        this.maeAccumulators = new ConcurrentHashMap<>();
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        // labels y predictions son NDList con un solo NDArray cada uno
        NDArray label = labels.singletonOrThrow();
        NDArray prediction = predictions.singletonOrThrow();

        // MAE = mean(|y_true - y_pred|)
        NDArray absError = label.sub(prediction).abs();
        return absError.mean(); // Devuelve un escalar en un NDArray
    }

    @Override
    public void addAccumulator(String key) {
        maeAccumulators.put(key, new Pair<>(0L, 0f));
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        NDArray maeValue = evaluate(labels, predictions);
        float mae = maeValue.toFloatArray()[0];
        long batchSize = labels.singletonOrThrow().size();

        maeAccumulators.compute(key, (k, oldPair) -> {
            if (oldPair == null) {
                return new Pair<>(batchSize, mae * batchSize);
            }
            long totalSize = oldPair.getKey() + batchSize;
            float totalMae = oldPair.getValue() + (mae * batchSize);
            return new Pair<>(totalSize, totalMae);
        });
    }

    @Override
    public void resetAccumulator(String key) {
        maeAccumulators.put(key, new Pair<>(0L, 0f));
    }

    @Override
    public float getAccumulator(String key) {
        Pair<Long, Float> pair = maeAccumulators.get(key);
        if (pair == null || pair.getKey() == 0L) {
            return Float.NaN;
        }
        return pair.getValue() / pair.getKey();
    }
}