package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.evaluator.Evaluator;
import org.jetbrains.annotations.NotNull;

import java.util.concurrent.ConcurrentHashMap;

public abstract class BaseAccuracy extends Evaluator {

    protected final ConcurrentHashMap<String, NDArray> correct = new ConcurrentHashMap<>();
    protected final ConcurrentHashMap<String, NDArray> total = new ConcurrentHashMap<>();

    /**
     * Creates an evaluator with abstract update methods.
     *
     * @param name the name of the evaluator
     */
    public BaseAccuracy(String name) {
        super(name);
    }

    @Override
    public float getAccumulator(String key) {
        double t = total.get(key).toArray()[0].doubleValue();
        if (t == 0d) {
            return 0f; // Evitar división por cero si todo fue Neutral
        }
        return (float) (correct.get(key).toArray()[0].doubleValue() / t * 100.0f);
    }

    @Override
    public void resetAccumulator(String key) {
        for (NDArray array : correct.values()) {
            array.close();
        }
        correct.clear();
        for (NDArray array : total.values()) {
            array.close();
        }
        total.clear();
    }

    @Override
    public void addAccumulator(String key) {
        // No se necesita implementación específica
    }

    protected void computeResult(String key, NDArray batchCorrect, NDArray batchTotal) {
        correct.compute(key, (k, old) -> {
            if (old == null) {
                batchCorrect.detach();
                return batchCorrect;
            } else {
                old.addi(batchCorrect);
                return old;
            }
        });
        total.compute(key, (k, old) -> {
            if (old == null) {
                batchTotal.detach();
                return batchTotal;
            } else {
                old.addi(batchTotal);
                return old;
            }
        });
    }


    protected record Direction(NDArray predAll, NDArray predDir, NDArray labelDir, NDArray predClass, NDArray trueClass) {
    }
}
