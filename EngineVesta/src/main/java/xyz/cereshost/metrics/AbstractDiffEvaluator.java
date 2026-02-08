package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.evaluator.Evaluator;
import org.jetbrains.annotations.NotNull;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public abstract class AbstractDiffEvaluator extends Evaluator {

    private final int columnIndex;
    private final ConcurrentHashMap<String, NDArray> sumAccumulator = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, AtomicLong> countAccumulator = new ConcurrentHashMap<>();

    protected AbstractDiffEvaluator(String name, int columnIndex) {
        super(name);
        this.columnIndex = columnIndex;
    }

    @Override
    public NDArray evaluate(@NotNull NDList labels, @NotNull NDList predictions) {
        NDArray label = labels.singletonOrThrow();
        NDArray pred = predictions.singletonOrThrow();

        try (NDArray labelCol = label.get(":," + columnIndex);
             NDArray predCol = pred.get(":," + columnIndex)) {
            return labelCol.sub(predCol).abs().sum();
        }
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        NDArray batchSum = evaluate(labels, predictions);

        sumAccumulator.compute(key, (k, currentSum) -> {
            if (currentSum == null) {
                batchSum.detach();
                return batchSum;
            } else {
                NDArray newSum = currentSum.add(batchSum);
                newSum.detach();
                currentSum.close();
                return newSum;
            }
        });

        long batchCount = labels.singletonOrThrow().getShape().get(0);
        countAccumulator.computeIfAbsent(key, k -> new AtomicLong(0))
                .addAndGet(batchCount);
    }

    @Override
    public float getAccumulator(String key) {
        NDArray sumArray = sumAccumulator.get(key);
        AtomicLong count = countAccumulator.get(key);

        if (sumArray == null || count == null || count.get() == 0) {
            return Float.NaN;
        }

        float totalError = sumArray.toType(DataType.FLOAT32, false).getFloat();
        return totalError / count.get();
    }

    @Override
    public void resetAccumulator(String key) {
        NDArray array = sumAccumulator.remove(key);
        if (array != null) {
            array.close();
        }
        countAccumulator.remove(key);
    }

    @Override
    public void addAccumulator(String key) {
        // No-op, handled by compute/computeIfAbsent
    }
}
