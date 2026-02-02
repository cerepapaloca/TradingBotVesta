package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class MAEEvaluator extends Evaluator {

    // Guardamos: key -> (totalElements, totalAbsError)
    private final ConcurrentHashMap<String, Pair<Long, Double>> accum = new ConcurrentHashMap<>();

    private NDArray totalSumArray; // Acumulador en GPU
    private final AtomicLong totalElements = new AtomicLong(0);

    public MAEEvaluator() {
        this("mae");
    }

    public MAEEvaluator(String name) {
        super(name);
    }

    @Override
    public NDArray evaluate(@NotNull NDList labels, @NotNull NDList predictions) {
        NDArray label = labels.singletonOrThrow();
        NDArray pred = predictions.singletonOrThrow();
        // Calculamos el error absoluto pero NO pedimos el valor a la CPU todavía
        return label.sub(pred).abs().sum();
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        // 1. Calcular la suma del batch en la GPU
        NDArray batchSum = evaluate(labels, predictions);

        // 2. Acumular en el array global (también en GPU)
        if (totalSumArray == null) {
            // detach() es vital para que el array sobreviva al cierre del manager del batch
            batchSum.detach();
            totalSumArray = batchSum;
        } else {
            NDArray oldSum = totalSumArray;
            // Sumamos y desvinculamos del manager actual
            NDArray sum = oldSum.add(batchSum);
            sum.detach();
            totalSumArray = sum;
            oldSum.close(); // Liberamos el puntero anterior en GPU
        }

        // 3. El conteo de elementos es un long simple, no bloquea
        totalElements.addAndGet(labels.singletonOrThrow().size());
    }

    @Override
    public float getAccumulator(String key) {
        if (totalSumArray == null || totalElements.get() == 0) {
            return Float.NaN;
        }

        // SOLO AQUÍ ocurre la sincronización (una vez por cada vez que necesites el log)
        float totalSum = totalSumArray.toType(ai.djl.ndarray.types.DataType.FLOAT32, false)
                .getFloat();

        return totalSum / totalElements.get();
    }

    @Override
    public void resetAccumulator(String key) {
        if (totalSumArray != null) {
            totalSumArray.close();
            totalSumArray = null;
        }
        totalElements.set(0);
    }

    @Override
    public void addAccumulator(String key) {
        // No requerido con esta implementación
    }
}
