package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.evaluator.Evaluator;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class MAEEvaluator extends Evaluator {

    // Mapa para mantener la SUMA TOTAL de errores en la GPU (NDArray)
    private final ConcurrentHashMap<String, NDArray> sumAccumulator = new ConcurrentHashMap<>();

    // Mapa para mantener el CONTEO TOTAL de elementos (CPU, AtomicLong es seguro y rápido)
    private final ConcurrentHashMap<String, AtomicLong> countAccumulator = new ConcurrentHashMap<>();

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

        // Calculamos la suma absoluta del error en el batch.
        // Retorna un Escalar NDArray.
        return label.sub(pred).abs().sum();
    }

    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        // 1. Obtener la suma de errores del batch actual
        NDArray batchSum = evaluate(labels, predictions);

        // 2. Acumular en el mapa asegurando la persistencia y gestión de memoria
        sumAccumulator.compute(key, (k, currentSum) -> {
            if (currentSum == null) {
                // Primer batch: detach() es vital para sacarlo del manager del batch
                // y que no se borre al terminar la iteración.
                batchSum.detach();
                return batchSum;
            } else {
                // Batches siguientes: Sumamos al acumulador existente
                // currentSum + batchSum = newSum
                NDArray newSum = currentSum.add(batchSum);

                // Hacemos detach del nuevo resultado para que sobreviva
                newSum.detach();

                // IMPORTANTE: Cerrar la referencia antigua para liberar VRAM
                currentSum.close();

                return newSum;
            }
        });

        // 3. Actualizar el conteo de elementos
        // labels.size() devuelve el número total de celdas (BatchSize * Columnas)
        long batchCount = labels.singletonOrThrow().size();

        countAccumulator.computeIfAbsent(key, k -> new AtomicLong(0))
                .addAndGet(batchCount);

        // Nota: 'batchSum' se cerrará automáticamente al final del scope del Trainer
        // porque no le hicimos detach explícito (solo a la copia dentro del compute).
    }

    @Override
    public float getAccumulator(String key) {
        NDArray sumArray = sumAccumulator.get(key);
        AtomicLong count = countAccumulator.get(key);

        if (sumArray == null || count == null || count.get() == 0) {
            return Float.NaN;
        }

        // Transferencia GPU -> CPU
        // Solo ocurre cuando se pide el log (ej. al final del epoch)
        float totalError = sumArray.toType(DataType.FLOAT32, false).getFloat();

        // MAE = Suma Error / Total Elementos
        return totalError / count.get();
    }

    @Override
    public void resetAccumulator(String key) {
        // Limpieza profunda de memoria GPU
        NDArray array = sumAccumulator.remove(key);
        if (array != null) {
            array.close();
        }
        countAccumulator.remove(key);
    }

    @Override
    public void addAccumulator(String key) {
        // No necesario con ConcurrentHashMap compute/computeIfAbsent
    }
}