package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.evaluator.Evaluator;
import lombok.SneakyThrows;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Evaluador Binario (Long vs Short).
 * Ignora completamente los casos Neutrales.
 *
 * Lógica:
 * 1. Si la IA predice Neutral -> Se descarta (No cuenta para el promedio).
 * 2. Si la etiqueta real es Neutral -> Se descarta.
 * 3. Solo se evalúa si IA dice L/S Y la realidad es L/S.
 *
 * Output esperado: [TP, SL, Long, Neutral, Short]
 */
public class BinaryDirectionEvaluator extends Evaluator {

    // Índices relativos al sub-array de dirección [L, N, S]
    // Long=0, Neutral=1, Short=2
    private static final int CLASS_NEUTRAL = 1;

    private final AtomicLong correctCount = new AtomicLong(0);
    private final AtomicLong totalValidCount = new AtomicLong(0);

    public BinaryDirectionEvaluator() {
        super("BinDirAcc"); // Nombre corto para los logs
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        // Obtenemos los arrays (Batch, 5)
        NDArray predAll = predictions.singletonOrThrow();
        NDArray labelAll = labels.singletonOrThrow();

        // 1. Extraer sub-sección de Dirección (Columnas 2, 3, 4)
        NDArray predDir = predAll.get(":, 2:5");
        NDArray labelDir = labelAll.get(":, 2:5");

        // 2. Convertir a Clases (0=Long, 1=Neutral, 2=Short)
        NDArray predClass = predDir.argMax(1);
        NDArray trueClass = labelDir.argMax(1);

        // 3. Crear MÁSCARA DE VALIDEZ (Filtro)
        // Queremos filas donde: (Pred != Neutral) AND (Real != Neutral)
        NDArray notNeutralPred = predClass.neq(CLASS_NEUTRAL);
        NDArray notNeutralLabel = trueClass.neq(CLASS_NEUTRAL);

        // Mask es 1 si ambos son válidos, 0 si alguno es neutral
        NDArray validMask = notNeutralPred.logicalAnd(notNeutralLabel);

        // 4. Comparar Aciertos
        NDArray isMatch = predClass.eq(trueClass);

        // 5. Aplicar Filtro: Un acierto solo cuenta si pasó la máscara
        // El resultado será 1 (acierto válido), 0 (fallo o ignorado)
        NDArray validMatches = isMatch.logicalAnd(validMask);

        // Retornamos array de 1s y 0s. Nota: Esto incluye 0s para los ignorados,
        // por lo que 'mean()' directo no sirve aquí, se necesita updateAccumulator.
        return validMatches.toType(DataType.FLOAT32, false);
    }

    @SneakyThrows
    @Override
    public void updateAccumulator(String key, NDList labels, NDList predictions) {
        // Recalculamos la lógica de filtrado aquí para tener el conteo exacto
        NDArray predAll = predictions.singletonOrThrow();
        NDArray labelAll = labels.singletonOrThrow();

        NDArray predClass = predAll.get(":, 2:5").argMax(1);
        NDArray trueClass = labelAll.get(":, 2:5").argMax(1);

        // Definir qué es válido
        CompletableFuture<NDArray> validMaskFuture = CompletableFuture.supplyAsync(() -> predClass.neq(CLASS_NEUTRAL)
                .logicalAnd(trueClass.neq(CLASS_NEUTRAL)));

        // Calcular aciertos sobre los válidos
        NDArray validMask = validMaskFuture.get();
        NDArray hits = predClass.eq(trueClass).logicalAnd(validMask);

        // Sumar
        long batchCorrect = hits.sum().getLong();
        long batchTotalValid = validMask.sum().getLong();

        // Solo actualizamos si hubo al menos una muestra válida en el batch
        if (batchTotalValid > 0) {
            correctCount.addAndGet(batchCorrect);
            totalValidCount.addAndGet(batchTotalValid);
        }

        // Limpieza de memoria (importante en DJL loop)
        predClass.close();
        trueClass.close();
        validMask.close();
        hits.close();
    }

    @Override
    public float getAccumulator(String key) {
        long total = totalValidCount.get();
        if (total == 0) {
            return 0f; // Evitar división por cero si todo fue Neutral
        }
        return (float) correctCount.get() / total * 100.0f;
    }

    @Override
    public void resetAccumulator(String key) {
        correctCount.set(0);
        totalValidCount.set(0);
    }

    @Override
    public void addAccumulator(String key) {
    }
}