package xyz.cereshost.metrics;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import lombok.SneakyThrows;
import xyz.cereshost.utils.EngineUtils;

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
public class BinaryDirectionEvaluator extends BaseAccuracy {

    // Índices relativos al sub-array de dirección [L, N, S]
    // Long=0, Neutral=1, Short=2
    private static final int CLASS_NEUTRAL = 1;

    public BinaryDirectionEvaluator() {
        super("2_dir"); // Nombre corto para los logs
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        // Obtenemos los arrays (Batch, 5)
        NDManager manager = labels.getManager();

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
        NDArray notNeutralPred = predClass.neq(EngineUtils.floatToNDArray(CLASS_NEUTRAL, manager));
        NDArray notNeutralLabel = trueClass.neq(EngineUtils.floatToNDArray(CLASS_NEUTRAL, manager));

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
        NDManager manager = labels.getManager();

        // Recalculamos la lógica de filtrado aquí para tener el conteo exacto
        NDArray predAll = predictions.singletonOrThrow();
        NDArray labelAll = labels.singletonOrThrow();

        NDArray predClass = predAll.get(":, 2:5").argMax(1);
        NDArray trueClass = labelAll.get(":, 2:5").argMax(1);

        // Definir qué es válido
        NDArray validMask = predClass.neq(EngineUtils.floatToNDArray(CLASS_NEUTRAL, manager))
                .logicalAnd(trueClass.neq(EngineUtils.floatToNDArray(CLASS_NEUTRAL, manager)));

        // Calcular aciertos sobre los válidos
        NDArray hits = predClass.eq(trueClass).logicalAnd(validMask);

        // Solo actualizamos si hubo al menos una muestra válida en el batch
        computeResult(key, hits.sum(), validMask.sum());

        // Limpieza de memoria (importante en DJL loop)
        predClass.close();
        trueClass.close();
        validMask.close();
        hits.close();
    }
}