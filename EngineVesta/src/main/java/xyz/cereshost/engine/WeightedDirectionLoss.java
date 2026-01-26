package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.loss.Loss;

public class WeightedDirectionLoss extends Loss {
    private final float directionWeight;

    public WeightedDirectionLoss(String name, float directionWeight) {
        super(name);
        this.directionWeight = directionWeight;
    }

    @Override
    public NDArray evaluate(NDList target, NDList prediction) {
        NDArray yTrue = target.singletonOrThrow();
        NDArray yPred = prediction.singletonOrThrow();

        NDArray dirTrue = yTrue.get(":, 2:3");
        NDArray dirPred = yPred.get(":, 2:3");

        float margin = (float) PredictionEngine.THRESHOLD;

        NDArray absTrue = dirTrue.abs();
        NDArray absPred = dirPred.abs();

        // 1. Identificar estados
        NDArray isTrueNeutral = absTrue.lt(margin);
        NDArray isPredNeutral = absPred.lt(margin);

        // 2. Definir penalizaciones agresivas
        // PENALIZACIÓN A: Error de Signo (Long vs Short)
        NDArray signError = dirTrue.mul(dirPred).lt(0);

        // PENALIZACIÓN B: Falsa Alarma (IA dice tendencia, mercado es neutral) -> CRÍTICO
        NDArray falseAlarm = isTrueNeutral.mul(absPred.gt(margin));

        // PENALIZACIÓN C: Ceguera (Mercado tiene tendencia, IA dice neutral)
        NDArray blindness = absTrue.gt(margin).mul(isPredNeutral);

        // 3. Cálculo del Peso (W)
        // Empezamos con peso base 1.0.
        // Si hay falsa alarma o ceguera, subimos a 5.0. Si hay error de signo, subimos a 10.0.
        NDArray penalty = signError.toType(DataType.FLOAT32, false).mul(4.0f) // Antes 10.0
                .add(falseAlarm.toType(DataType.FLOAT32, false).mul(2.0f)) // Antes 5.0
                .add(blindness.toType(DataType.FLOAT32, false).mul(1.5f)); // Antes 5.0

        NDArray weights = penalty.add(1.0f);

        // 4. Loss Final: MSE pesado
        NDArray lossDir = dirTrue.sub(dirPred).pow(2).mul(weights).mean().mul(directionWeight);

        // Añadir el Loss de TP/SL (MSE normal)
        NDArray lossTpSl = yTrue.get(":, 0:2").sub(yPred.get(":, 0:2")).pow(2).mean();

        return lossTpSl.add(lossDir);
    }
}
