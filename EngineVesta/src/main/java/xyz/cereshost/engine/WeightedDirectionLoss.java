package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.training.loss.Loss;

public class WeightedDirectionLoss extends Loss {
    private final float directionWeight;
    private float lastLossTP;
    private float lastLossSL;
    private float lastLossDir;
    private float lastLossTotal;

    public WeightedDirectionLoss(String name, float directionWeight) {
        super(name);
        this.directionWeight = directionWeight;
    }

    @Override
    public NDArray evaluate(NDList target, NDList prediction) {
        NDArray yTrue = target.singletonOrThrow();
        NDArray yPred = prediction.singletonOrThrow();

        // Separar las 3 salidas
        NDArray tpTrue = yTrue.get(":, 0:1");
        NDArray slTrue = yTrue.get(":, 1:2");
        NDArray dirTrue = yTrue.get(":, 2:3");

        NDArray tpPred = yPred.get(":, 0:1");
        NDArray slPred = yPred.get(":, 1:2");
        NDArray dirPred = yPred.get(":, 2:3");

        float margin = (float) PredictionEngine.THRESHOLD;

        NDArray absTrue = dirTrue.abs();
        NDArray absPred = dirPred.abs();

        // 1. Identificar estados
        NDArray isTrueNeutral = absTrue.lt(margin);
        NDArray isPredNeutral = absPred.lt(margin);

        // 2. Definir penalizaciones agresivas
        NDArray signError = dirTrue.mul(dirPred).lt(0);
        NDArray falseAlarm = isTrueNeutral.mul(absPred.gt(margin));
        NDArray blindness = absTrue.gt(margin).mul(isPredNeutral);

        // 3. Cálculo del Peso (W)
        NDArray penalty = signError.toType(DataType.FLOAT32, false).mul(2)
                .add(falseAlarm.toType(DataType.FLOAT32, false).mul(1))
                .add(blindness.toType(DataType.FLOAT32, false).mul(1.2));

        NDArray weights = penalty.add(1.0f);

        // 4. Loss Final: MSE pesado
        NDArray lossDir = dirTrue.sub(dirPred).pow(2).mul(weights).mean().mul(directionWeight);

        // Añadir el Loss de TP/SL (MSE normal)
        NDArray lossTpSl = yTrue.get(":, 0:2").sub(yPred.get(":, 0:2")).pow(2).mean();

        NDArray totalLoss = lossTpSl.add(lossDir);

        this.lastLossTP = yTrue.get(":, 0:1").sub(yPred.get(":, 0:1")).pow(2).mean().getFloat();
        this.lastLossSL = yTrue.get(":, 1:2").sub(yPred.get(":, 1:2")).pow(2).mean().getFloat();
        this.lastLossDir = lossDir.getFloat();
        this.lastLossTotal = totalLoss.getFloat();

        return totalLoss;
    }

    // Métodos para obtener losses como float
    public float getLossTP() {
        return lastLossTP;
    }

    public float getLossSL() {
        return lastLossSL;
    }

    public float getLossDir() {
        return lastLossDir;
    }

    public float getLossTotal() {
        return lastLossTotal;
    }
}
