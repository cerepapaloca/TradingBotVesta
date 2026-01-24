package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
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

        // Separar TP/SL (indices 0, 1) y Dirección (indice 2)
        NDArray tpSlTrue = yTrue.get(":, 0:2");
        NDArray tpSlPred = yPred.get(":, 0:2");
        NDArray dirTrue = yTrue.get(":, 2:3");
        NDArray dirPred = yPred.get(":, 2:3");

        // Error cuadrático para TP y SL
        NDArray lossTpSl = tpSlTrue.sub(tpSlPred).pow(2).mean();

        // Error cuadrático para Dirección MULTIPLICADO por el peso de audacia
        // Esto castiga mucho más el estar lejos del 1 o -1
        NDArray lossDir = dirTrue.sub(dirPred).pow(2).mean().mul(directionWeight);

        return lossTpSl.add(lossDir);
    }
}
