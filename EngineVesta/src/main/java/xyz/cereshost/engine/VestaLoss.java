package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.training.loss.Loss;
import lombok.SneakyThrows;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;

public class VestaLoss extends Loss {

    public record LossReport(
            float total, float tp, float sl, float longL, float neutralL, float shortL
    ) {}

    private volatile CompletableFuture<LossReport> dataRequest = null;

    public VestaLoss(String name) {
        super(name);
    }

    private NDArray lastResult = null;
    private String lastPredictionId = "";

    @SneakyThrows
    @Override
    public NDArray evaluate(NDList target, NDList prediction) {
        NDArray yPred = prediction.singletonOrThrow();
        NDArray yTrue = target.singletonOrThrow();

//        if (yPred.getUid().equals(lastPredictionId) && lastResult != null) {
//            return lastResult;
//        }


        // 1. Cálculos vectorizados rápidos (GPU)
        NDList trueParts = yTrue.split(new long[]{1, 2, 3, 4}, 1);
        NDList predParts = yPred.split(new long[]{1, 2, 3, 4}, 1);
        NDManager manager = target.getManager();
        NDArray lossTP = TPSLAdvance(trueParts, trueParts.get(0), predParts.get(0));
        NDArray lossSL = TPSLAdvance(trueParts, trueParts.get(1), predParts.get(1));
        NDArray lLong = binaryCrossEntropy(trueParts.get(2), predParts.get(2)).mul(EngineUtils.floatToNDArray(1f, manager));
        NDArray lNeutral = binaryCrossEntropy(trueParts.get(3), predParts.get(3)).mul(EngineUtils.floatToNDArray(2f, manager));
        NDArray lShort =  binaryCrossEntropy(trueParts.get(4), predParts.get(4)).mul(EngineUtils.floatToNDArray(1f, manager));

        // 3. PLUS DE PENALIZACIÓN (Inversión de tendencia)
        NDArray crossError = trueParts.get(2).mul(predParts.get(4))  // Long real * Predicción Short
                        .add(trueParts.get(4).mul(predParts.get(2)))        // Short real * Predicción Long
                        .mul(EngineUtils.floatToNDArray(3f, manager)).mean();


        NDArray totalLoss = lossTP.add(lossSL).add(lLong).add(lNeutral).add(lShort).add(crossError);

        // 2. ¿Alguien está esperando datos? (Sincronización Inteligente)
        // Solo entramos aquí si llamaste a awaitNextBatchData()
        CompletableFuture<LossReport> request = dataRequest;
        if (request != null && !request.isDone()) {
            // Solo aquí pagamos el costo de sincronización GPU -> CPU
            request.complete(new LossReport(
                    totalLoss.getFloat(),
                    lossTP.getFloat(),
                    lossSL.getFloat(),
                    lLong.getFloat(),
                    lNeutral.getFloat(),
                    lShort.getFloat()
            ));
            dataRequest = null; // Limpiamos la petición
        }
//        this.lastPredictionId = yPred.getUid();
//        if (lastResult != null) lastResult.close();
//        this.lastResult = totalLoss.duplicate();
        return totalLoss;
    }

    private NDArray TPSLAdvance(NDList trueParts, NDArray slTrue, NDArray slPred) {
        NDManager manager = slTrue.getManager();;
        NDArray isNeutral = trueParts.get(3);
        NDArray mask = isNeutral.mul(EngineUtils.floatToNDArray(-1f, manager)).add(EngineUtils.floatToNDArray(1f, manager));

        NDArray loss = slTrue.sub(slPred)
                .abs()
                .mul(mask)
                .mul(EngineUtils.floatToNDArray(0.7f, manager));

        NDArray valid = mask.sum().maximum(EngineUtils.floatToNDArray(1e-7f, manager));

        return loss.sum().div(valid);
    }

    // Método auxiliar para usar la implementación nativa más rápida
    private NDArray binaryCrossEntropy(NDArray target, NDArray prediction) {
        NDManager manager = target.getManager();
        NDArray p = prediction.minimum(EngineUtils.floatToNDArray(1.0f - 1e-7f, manager)).maximum(EngineUtils.floatToNDArray(1e-7f, manager));
        return target.mul(p.log()).add(target.sub(EngineUtils.floatToNDArray(1f, manager)).neg().mul(p.sub(EngineUtils.floatToNDArray(1f, manager)).neg().log())).neg().mean();
    }

    private NDArray categoricalCrossEntropy(NDArray target, NDArray prediction) {
        NDManager manager = target.getManager();
        NDArray p = prediction.minimum(EngineUtils.floatToNDArray(1e-7f, manager)).maximum(EngineUtils.floatToNDArray(1.0f - 1e-7f, manager));
        return target.mul(p.log()).sum(new int[]{1}).neg().mean();
    }

    private NDArray directionalPenaltySoft(NDArray trueOneHot, NDArray probs) {
        NDArray pTrue = trueOneHot.mul(probs).sum(new int[]{1}, true);
        NDArray pFalse = pTrue.mul(-1f).add(1f);
        NDArray penaltyPerSample = pFalse.pow(2);
        return penaltyPerSample.mean();
    }
    /**
     * Este método bloquea el hilo que lo llama hasta que el entrenamiento
     * termine el siguiente batch y devuelva los resultados.
     */
    public LossReport awaitNextBatchData() {
        dataRequest = new CompletableFuture<>();
        try {
            // Se queda bloqueado aquí hasta que evaluate() llame a .complete()
            return dataRequest.get();
        } catch (InterruptedException | ExecutionException e) {
            Thread.currentThread().interrupt();
            return new LossReport(0,0,0,0,0,0);
        }
    }
}