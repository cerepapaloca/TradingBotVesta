package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.loss.Loss;
import lombok.SneakyThrows;
import xyz.cereshost.utils.EngineUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class VestaLoss extends Loss {

    public record LossReport(
            float total, float tp, float sl, float longL, float neutralL, float shortL, float directionMemory, float biasPenalty
    ) {}

    private static final float NEUTRAL_GRAD_CLIP = 0.35f;
    private static final float DIRECTION_MEMORY_DECAY = 0.8f;
    private static final float DIRECTION_MEMORY_LR = 1.0f - DIRECTION_MEMORY_DECAY;
    private static final float BIAS_PENALTY_SCALE = 0f;

    private volatile CompletableFuture<LossReport> dataRequest = null;
    private NDArray directionMemory = null;

    public VestaLoss(String name) {
        super(name);
    }

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
        NDArray lossTP = TPSLAdvance(trueParts, trueParts.get(0), predParts.get(0)).mul(EngineUtils.floatToNDArray(0.6f, manager));
        NDArray lossSL = TPSLAdvance(trueParts, trueParts.get(1), predParts.get(1)).mul(EngineUtils.floatToNDArray(0.6f, manager));;
        NDArray lLong = directionalPenaltySoft(trueParts.get(2), predParts.get(2)).mul(EngineUtils.floatToNDArray(1f, manager));
        //NDArray clippedNeutralPred = clipNeutralGradient(trueParts.get(3), predParts.get(3), NEUTRAL_GRAD_CLIP);
        NDArray lNeutral = binaryCrossEntropy(trueParts.get(3), predParts.get(3)).mul(EngineUtils.floatToNDArray(2f, manager));
        NDArray lShort =  directionalPenaltySoft(trueParts.get(4), predParts.get(4)).mul(EngineUtils.floatToNDArray(1f, manager));

        // 3. PLUS DE PENALIZACIÓN (Inversión de tendencia)
        NDArray longTrue = trueParts.get(2);
        NDArray shortTrue = trueParts.get(4);
        NDArray directionMask = EngineUtils.floatToNDArray(1f, manager).sub(trueParts.get(3));
        NDArray crossError = longTrue.mul(predParts.get(4))  // Long real * Predicción Short
                        .add(shortTrue.mul(predParts.get(2)))        // Short real * Predicción Long
                        .mul(directionMask)
                        .mul(EngineUtils.floatToNDArray(2f, manager)).mean();

        NDArray biasPenalty = computeDirectionalBiasPenalty(trueParts, predParts, manager);


        NDArray directionLoss = lLong.add(lNeutral).add(lShort).add(crossError).add(biasPenalty);
        NDArray totalLoss = lossTP.add(lossSL).add(directionLoss);
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
                    lShort.getFloat(),
                    directionMemory.getFloat(),
                    biasPenalty.getFloat()
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

    private NDArray clipNeutralGradient(NDArray neutralTrue, NDArray neutralPred, float limit) {
        NDManager manager = neutralPred.getManager();
        NDArray clipMin = EngineUtils.floatToNDArray(-limit, manager);
        NDArray clipMax = EngineUtils.floatToNDArray(limit, manager);
        NDArray diff = neutralPred.sub(neutralTrue);
        NDArray clipped = diff.maximum(clipMin).minimum(clipMax);
        return neutralTrue.add(clipped);
    }

    private NDArray computeDirectionalBiasPenalty(NDList trueParts, NDList predParts, NDManager batchManager) {
        NDArray trueDir = NDArrays.concat(new NDList(trueParts.get(2), trueParts.get(3), trueParts.get(4)), 1);   // [B,3]
        NDArray predDir = NDArrays.concat(new NDList(predParts.get(2), predParts.get(3), predParts.get(4)), 1);   // [B,3]

        NDArray trueIdx = trueDir.argMax(1); // 0=Long,1=Neutral,2=Short
        NDArray predIdx = predDir.argMax(1);

        NDArray correctMask = predIdx.eq(trueIdx).toType(DataType.FLOAT32, false);
        NDArray wrongMask = EngineUtils.floatToNDArray(1f, batchManager).sub(correctMask);

        NDArray sign = predIdx.eq(EngineUtils.floatToNDArray(0, batchManager)).toType(DataType.FLOAT32, false)
                .sub(predIdx.eq(EngineUtils.floatToNDArray(2, batchManager)).toType(DataType.FLOAT32, false)); // +1 long, -1 short, 0 neutral

        NDArray signCorrect = sign.mul(correctMask);
        NDArray signWrong = sign.mul(wrongMask);

        // Lazy init or resurrect directionMemory
        if (directionMemory == null) {
            NDManager memMgr = VestaEngine.getRootManager() != null ? VestaEngine.getRootManager() : batchManager;
            directionMemory = memMgr.zeros(new Shape(1));
        }

        // Update memory in its own manager (EMA of correct signed hits)
        NDManager memMgr = directionMemory.getManager();
        NDArray signCorrectMem = signCorrect.toDevice(memMgr.getDevice(), false);
        NDArray decay = EngineUtils.floatToNDArray(DIRECTION_MEMORY_DECAY, memMgr);
        NDArray lr = EngineUtils.floatToNDArray(DIRECTION_MEMORY_LR, memMgr);
        directionMemory = directionMemory.mul(decay).add(signCorrectMem.mean().mul(lr));

        // Penalty uses wrong signed predictions scaled by current memory (bias)
        NDArray memoryLocal = directionMemory.toDevice(batchManager.getDevice(), false);
        NDArray memorySign = memoryLocal.sign(); // -1, 0, +1
        // signWrong: +1 wrong-long, -1 wrong-short, 0 otherwise
        NDArray targetedWrongMask = signWrong.mul(memorySign).gt(EngineUtils.floatToNDArray(0f, batchManager)).toType(DataType.FLOAT32, false); // 1 donde wrong y predicho coincide con signo de memoria

        NDArray bpPerSample = targetedWrongMask.mul(memoryLocal.abs()); // solo penaliza esos
        return bpPerSample.mean().mul(EngineUtils.floatToNDArray(BIAS_PENALTY_SCALE, batchManager));
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
        NDArray pFalse = pTrue.mul(EngineUtils.floatToNDArray(-1f, trueOneHot.getManager())).add(EngineUtils.floatToNDArray(1f, trueOneHot.getManager()));
        NDArray penaltyPerSample = pFalse.pow(EngineUtils.floatToNDArray(2f, trueOneHot.getManager()));
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
            return new LossReport(0,0,0,0,0,0, 0, 0);
        }
    }
}
