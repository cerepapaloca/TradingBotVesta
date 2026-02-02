package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
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

        if (yPred.getUid().equals(lastPredictionId) && lastResult != null) {
            return lastResult;
        }


        // 1. Cálculos vectorizados rápidos (GPU)
        NDList trueParts = yTrue.split(new long[]{1, 2, 3, 4}, 1);
        NDList predParts = yPred.split(new long[]{1, 2, 3, 4}, 1);

        CompletableFuture<NDArray> lossTP = CompletableFuture.supplyAsync(() -> trueParts.get(0).sub(predParts.get(0)).mul(0.7f).abs().mean(), VestaEngine.EXECUTOR_TRAINING);
        CompletableFuture<NDArray> lossSL = CompletableFuture.supplyAsync(() -> trueParts.get(1).sub(predParts.get(1)).mul(0.7f).abs().mean(), VestaEngine.EXECUTOR_TRAINING);

        CompletableFuture<NDArray> lLong = CompletableFuture.supplyAsync(() -> binaryCrossEntropy(trueParts.get(2), predParts.get(2)).mul(1.0f), VestaEngine.EXECUTOR_TRAINING);
        CompletableFuture<NDArray> lNeutral = CompletableFuture.supplyAsync(() -> binaryCrossEntropy(trueParts.get(3), predParts.get(3)).mul( 2.0f), VestaEngine.EXECUTOR_TRAINING);
        CompletableFuture<NDArray> lShort = CompletableFuture.supplyAsync(() -> binaryCrossEntropy(trueParts.get(4), predParts.get(4)).mul(1.0f), VestaEngine.EXECUTOR_TRAINING);

        // 3. PLUS DE PENALIZACIÓN (Inversión de tendencia)
        CompletableFuture<NDArray> crossError = CompletableFuture.supplyAsync(() ->
                trueParts.get(2).mul(predParts.get(4))  // Long real * Predicción Short
                        .add(trueParts.get(4).mul(predParts.get(2)))        // Short real * Predicción Long
                        .mul(3f).mean(), VestaEngine.EXECUTOR_TRAINING);


        NDArray totalLoss = lossTP.get().add(lossSL.get()).add(lLong.get()).add(lNeutral.get()).add(lShort.get()).add(crossError.get());

        // 2. ¿Alguien está esperando datos? (Sincronización Inteligente)
        // Solo entramos aquí si llamaste a awaitNextBatchData()
        CompletableFuture<LossReport> request = dataRequest;
        if (request != null && !request.isDone()) {
            // Solo aquí pagamos el costo de sincronización GPU -> CPU
            request.complete(new LossReport(
                    totalLoss.getFloat(),
                    lossTP.get().getFloat(),
                    lossSL.get().getFloat(),
                    lLong.get().getFloat(),
                    lNeutral.get().getFloat(),
                    lShort.get().getFloat()
            ));
            dataRequest = null; // Limpiamos la petición
        }
        this.lastPredictionId = yPred.getUid();
        if (lastResult != null) lastResult.close();
        this.lastResult = totalLoss.duplicate();
        return totalLoss;
    }

    // Método auxiliar para usar la implementación nativa más rápida
    private NDArray binaryCrossEntropy(NDArray target, NDArray prediction) {
        NDArray p = prediction.clip(1e-7, 1.0 - 1e-7);
        return target.mul(p.log()).add(target.sub(1).neg().mul(p.sub(1).neg().log())).neg().mean();
    }

    private NDArray categoricalCrossEntropy(NDArray target, NDArray prediction) {
        NDArray p = prediction.clip(1e-7, 1.0 - 1e-7);
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