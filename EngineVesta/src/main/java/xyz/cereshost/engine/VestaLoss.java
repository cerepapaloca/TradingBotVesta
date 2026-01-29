package xyz.cereshost.engine;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.training.loss.Loss;
import lombok.SneakyThrows;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicReference;

public class VestaLoss extends Loss {
    private final float directionWeight;
    private final float classificationWeight;
    public record LossReport(
            float total, float tp, float sl, float longL, float neutralL, float shortL
    ) {}

    private volatile CompletableFuture<LossReport> dataRequest = null;

    public VestaLoss(String name, float directionWeight, float classificationWeight) {
        super(name);
        this.directionWeight = directionWeight;
        this.classificationWeight = classificationWeight;
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

        NDArray lossTP = trueParts.get(0).sub(predParts.get(0)).abs().mean();
        NDArray lossSL = trueParts.get(1).sub(predParts.get(1)).abs().mean();

        CountDownLatch latch = new CountDownLatch(3);
        AtomicReference<NDArray> lLong = new AtomicReference<>();
        VestaEngine.EXECUTOR_TRAINING.submit(() -> {
            lLong.set(binaryCrossEntropy(trueParts.get(2), predParts.get(2)).mul(1));
            latch.countDown();
        });
        AtomicReference<NDArray> lNeutral = new AtomicReference<>();
        VestaEngine.EXECUTOR_TRAINING.submit(() -> {
            lNeutral.set(binaryCrossEntropy(trueParts.get(3), predParts.get(3)).mul(1));
            latch.countDown();
        });
        AtomicReference<NDArray> lShort = new AtomicReference<>();
        VestaEngine.EXECUTOR_TRAINING.submit(() -> {
            lShort.set(binaryCrossEntropy(trueParts.get(4), predParts.get(4)).mul(1));
            latch.countDown();
        });
        latch.await();

        // 3. PLUS DE PENALIZACIÓN (Inversión de tendencia)
        NDArray crossError = trueParts.get(2).mul(predParts.get(4))  // Long real * Predicción Short
                .add(trueParts.get(4).mul(predParts.get(2)))        // Short real * Predicción Long
                .mul(2).mean(); // Peso de castigo alto (10) por ser el error más peligroso

        NDArray totalLoss = lossTP.add(lossSL).add(lLong.get()).add(lNeutral.get()).add(lShort.get()).add(crossError);

        // 2. ¿Alguien está esperando datos? (Sincronización Inteligente)
        // Solo entramos aquí si llamaste a awaitNextBatchData()
        CompletableFuture<LossReport> request = dataRequest;
        if (request != null && !request.isDone()) {
            // Solo aquí pagamos el costo de sincronización GPU -> CPU
            request.complete(new LossReport(
                    totalLoss.getFloat(),
                    lossTP.getFloat(),
                    lossSL.getFloat(),
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