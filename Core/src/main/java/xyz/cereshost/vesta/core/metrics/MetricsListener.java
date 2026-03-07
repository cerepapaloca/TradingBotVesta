package xyz.cereshost.vesta.core.metrics;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;
import org.jfree.data.xy.XYSeriesCollection;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.utils.ChartUtils;
import xyz.cereshost.vesta.core.Main;
import xyz.cereshost.vesta.core.engine.VestaEngine;
import xyz.cereshost.vesta.core.engine.VestaLoss;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class MetricsListener extends TrainingListenerAdapter {

    private long lastTime = -1;
    private long startTime = -1;
    private XYSeriesCollection datasetNormal = null;
    private XYSeriesCollection datasetLoss = null;
    private XYSeriesCollection datasetRoi = null;
    private final List<String> symbols = new ArrayList<>();

    private int count = 1;

    @Override
    public void onTrainingBegin(Trainer trainer) {
        startTime = System.currentTimeMillis();
    }

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();
        // Obtener los resultados
        float lossTrain = result.getTrainLoss();
        float lossValidation = result.getValidateLoss();
        float maeTrain = result.getTrainEvaluation("mae");
        float maeValidation = result.getValidateEvaluation("mae");
        float minValidation = result.getValidateEvaluation("min_diff");
        float maxValidation = result.getValidateEvaluation("max_diff");
        // Calucar porgreso
        double progress = (double) trainer.getTrainingResult().getEpoch() / (VestaEngine.EPOCH * (Main.MAX_MONTH_TRAINING) * VestaEngine.AUXILIAR_EPOCH);
        long time = System.currentTimeMillis();
        long delta = Math.abs(lastTime - time);
        // Mostrar mensaje
        Vesta.info(
                String.format("Progreso: %.2f%% | %.2f Bach/s | tiempo: %.2fs -T: %dm +T: %dm\n[%s] Bach: %,d",
                        (progress) * 100,
                        ((float) countBachOnEpoch) / (delta / 1000f),
                        (double) delta / 1000d,
                        (int) ((((VestaEngine.EPOCH * Main.MAX_MONTH_TRAINING * VestaEngine.AUXILIAR_EPOCH) - trainer.getTrainingResult().getEpoch()) * delta) / 1000) / 60,
                        (int) (((System.currentTimeMillis() - startTime) / 1000) / 60),
                        "#".repeat((int) (progress * 100)) + " ".repeat((int) (Math.abs(progress - 1) * 100)),
                        countBach
                )
        );

        // Ejecutar tarea de forma asincrónico
        VestaEngine.EXECUTOR.execute(() -> {
            lastTime = time;
            VestaLoss customLoss = (VestaLoss) trainer.getLoss();
            VestaLoss.LossReport l = customLoss.awaitNextBatchData();
            if (datasetLoss == null && datasetNormal == null) {
                datasetNormal = ChartUtils.plot("Training Loss/MAE " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss T", List.of(Math.min(lossTrain, 50)), new Color(41, 82, 33), ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("Loss V", List.of(Math.min(lossValidation, 50)), new Color(108, 217, 91), ChartUtils.DataPlot.StyleLine.NORMAL),
                                new ChartUtils.DataPlot("MAE T", List.of(Math.min(maeTrain, 50)), new Color(91, 63, 63), ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("MAE V", List.of(Math.min(maeValidation, 50)), Color.PINK, ChartUtils.DataPlot.StyleLine.NORMAL)
                        )
                );
                datasetLoss = ChartUtils.plot("Training Losses Max/Min " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss max", List.of(l.max()), new Color(0, 64, 0), ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("Loss min", List.of(l.min()), new Color(64, 0, 0), ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("max", List.of(maxValidation), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL),
                                new ChartUtils.DataPlot("min", List.of(minValidation), Color.RED, ChartUtils.DataPlot.StyleLine.NORMAL)
                        )
                );
                datasetRoi = ChartUtils.plot("diff " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("diff", List.of(l.meanReal() - l.meanPred()), Color.RED, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("diff Abs", List.of(Math.abs(l.meanReal() - l.meanPred())), new Color(64, 0, 0), ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                        new ChartUtils.DataPlot("real", List.of(l.meanReal()), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL),
                        new ChartUtils.DataPlot("pred", List.of(l.meanPred()), new Color(108, 217, 91), ChartUtils.DataPlot.StyleLine.DISCONTINUA)
                ));
            }
            datasetNormal.getSeries("Loss T").add(count, lossTrain);
            datasetNormal.getSeries("MAE T").add(count, maeTrain);
            datasetNormal.getSeries("Loss V").add(count, lossValidation);
            datasetNormal.getSeries("MAE V").add(count, maeValidation);
            datasetLoss.getSeries("max").add(count, maxValidation);
            datasetLoss.getSeries("min").add(count, minValidation);
            datasetLoss.getSeries("Loss max").add(count, l.max());
            datasetLoss.getSeries("Loss min").add(count, l.min());
            datasetRoi.getSeries("diff").add(count, l.meanReal() - l.meanPred());
            datasetRoi.getSeries("diff Abs").add(count, Math.abs(l.meanReal() - l.meanPred()));
            datasetRoi.getSeries("real").add(count, l.meanReal());
            datasetRoi.getSeries("pred").add(count, l.meanPred());

            count++;
        });
        countBachOnEpoch = 0;
    }

    private int countBach = 0;
    private int countBachOnEpoch = 0;

    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {
        countBach++;
        countBachOnEpoch++;
    }

}
