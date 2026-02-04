package xyz.cereshost.metrics;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;
import org.jfree.data.xy.XYSeriesCollection;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.engine.VestaLoss;

import java.awt.*;
import java.util.ArrayList;
import java.util.List;

public class MetricsListener extends TrainingListenerAdapter {

    private long lastTime = -1;
    private double lastMae = -1;
    private XYSeriesCollection datasetNormal = null;
    private XYSeriesCollection datasetLoss = null;
    private XYSeriesCollection datasetDireccion = null;
    private XYSeriesCollection datasetRateDireccion = null;
    private final List<String> symbols = new ArrayList<>();

    private int count = 1;

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();

        // Obtener los resultados
        float lossTrain = result.getTrainLoss();
        float lossValidation = result.getValidateLoss();
        float maeTrain = result.getTrainEvaluation("mae");
        float daeTrain = result.getTrainEvaluation("DirrAcc");
        float daelsTrain = result.getTrainEvaluation("BinDirAcc");
        float maeValidation = result.getValidateEvaluation("mae");
        float daeValidation = result.getValidateEvaluation("DirrAcc");
        float daelsValidation = result.getValidateEvaluation("BinDirAcc");
        // Calucar porgreso
        double progress = (double) trainer.getTrainingResult().getEpoch() / (VestaEngine.EPOCH* (Main.MAX_MONTH_TRAINING/VestaEngine.SPLIT_DATASET) *VestaEngine.EPOCH_SUB);
        long time = System.currentTimeMillis();
        long delta = Math.abs(lastTime - time);
        // Mostrar mensaje
        Vesta.info(
                String.format("Progreso=%.2f Tiempo=%.2fs -T=%sm MAEr=%.4f [%s]\n",
                        (progress)*100,
                        (double) delta/1000,
                        (int) ((((VestaEngine.EPOCH*Main.MAX_MONTH_TRAINING*VestaEngine.EPOCH_SUB) - trainer.getTrainingResult().getEpoch())*delta)/1000)/60,
                        lastMae - maeValidation,
                        "#".repeat((int) (progress*100)) + " " .repeat((int) (Math.abs(progress-1)*100))
                )
        );

        // Ejecutar tarea de forma asincrónico
        VestaEngine.EXECUTOR.execute(() -> {
            lastMae = maeValidation;
            lastTime = time;
            VestaLoss customLoss = (VestaLoss) trainer.getLoss();
            VestaLoss.LossReport l = customLoss.awaitNextBatchData();
            if (datasetLoss == null || datasetNormal == null || datasetRateDireccion == null) {
                datasetNormal = ChartUtils.plot("Training Loss/MAE " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss T", List.of(lossTrain), Color.GREEN, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("MAE T", List.of(maeTrain), Color.RED, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("Loss V", List.of(lossValidation), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL),
                                new ChartUtils.DataPlot("MAE V", List.of(maeValidation), Color.RED, ChartUtils.DataPlot.StyleLine.NORMAL)

                        )
                );
                datasetLoss = ChartUtils.plot("Training Losses TP/SL " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss TP", List.of(l.tp())),
                                new ChartUtils.DataPlot("Loss SL", List.of(l.sl()))
                        )
                );
                datasetDireccion = ChartUtils.plot("Training Losses Dirección" + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss L", List.of(l.longL())),
                                new ChartUtils.DataPlot("Loss S", List.of(l.shortL())),
                                new ChartUtils.DataPlot("Loss N", List.of(l.neutralL()))
                        )
                );
                datasetRateDireccion = ChartUtils.plot("Rate Dirección" + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("DAE T", List.of(daeTrain), Color.RED, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("DAELS T", List.of(daelsTrain), Color.GREEN, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("DAE V", List.of(daeValidation), Color.RED, ChartUtils.DataPlot.StyleLine.NORMAL),
                                new ChartUtils.DataPlot("DAELS V", List.of(daelsValidation), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL)
                        )

                );

            }
            datasetNormal.getSeries("Loss T").add(count, lossTrain);
            datasetNormal.getSeries("MAE T").add(count, maeTrain);
            datasetNormal.getSeries("Loss V").add(count, lossValidation);
            datasetNormal.getSeries("MAE V").add(count, maeValidation);
            datasetRateDireccion.getSeries("DAE T").add(count, daeTrain);
            datasetRateDireccion.getSeries("DAELS T").add(count, daelsTrain);
            datasetRateDireccion.getSeries("DAE V").add(count, daeValidation);
            datasetRateDireccion.getSeries("DAELS V").add(count, daelsValidation);
            datasetLoss.getSeries("Loss TP").add(count, l.tp());
            datasetLoss.getSeries("Loss SL").add(count, l.sl());
            datasetDireccion.getSeries( "Loss L").add(count, l.longL());
            datasetDireccion.getSeries("Loss S").add(count, l.shortL());
            datasetDireccion.getSeries( "Loss N").add(count, l.neutralL());
            count++;
        });
    }
}
