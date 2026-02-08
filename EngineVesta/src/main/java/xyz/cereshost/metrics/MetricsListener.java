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
    private long startTime = -1;
    private XYSeriesCollection datasetNormal = null;
    private XYSeriesCollection datasetLoss = null;
    private XYSeriesCollection datasetDireccion = null;
    private XYSeriesCollection datasetRateDireccion = null;
    private XYSeriesCollection datasetDireccionMemory = null;
    private final List<String> symbols = new ArrayList<>();

    private int count = 1;

    public void startTraining() {
        startTime = System.currentTimeMillis();
    }

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();
        // Obtener los resultados
        float lossTrain = result.getTrainLoss();
        float lossValidation = result.getValidateLoss();
        float maeTrain = result.getTrainEvaluation("mae");
//        float daeTrain = result.getTrainEvaluation("3_dir");
//        float daelsTrain = result.getTrainEvaluation("2_dir");
        float maeValidation = result.getValidateEvaluation("mae");
        float minValidation = result.getValidateEvaluation("min_diff");
        float maxValidation = result.getValidateEvaluation("max_diff");
//        float daeValidation = result.getValidateEvaluation("3_dir");
//        float daelsValidation = result.getValidateEvaluation("2_dir");
        // Calucar porgreso
        double progress = (double) trainer.getTrainingResult().getEpoch() / (VestaEngine.EPOCH* (Main.MAX_MONTH_TRAINING/VestaEngine.SPLIT_DATASET) *VestaEngine.EPOCH_SUB);
        long time = System.currentTimeMillis();
        long delta = Math.abs(lastTime - time);
        // Mostrar mensaje
        Vesta.info(
                String.format("Progreso: %.2f%% Tiempo: %.2fs -T: %dm +T: %dm\n[%s]",
                        (progress)*100,
                        (double) delta/1000,
                        (int) ((((VestaEngine.EPOCH*Main.MAX_MONTH_TRAINING*VestaEngine.EPOCH_SUB) - trainer.getTrainingResult().getEpoch())*delta)/1000)/60,
                        (int) (((System.currentTimeMillis() - startTime)/1000)/60),
                        "#".repeat((int) (progress*100)) + " " .repeat((int) (Math.abs(progress-1)*100))
                )
        );

        // Ejecutar tarea de forma asincrónico
        VestaEngine.EXECUTOR.execute(() -> {
            lastTime = time;
            VestaLoss customLoss = (VestaLoss) trainer.getLoss();
            VestaLoss.LossReport l = customLoss.awaitNextBatchData();
            if (datasetLoss == null && datasetNormal == null && datasetRateDireccion == null) {
                datasetNormal = ChartUtils.plot("Training Loss/MAE " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss T", List.of(lossTrain), Color.GREEN, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("MAE T", List.of(maeTrain), Color.RED, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("Loss V", List.of(lossValidation), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL),
                                new ChartUtils.DataPlot("MAE V", List.of(maeValidation), Color.RED, ChartUtils.DataPlot.StyleLine.NORMAL)

                        )
                );
                datasetLoss = ChartUtils.plot("Training Losses TP/SL " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss Max", List.of(l.tp()), Color.GREEN, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("Loss Min", List.of(l.sl()), Color.RED, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
                                new ChartUtils.DataPlot("Max", List.of(maxValidation), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL),
                                new ChartUtils.DataPlot("Min", List.of(minValidation), Color.RED, ChartUtils.DataPlot.StyleLine.NORMAL)
                        )
                );
//                datasetDireccion = ChartUtils.plot("Training Losses Dirección" + String.join(", ", symbols), "epochs",
//                        List.of(new ChartUtils.DataPlot("Loss L", List.of(l.longL())),
//                                new ChartUtils.DataPlot("Loss S", List.of(l.shortL())),
//                                new ChartUtils.DataPlot("Loss N", List.of(l.neutralL()))
//                        )
//                );
//                datasetRateDireccion = ChartUtils.plot("Rate Dirección" + String.join(", ", symbols), "epochs",
//                        List.of(new ChartUtils.DataPlot("DAE T", List.of(daeTrain), Color.RED, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
//                                new ChartUtils.DataPlot("DAELS T", List.of(daelsTrain), Color.GREEN, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
//                                new ChartUtils.DataPlot("DAE V", List.of(daeValidation), Color.RED, ChartUtils.DataPlot.StyleLine.NORMAL),
//                                new ChartUtils.DataPlot("DAELS V", List.of(daelsValidation), Color.GREEN, ChartUtils.DataPlot.StyleLine.NORMAL)
//                        )
//                );
//                datasetDireccionMemory = ChartUtils.plot("Penalización por dirección" + String.join(", ", symbols), "epochs",
//                        List.of(new ChartUtils.DataPlot("Direction ABS", List.of(l.biasPenalty()), Color.ORANGE, ChartUtils.DataPlot.StyleLine.DISCONTINUA),
//                        new ChartUtils.DataPlot("Direction Relative", List.of(l.directionMemory()), Color.YELLOW, ChartUtils.DataPlot.StyleLine.NORMAL)
//                        )
//                );

            }
            datasetNormal.getSeries("Loss T").add(count, lossTrain);
            datasetNormal.getSeries("MAE T").add(count, maeTrain);
            datasetNormal.getSeries("Loss V").add(count, lossValidation);
            datasetNormal.getSeries("MAE V").add(count, maeValidation);
            datasetLoss.getSeries("Max").add(count, maxValidation);
            datasetLoss.getSeries("Min").add(count, minValidation);
//            datasetRateDireccion.getSeries("DAE T").add(count, daeTrain);
//            datasetRateDireccion.getSeries("DAELS T").add(count, daelsTrain);
//            datasetRateDireccion.getSeries("DAE V").add(count, daeValidation);
//            datasetRateDireccion.getSeries("DAELS V").add(count, daelsValidation);
            datasetLoss.getSeries("Loss Max").add(count, l.tp());
            datasetLoss.getSeries("Loss Min").add(count, l.sl());
//            datasetDireccion.getSeries( "Loss L").add(count, l.longL());
//            datasetDireccion.getSeries("Loss S").add(count, l.shortL());
//            datasetDireccion.getSeries( "Loss N").add(count, l.neutralL());
//            datasetDireccion.getSeries( "Loss N").add(count, l.neutralL());
//            datasetDireccionMemory.getSeries("Direction ABS").add(count, l.biasPenalty());
//            datasetDireccionMemory.getSeries("Direction Relative").add(count, l.directionMemory());
            count++;
        });
    }
}
