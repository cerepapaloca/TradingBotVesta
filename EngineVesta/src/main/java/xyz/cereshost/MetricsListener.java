package xyz.cereshost;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;
import org.jfree.data.category.DefaultCategoryDataset;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.engine.VestaLoss;

import java.util.ArrayList;
import java.util.List;

public class MetricsListener extends TrainingListenerAdapter {
    private final List<Float> trainLoss = new ArrayList<>();
    private final List<Float> trainMae  = new ArrayList<>();

    private long lastTime = -1;
    private double lastMae = -1;
    private DefaultCategoryDataset datasetNormal = null;
    private DefaultCategoryDataset datasetLoss = null;
    private DefaultCategoryDataset datasetDireccion = null;
    private final List<String> symbols = new ArrayList<>();

    public void MetricsListener(List<String> symbols) {
        this.symbols.addAll(symbols);
    }

    private int count = 1;

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();


        float loss = result.getTrainLoss();
        float mae = result.getTrainEvaluation("mae");
        trainLoss.add(loss);
        trainMae.add(mae);
        double progress = (double) trainer.getTrainingResult().getEpoch() / VestaEngine.EPOCH;
        long time = System.currentTimeMillis();
        long delta = Math.abs(lastTime - time);
        Vesta.info(
                String.format("Progreso=%.2f Tiempo=%.2fs -T=%sm MAEr=%.4f [%s]\n",
                        (progress)*100,
                        (double) delta/1000,
                        (int) (((VestaEngine.EPOCH - trainer.getTrainingResult().getEpoch())*delta)/1000)/60,
                        lastMae - mae,
                        "#".repeat((int) (progress*100)) + " " .repeat((int) (Math.abs(progress-1)*100))
                )
        );


        VestaEngine.EXECUTOR.execute(() -> {
            lastMae = mae;
            lastTime = time;
            VestaLoss customLoss = (VestaLoss) trainer.getLoss();
            VestaLoss.LossReport l = customLoss.awaitNextBatchData();
            if (datasetLoss == null || datasetNormal == null) {
                datasetNormal = ChartUtils.plot("Training Loss/MAE " + String.join(", ", symbols), "epochs",
                        List.of(new ChartUtils.DataPlot("Loss", List.of(loss)),
                                new ChartUtils.DataPlot("MAE", List.of(mae))
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

            }
            datasetNormal.addValue(loss, "Loss", String.valueOf(count));
            datasetNormal.addValue(mae, "MAE", String.valueOf(count));
            datasetLoss.addValue(l.tp(), "Loss TP", String.valueOf(count));
            datasetLoss.addValue(l.sl(), "Loss SL", String.valueOf(count));
            datasetDireccion.addValue(l.longL(), "Loss L", String.valueOf(count));
            datasetDireccion.addValue(l.shortL(), "Loss S", String.valueOf(count));
            datasetDireccion.addValue(l.neutralL(), "Loss N", String.valueOf(count));
            count++;
        });
    }
}
