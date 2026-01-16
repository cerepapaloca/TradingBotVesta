package xyz.cereshost;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;

import java.util.ArrayList;
import java.util.List;

public class MetricsListener extends TrainingListenerAdapter {
    private final List<Float> trainLoss = new ArrayList<>();
    private final List<Float> trainMae  = new ArrayList<>();

    private long lastTime = -1;

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();

        trainLoss.add(result.getTrainLoss());
        trainMae.add(result.getTrainEvaluation("mae"));
        double progress = (double) trainer.getTrainingResult().getEpoch() / VestaEngine.EPOCH;
        long time = System.currentTimeMillis();
        long delta = Math.abs(lastTime - time);
        System.out.printf(
                "Progreso=%.2f Tiempo=%.2fs -T=%sm [%s]\n",
                (progress)*100,
                (double) delta/1000,
                (int) (((VestaEngine.EPOCH - trainer.getTrainingResult().getEpoch())*delta)/1000)/60,
                "#".repeat((int) (progress*100)) + " " .repeat((int) (Math.abs(progress-1)*100))
        );

        lastTime = time;
    }

    public List<Float> getLoss() {
        return trainLoss;
    }

    public List<Float> getMae() {
        return trainMae;
    }
}
