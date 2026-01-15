package xyz.cereshost;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;

import java.util.ArrayList;
import java.util.List;

public class MetricsListener extends TrainingListenerAdapter {
    private final List<Float> trainLoss = new ArrayList<>();
    private final List<Float> trainMae  = new ArrayList<>();

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();

        trainLoss.add(result.getTrainLoss());
        trainMae.add(result.getTrainEvaluation("mae"));

        System.out.printf(
                "Progreso=%.6f\n",
                ((double) trainer.getTrainingResult().getEpoch()/Main.EPOCH)*100
        );
    }

    public List<Float> getLoss() {
        return trainLoss;
    }

    public List<Float> getMae() {
        return trainMae;
    }
}
