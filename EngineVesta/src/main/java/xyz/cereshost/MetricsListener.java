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
        trainMae.add(result.getTrainEvaluation("Accuracy"));

        System.out.printf(
                "Epoch %d | loss=%.6f | MAE=%.6f%n",
                trainer.getTrainingResult().getEpoch(),
                result.getTrainLoss(),
                result.getTrainEvaluation("Accuracy")
        );
    }

    public List<Float> getLoss() {
        return trainLoss;
    }

    public List<Float> getMae() {
        return trainMae;
    }
}
