package xyz.cereshost.utils;

import ai.djl.training.Trainer;
import ai.djl.training.listener.TrainingListenerAdapter;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.VestaEngine;

public class AutoStopListener extends TrainingListenerAdapter {

    private float minLast = Float.MAX_VALUE;
    private float maxLast = Float.MAX_VALUE;
    private float lossLast = Float.MAX_VALUE;

    private static final float THRESHOLD_STOP = 8f;
    private static final int MAX_GAMMA = 35;
    private static final int MAX_BETA = 50;
    private static final int MIN_EPOCH = 2;
    private final Runnable onStop;

    public AutoStopListener(){
        onStop = VestaEngine::stopTraining;
    }

    public AutoStopListener(Runnable onStop){
        this.onStop = onStop;
    }

    private int gamma = 0;
    private int beta = 0;

    @Override
    public void onEpoch(Trainer trainer) {
        var result = trainer.getTrainingResult();

        float minValidation = result.getValidateEvaluation("min_diff");
        float maxValidation = result.getValidateEvaluation("max_diff");
        float lossValidation = result.getValidateLoss();
        if (minLast != Float.MAX_VALUE &&  maxLast!= Float.MAX_VALUE &&  lossLast != Float.MAX_VALUE && MIN_EPOCH < result.getEpoch()) {
            float diffMin = ((minValidation - minLast)/ minLast)*100;
            float diffMax = ((maxValidation - maxLast)/ maxLast)*100;
            float diffLoss = ((lossValidation - lossLast)/ lossLast)*100;
            float total = (diffMax + diffMin) + diffLoss*2;
            if (total > THRESHOLD_STOP*4) {
                if (gamma > MAX_GAMMA){
                    onStop.run();
                }
                gamma++;
            }else gamma = 0;
            if (total > 0){
                if (beta > MAX_BETA){
                    onStop.run();
                }
                beta++;
            } else beta = 0;
            Vesta.info("Gamma:%d Beta:%d | %s %s %s = %s<%s".formatted(gamma, beta, diffMax, diffMin, diffLoss, (diffMax + diffMin) + diffLoss * 2, THRESHOLD_STOP * 4));
        }
        if (minLast > minValidation) minLast = minValidation;
        if (maxLast > maxValidation) maxLast = maxValidation;
        if (lossLast > lossValidation) lossLast = lossValidation;
    }

    public void reset() {
        minLast = Float.MAX_VALUE;
        maxLast = Float.MAX_VALUE;
        lossLast = Float.MAX_VALUE;
        gamma = 0;
        beta = 0;
    }

}
