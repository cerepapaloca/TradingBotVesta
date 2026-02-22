package xyz.cereshost.utils;

import ai.djl.Model;
import ai.djl.training.ParameterServer;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingConfig;

import java.lang.reflect.Field;

public class VestaTrainer extends Trainer {
    /**
     * Creates an instance of {@code Trainer} with the given {@link Model} and {@link
     * TrainingConfig}.
     *
     * @param model          the model the trainer will train on
     * @param trainingConfig the configuration used by the trainer
     */
    public VestaTrainer(Model model, TrainingConfig trainingConfig) {
        super(model, trainingConfig);
        try{
            Field fieldManager = Trainer.class.getDeclaredField("parameterStore");
            fieldManager.setAccessible(true);
            ParameterStore parameterStore = (ParameterStore) fieldManager.get((Trainer) this);

            VestaParameterStore vesta = new VestaParameterStore(parameterStore.getManager(), false);
            ParameterServer parameterServer =
                    parameterStore.getManager().getEngine().newParameterServer(trainingConfig.getOptimizer());
            vesta.setParameterServer(parameterServer ,this.getDevices());
            fieldManager.set((Trainer) this, vesta);
        }catch(NoSuchFieldException | SecurityException | IllegalArgumentException | IllegalAccessException ignored){}
    }
}
