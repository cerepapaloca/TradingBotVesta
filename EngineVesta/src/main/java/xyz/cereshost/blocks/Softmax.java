package xyz.cereshost.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import lombok.Builder;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.engine.EngineUtils;

@Builder
public class Softmax extends AbstractBlock {


    private static final byte VERSION = 1;

    private final float temperature;

    @Builder
    public Softmax(float temperature) {
        super(VERSION);
        if (temperature <= 0f) {
            throw new IllegalArgumentException("temperature must be > 0");
        }
        this.temperature = temperature;
    }

    @Override
    protected NDList forwardInternal(
            @NotNull ParameterStore ps,
            @NotNull NDList inputs,
            boolean training,
            ai.djl.util.PairList<String, Object> params
    ) {
        NDArray input = inputs.singletonOrThrow();

        // input / temperature
        NDArray scaled = input.div(EngineUtils.floatToNDArray(temperature, input.getManager()));

        // estabilidad numérica
        NDArray max = scaled.max(new int[]{-1}, true);
        NDArray shifted = scaled.sub(max);

        NDArray exp = shifted.exp();
        NDArray sum = exp.sum(new int[]{-1}, true);

        NDArray output = exp.div(sum);
        return new NDList(output);
    }

    @Override
    public Shape[] getOutputShapes(@NotNull Shape[] inputShapes) {
        // Softmax no cambia la forma
        return inputShapes;
    }

    @Override
    public void initializeChildBlocks(
            @NotNull NDManager manager,
            @NotNull DataType dataType,
            @NotNull Shape... inputShapes
    ) {
        // No tiene parámetros entrenables
    }
}
