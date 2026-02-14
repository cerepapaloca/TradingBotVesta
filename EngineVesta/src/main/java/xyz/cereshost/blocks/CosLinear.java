package xyz.cereshost.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import lombok.Builder;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.utils.EngineUtils;

@Builder
public class CosLinear extends AbstractBlock {


    private static final byte VERSION = 1;

    @Builder
    private CosLinear() {
        super(VERSION);
    }

    @Override
    protected NDList forwardInternal(
            @NotNull ParameterStore ps,
            @NotNull NDList inputs,
            boolean training,
            ai.djl.util.PairList<String, Object> params
    ) {
        NDArray input = inputs.singletonOrThrow();
        NDManager manager = ps.getManager();

        NDArray output = input.mul(EngineUtils.floatToNDArray((float) Math.PI, manager)).cos().mul(input);
        // input / temperature

        return new NDList(output);
    }

    @Override
    public Shape[] getOutputShapes(@NotNull Shape[] inputShapes) {
        return inputShapes;
    }

    @Override
    public void initializeChildBlocks(
            @NotNull NDManager manager,
            @NotNull DataType dataType,
            @NotNull Shape... inputShapes
    ) {}
}
