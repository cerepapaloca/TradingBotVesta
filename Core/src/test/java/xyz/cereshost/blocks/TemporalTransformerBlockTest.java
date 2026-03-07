package xyz.cereshost.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.Test;
import xyz.cereshost.vesta.core.blocks.CausalMaskManager;
import xyz.cereshost.vesta.core.blocks.TemporalTransformerBlock;
import xyz.cereshost.vesta.core.engine.VestaEngine;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class TemporalTransformerBlockTest {

    @Test
    void causalMaskBlocksFutureTokens() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            setRootManager(manager);

            TemporalTransformerBlock block = TemporalTransformerBlock.builder()
                    .setModelDim(8)
                    .setNumHeads(2)
                    .setFeedForwardDim(16)
                    .setDropoutRate(0.0f)
                    .setMaxSequenceLength(4)
                    .setOftenClearCache(1000)
                    .build();

            Shape inputShape = new Shape(2, 4, 8);
            VestaEngine.setRootManager(manager);

            block.initialize(manager, DataType.FLOAT32, inputShape);

            NDArray input = manager.zeros(inputShape);
            NDArray base0 = manager.ones(new Shape(8));
            NDArray base1 = manager.ones(new Shape(8)).mul(2f);
            NDArray base2 = manager.ones(new Shape(8)).mul(3f);

            input.set(new NDIndex("0, 0, :"), base0);
            input.set(new NDIndex("0, 1, :"), base1);
            input.set(new NDIndex("0, 2, :"), base2);
            input.set(new NDIndex("0, 3, :"), manager.ones(new Shape(8)).mul(10f));

            input.set(new NDIndex("1, 0, :"), base0);
            input.set(new NDIndex("1, 1, :"), base1);
            input.set(new NDIndex("1, 2, :"), base2);
            input.set(new NDIndex("1, 3, :"), manager.ones(new Shape(8)).mul(100f));

            ParameterStore ps = new ParameterStore(manager, false);
            NDArray output = block.forward(ps, new NDList(input), false).singletonOrThrow();

            NDArray[] maskTriple = CausalMaskManager.getMaskAndDerived(manager, 4, DataType.FLOAT32, 1000);

            NDArray prefix0 = output.get(new NDIndex("0, 0:3, :"));
            NDArray prefix1 = output.get(new NDIndex("1, 0:3, :"));
            float maxDiff = prefix0.sub(prefix1).abs().max().getFloat();
            assertTrue(maxDiff < 1e-4f, "Future token should not affect earlier outputs");
        }
    }

    @Test
    void pastTokensAffectLaterOutputs() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            setRootManager(manager);

            TemporalTransformerBlock block = TemporalTransformerBlock.builder()
                    .setModelDim(8)
                    .setNumHeads(2)
                    .setFeedForwardDim(16)
                    .setDropoutRate(0.0f)
                    .setMaxSequenceLength(4)
                    .setOftenClearCache(1000)
                    .build();

            Shape inputShape = new Shape(1, 4, 8);
            block.initialize(manager, DataType.FLOAT32, inputShape);

            NDArray inputA = manager.zeros(inputShape);
            NDArray inputB = manager.zeros(inputShape);

            NDArray strongSignal = manager.ones(new Shape(8)).mul(50f);
            inputA.set(new NDIndex("0, 0, :"), strongSignal);

            ParameterStore ps = new ParameterStore(manager, false);
            NDArray outA = block.forward(ps, new NDList(inputA), false).singletonOrThrow();
            float[] outA3 = outA.get(new NDIndex("0, 3, :")).toFloatArray();

            NDArray outB = block.forward(ps, new NDList(inputB), false).singletonOrThrow();
            float[] outB3 = outB.get(new NDIndex("0, 3, :")).toFloatArray();

            float maxDiff = maxAbsDiff(outA3, outB3);
            assertTrue(maxDiff > 1e-4f, "Past tokens should influence later outputs");
        }
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float max = 0f;
        int len = Math.min(a.length, b.length);
        for (int i = 0; i < len; i++) {
            float d = Math.abs(a[i] - b[i]);
            if (d > max) max = d;
        }
        return max;
    }

    private static void setRootManager(NDManager manager) throws Exception {
        Field field = VestaEngine.class.getDeclaredField("rootManager");
        field.setAccessible(true);
        field.set(null, manager);
    }
}
