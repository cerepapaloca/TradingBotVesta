package xyz.cereshost.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.transformer.ScaledDotProductAttentionBlock;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScaledDotProductAttentionMaskTest {

    @Test
    void maskBlocksFutureTokensInSelfAttention() {
        try (NDManager manager = NDManager.newBaseManager()) {
            ScaledDotProductAttentionBlock attention =
                    ScaledDotProductAttentionBlock.builder()
                            .setHeadCount(2)
                            .setEmbeddingSize(8)
                            .optAttentionProbsDropoutProb(0.0f)
                            .build();

            Shape inputShape = new Shape(2, 4, 8);
            attention.initialize(manager, DataType.FLOAT32, inputShape);

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

            float[] data = {
                    // batch 0
                    1,0,0,0,
                    1,1,0,0,
                    1,1,1,0,
                    1,1,1,1,

                    // batch 1
                    1,0,0,0,
                    1,1,0,0,
                    1,1,1,0,
                    1,1,1,1
            };

            NDArray mask = manager.create(data, new Shape(2, 4, 4)); // [B,F,F]

            ParameterStore ps = new ParameterStore(manager, false);
            NDArray output = attention.forward(ps, new NDList(input, mask), false).singletonOrThrow();

            NDArray prefix0 = output.get(new NDIndex("0, 0:3, :"));
            NDArray prefix1 = output.get(new NDIndex("1, 0:3, :"));
            float maxDiff = prefix0.sub(prefix1).abs().max().getFloat();

            assertTrue(maxDiff < 1e-4f, "Future token should not affect earlier outputs");
        }
    }
}
