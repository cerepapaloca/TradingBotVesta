package xyz.cereshost.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.ParameterStore;
import org.junit.jupiter.api.Test;
import xyz.cereshost.vesta.core.blocks.SoftPlusLogBlock;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class SoftPlusLogBlockTest {

    @Test
    void outputsVaryAcrossInputs() {
        try (NDManager manager = NDManager.newBaseManager()) {
            SoftPlusLogBlock block = SoftPlusLogBlock.builder().build();

            float[] inputs = new float[]{-6f, -3f, -1f, -0.1f, 0f, 0.1f, 1f, 3f};
            NDArray input = manager.create(inputs, new Shape(inputs.length, 1));

            block.initialize(manager, DataType.FLOAT32, input.getShape());
            ParameterStore ps = new ParameterStore(manager, false);
            NDArray output = block.forward(ps, new NDList(input), false).singletonOrThrow();

            float[] out = output.toFloatArray();
            float min = out[0];
            float max = out[0];
            for (float v : out) {
                if (v < min) min = v;
                if (v > max) max = v;
            }
            assertTrue(max - min > 1e-4f, "Outputs should vary across inputs");

            for (int i = 1; i < out.length; i++) {
                assertTrue(out[i] >= out[i - 1] - 1e-6f, "Output should be non-decreasing");
            }

            for (int i = 0; i < inputs.length; i++) {
                System.out.printf("SoftPlusLogBlock x=%f -> y=%f%n", inputs[i], out[i]);
            }
        }
    }
}
