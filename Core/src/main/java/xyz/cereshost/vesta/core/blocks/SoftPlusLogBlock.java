package xyz.cereshost.vesta.core.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.Parameter;
import ai.djl.util.PairList;
import lombok.Builder;
import xyz.cereshost.vesta.core.utils.EngineUtils;

/**
 * f(x) = alpha * (gamma * asinh(x) + (1 - gamma) * x)
 * Gamma se limita a [0,1] con sigmoid.
 * Alfa se limita a > 0 con softplus.
 */
public final class SoftPlusLogBlock extends AbstractBlock {

    private static final float EPS = 1e-6f;

    private static final float HEAD_GAMMA_INIT = 0.0f;
    private static final float HEAD_ALPHA_INIT = 1.0f;

    private final float initGamma;
    private final float initAlpha;
    private final boolean learnable;

    private final Parameter gammaParam;
    private final Parameter alphaParam;
    @Builder
    private SoftPlusLogBlock() {
        this(true);
    }

    @Builder
    private SoftPlusLogBlock(boolean learnable) {
        this.initGamma = HEAD_GAMMA_INIT;
        this.initAlpha = HEAD_ALPHA_INIT;
        this.learnable = learnable;

        if (learnable) {
            gammaParam = addParameter(Parameter.builder()
                    .setName("gamma")
                    .setType(Parameter.Type.BIAS)
                    .optShape(new Shape(1))
                    .build());
            alphaParam = addParameter(Parameter.builder()
                    .setName("alpha")
                    .setType(Parameter.Type.BIAS)
                    .optShape(new Shape(1))
                    .build());
        } else {
            gammaParam = null;
            alphaParam = null;
        }
    }


    @Override
    protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        if (!learnable) {
            return;
        }
        gammaParam.initialize(manager, dataType);
        alphaParam.initialize(manager, dataType);

//        float gammaInit = clamp(initGamma, 1e-3f, 1f - 1e-3f);
//        float gammaRaw = (float) Math.log(gammaInit / (1f - gammaInit)); // logit
//        float alphaInit = Math.max(initAlpha, EPS);
//        float alphaRaw = inverseSoftplus(alphaInit);

        gammaParam.getArray().set(new float[]{initGamma});
        alphaParam.getArray().set(new float[]{initAlpha});
    }

    @Override
    protected NDList forwardInternal(ai.djl.training.ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray x = inputs.singletonOrThrow();
        NDManager manager = parameterStore.getManager();
        NDArray gamma;
        NDArray alpha;
        if (learnable) {
            NDArray gammaRaw = parameterStore.getValue(gammaParam, x.getDevice(), training);
            NDArray alphaRaw = parameterStore.getValue(alphaParam, x.getDevice(), training);
            gamma = Activation.sigmoid(gammaRaw);
            alpha = Activation.softPlus(alphaRaw).add(EngineUtils.floatToNDArray(EPS, manager));
        } else {
            gamma = x.getManager().create(initGamma);
            alpha = x.getManager().create(initAlpha);
        }

        NDArray asinh = asinh(x, manager);
        NDArray one = EngineUtils.floatToNDArray(1f, manager);
        NDArray linearMix = asinh.mul(gamma).add(x.mul(one.sub(gamma)));
        NDArray out = linearMix.mul(alpha);
        return new NDList(out);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputs) {
        return inputs;
    }

    private static float clamp(float v, float min, float max) {
        return Math.max(min, Math.min(max, v));
    }

    private static float inverseSoftplus(float y) {
        double ey = Math.exp(Math.max(y, 1e-6));
        double v = Math.log(Math.max(ey - 1.0, 1e-6));
        return (float) v;
    }

    private static NDArray asinh(NDArray x, NDManager manager) {
        NDArray x2 = x.mul(x);
        NDArray sqrt = x2.add(EngineUtils.floatToNDArray(1f, manager)).sqrt();
        return x.add(sqrt).log();
    }
}
