package xyz.cereshost.blocks;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.ScaledDotProductAttentionBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import xyz.cereshost.engine.VestaEngine;


/**
 * Bloque principal de Temporal Transformer para series de tiempo.
 * Implementación corregida según los problemas identificados.
 */
public class TemporalTransformerBlock extends AbstractBlock {

    private static final byte VERSION = 1;

    // Hiperparámetros configurables
    private final int modelDim;        // Dimensión del modelo (d_model)
    private final int maxSequenceLength; // Longitud máxima de secuencia
    private final int oftenClearCache; // Cada cuanto limpia la cache

    // Sub-bloques componentes
    private final ScaledDotProductAttentionBlock attentionBlock;
    private final Linear attentionOutputDense;
    private final LayerNorm layerNorm1;
    private final LayerNorm layerNorm2;
    private final SequentialBlock feedForwardBlock;
    private final Dropout dropout;
    //private NDArray causalMask; // Máscara causal precomputada
    private final Linear inputProjection; // Proyección de entrada para ajustar dimensiones
    private NDManager manager = null;

    /**
     * Constructor para crear un bloque Temporal Transformer configurable.
     */
    private TemporalTransformerBlock(int modelDim, int numHeads, int ffDim,
                                    float dropoutRate, int maxSequenceLength, int oftenClearCache) {
        super(VERSION);
        this.modelDim = modelDim;
        this.maxSequenceLength = maxSequenceLength;
        this.oftenClearCache = oftenClearCache;

        // Validación de parámetros
        if (modelDim % numHeads != 0) {
            throw new IllegalArgumentException(
                    "modelDim (" + modelDim + ") debe ser divisible entre numHeads (" + numHeads + ")"
            );
        }

        // 0. Proyección de entrada para ajustar dimensiones
        this.inputProjection = Linear.builder()
                .setUnits(modelDim)
                .optBias(true)
                .build();

        // 1. Bloque de atención escalada multi-cabeza
        this.attentionBlock = ScaledDotProductAttentionBlock.builder()
                .setHeadCount(numHeads)
                .setEmbeddingSize(modelDim)
                .optAttentionProbsDropoutProb(dropoutRate)
                .build();

        // 2. Capa densa para proyección después de la atención
        this.attentionOutputDense = Linear.builder()
                .setUnits(modelDim)
                .optBias(true)
                .build();

        // 3. Capas de normalización (LayerNorm)
        this.layerNorm1 = LayerNorm.builder().axis(2).optEpsilon(1e-6f).build();
        this.layerNorm2 = LayerNorm.builder().axis(2).optEpsilon(1e-6f).build();

        // 4. Capa feed-forward de 2 niveles (FFN)
        this.feedForwardBlock = new SequentialBlock()
                .add(Linear.builder().setUnits(ffDim).optBias(true).build())
                .add(Activation::gelu)
                .add(Dropout.builder().optRate(dropoutRate).build())
                .add(Linear.builder().setUnits(modelDim).optBias(true).build());

        // 5. Dropout para residuos
        this.dropout = Dropout.builder().optRate(dropoutRate).build();

        // Registrar todos los bloques hijos
        addChildBlock("input_projection", inputProjection);
        addChildBlock("attention", attentionBlock);
        addChildBlock("attention_output", attentionOutputDense);
        addChildBlock("layer_norm_1", layerNorm1);
        addChildBlock("layer_norm_2", layerNorm2);
        addChildBlock("feed_forward", feedForwardBlock);
        addChildBlock("dropout", dropout);
    }

    @Override
    protected void initializeChildBlocks(NDManager m, DataType dataType, Shape... inputShapes) {
        NDManager managerInit = m.newSubManager();
        Shape inputShape = inputShapes[0]; // (batch_size, seq_len, input_dim)

        // Inicializar proyección de entrada
        inputProjection.initialize(managerInit, dataType, inputShape);
        Shape projectedShape = inputProjection.getOutputShapes(new Shape[]{inputShape})[0];

        // Inicializar bloques hijos con la forma proyectada
        attentionBlock.initialize(managerInit, dataType, projectedShape, projectedShape, projectedShape);

        attentionOutputDense.initialize(managerInit, dataType, projectedShape);
        layerNorm1.initialize(managerInit, dataType, projectedShape);
        layerNorm2.initialize(managerInit, dataType, projectedShape);
        feedForwardBlock.initialize(managerInit, dataType, projectedShape);

    }

    

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        if (manager != null) manager.close();
        manager = inputs.getManager().newSubManager();
//        System.out.println("SubManager Inicio: " + manager);
//        System.out.println("Manager: " + inputs.getManager());
//        System.out.println("Manager Root: " + inputs.getManager().getParentManager().getParentManager().getParentManager().getParentManager());
        inputs.attach(manager);
        // Input shape: (batch_size, sequence_length, input_dim)
        NDArray hiddenStates = inputs.singletonOrThrow();

        // Variables para cerrar más tarde
        NDArray currentMask = null;
        NDArray attentionContext = null;
        NDArray projectedAttention = null;
        NDArray attentionResidual = null;
        NDArray attentionOutputNorm = null;
        NDArray ffnResult = null;
        NDArray ffnResidual = null;
        NDArray finalOutput = null;

        try {
            // Proyectar la entrada a modelDim
            NDArray projectedInput = inputProjection.forward(
                    parameterStore, new NDList(hiddenStates), training
            ).singletonOrThrow();

            long batchSize = projectedInput.getShape().get(0);
            long seqLength = projectedInput.getShape().get(1);
            long maskSide = Math.max(seqLength, maxSequenceLength);
            if (maskSide > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Sequence length (" + seqLength + ") exceeds supported int range");
            }
            int maskLength = (int) maskSide;

            // === 1. Atención Multi-Cabeza con Máscara Causal ===
            // Tomar slice de la máscara causal según la longitud actual
            NDArray baseMask =
                    CausalMaskManager
                            .getMaskAndDerived(
                                    VestaEngine.getRootManager(),
                                    maskLength,
                                    inputs.getFirst().getDataType(),
                                    oftenClearCache
                            )[0]; // [1,1,T,T]

// Slice correcto (mask 0/1) y expandir a [B, seq, seq]
            NDArray slicedMask2d =
                    baseMask.get(
                            new NDIndex("0, 0, 0:" + seqLength + ", 0:" + seqLength)
                    ); // [seq,seq]
            NDArray slicedMask3d = slicedMask2d.expandDims(0); // [1,seq,seq]

// EXPANDIR A BATCH (OBLIGATORIO EN DJL)
            currentMask = slicedMask3d.repeat(0, (int) batchSize); // [B,seq,seq]
            currentMask = currentMask.toDevice(manager.getDevice(), false);
            NDList attentionInput = new NDList(projectedInput, currentMask);

            NDList attentionOutput = attentionBlock.forward(parameterStore, attentionInput, training);
            attentionContext = attentionOutput.singletonOrThrow();
            // Proyección lineal después de atención
            NDList listAttentionContext = new NDList(attentionContext);
            projectedAttention = attentionOutputDense.forward(
                    parameterStore, listAttentionContext, training
            ).singletonOrThrow();
            // Dropout + residual connection + layer norm
            NDList listProjectedAttention = new NDList(projectedAttention);
            attentionResidual = dropout.forward(
                    parameterStore, listProjectedAttention, training
            ).singletonOrThrow();

            NDArray attentionPlusResidual = projectedInput.add(attentionResidual);
            NDList listAttentionPlusResidual = new NDList(attentionPlusResidual);
            attentionOutputNorm = layerNorm1.forward(
                    parameterStore,
                    listAttentionPlusResidual,
                    training
            ).singletonOrThrow();
            // Cerrar temporales ya utilizados
            attentionPlusResidual.close();

            attentionInput.close();
            listAttentionPlusResidual.close();
            listAttentionContext.close();
            listProjectedAttention.close();

            // === 2. Feed-Forward Network ===
            NDList listAttentionOutputNorm = new NDList(attentionOutputNorm);
            NDList ffnOutput = feedForwardBlock.forward(
                    parameterStore, listAttentionOutputNorm, training
            );
            ffnResult = ffnOutput.singletonOrThrow();

            // Dropout + residual connection + layer norm
            NDList listFfnResult = new NDList(ffnResult);
            ffnResidual = dropout.forward(
                    parameterStore, listFfnResult, training
            ).singletonOrThrow();

            NDArray ffnPlusResidual = attentionOutputNorm.add(ffnResidual);
            NDList listFfnPlusResidual = new NDList(ffnPlusResidual);
            finalOutput = layerNorm2.forward(
                    parameterStore,
                    listFfnPlusResidual,
                    training
            ).singletonOrThrow();

            // Cerrar temporales restantes
            if (ffnResult != null && ffnResult != ffnResidual) {
                ffnResult.close();
            }
            if (ffnResidual != null && ffnResidual != finalOutput) {
                ffnResidual.close();
            }

            listAttentionOutputNorm.close();
            listFfnResult.close();
            listFfnPlusResidual.close();
            ffnPlusResidual.close();
            //            System.out.println("SubManager final: " + manager);
            return new NDList(finalOutput);

        } catch (Exception e) {
            // En caso de error, asegurarse de cerrar todo
            closeIfNotNull(currentMask);
            closeIfNotNull(attentionContext);
            closeIfNotNull(projectedAttention);
            closeIfNotNull(attentionResidual);
            closeIfNotNull(attentionOutputNorm);
            closeIfNotNull(ffnResult);
            closeIfNotNull(ffnResidual);
            closeIfNotNull(finalOutput);
            throw e;
        }
    }

    /**
     * Método auxiliar para cerrar NDArray de forma segura
     */
    private void closeIfNotNull(NDArray array) {
        if (array != null) {
            try {
                array.close();
            } catch (Exception e) {
                // Ignorar errores al cerrar
            }
        }
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        // Output shape es (batch_size, sequence_length, modelDim)
        long batchSize = inputShapes[0].get(0);
        long seqLength = inputShapes[0].get(1);
        return new Shape[]{new Shape(batchSize, seqLength, modelDim)};
    }

    // ===================================================================
    // FÁBRICA PARA CREAR BLOQUES CON CONFIGURACIONES PREDEFINIDAS
    // ===================================================================

    public static class Builder {
        private int modelDim = 64;
        private int numHeads = 4;
        private int ffDim = 256;
        private float dropoutRate = 0.1f;
        private int maxSequenceLength = 45;
        private int oftenClearCache = 50;

        public Builder setModelDim(int modelDim) {
            this.modelDim = modelDim;
            return this;
        }

        public Builder setNumHeads(int numHeads) {
            this.numHeads = numHeads;
            return this;
        }

        public Builder setFeedForwardDim(int ffDim) {
            this.ffDim = ffDim;
            return this;
        }

        public Builder setDropoutRate(float dropoutRate) {
            this.dropoutRate = dropoutRate;
            return this;
        }

        public Builder setMaxSequenceLength(int maxSequenceLength) {
            this.maxSequenceLength = maxSequenceLength;
            return this;
        }

        public Builder setOftenClearCache(int oftenClearCache) {
            this.oftenClearCache = oftenClearCache;
            return this;
        }

        public TemporalTransformerBlock build() {
            return new TemporalTransformerBlock(modelDim, numHeads, ffDim, dropoutRate, maxSequenceLength, oftenClearCache);
        }
    }

    public static Builder builder() {
        return new Builder();
    }
}
