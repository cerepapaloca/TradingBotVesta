package xyz.cereshost;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;
import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.List;

@Getter
public class PredictionEngine {

    private final Model model;
    private final RobustNormalizer xNormalizer;
    private final MultiSymbolNormalizer yNormalizer;
    private final int lookBack;
    private final int features;
    private final Device device;

    public PredictionEngine(RobustNormalizer xNormalizer,
                            MultiSymbolNormalizer yNormalizer, int lookBack,
                            int features) throws IOException {
        this.device = Device.gpu();
        this.model = IOdata.loadModel();
        this.xNormalizer = xNormalizer;
        this.yNormalizer = yNormalizer;
        this.lookBack = lookBack;
        this.features = features;
    }

    /**
     * Hacer predicción individual para una secuencia de datos
     */
    public float predict(float[][][] inputSequence) {
        try (NDManager manager = NDManager.newBaseManager(device)) {
            int batchSize = inputSequence.length;
            int sequenceLength = inputSequence[0].length;
            int actualFeatures = inputSequence[0][0].length;

            if (actualFeatures != features) {
                Vesta.waring("Adjusting features from " + features + " to " + actualFeatures);
            }

            // Normalizar la secuencia de entrada
            float[][][] normalizedInput = xNormalizer.transform(inputSequence);

            // Aplanar a 1D
            float[] flatInput = EngineUtils.flatten3DArray(normalizedInput);

            NDArray inputArray = manager.create(flatInput, new Shape(batchSize, sequenceLength, actualFeatures));

            // Realizar predicción
            var block = model.getBlock();
            var parameterStore = new ai.djl.training.ParameterStore(manager, false);

            NDList output = block.forward(parameterStore, new NDList(inputArray), false);
            NDArray prediction = output.singletonOrThrow();

            // Obtener valor float
            float normalizedPrediction = prediction.toFloatArray()[0];

            // Desnormalizar
            float[] denormalized = yNormalizer.inverseTransform(new float[]{normalizedPrediction});

            return denormalized[0];
        }
    }


    /**
     * Predecir el próximo precio basado en las últimas velas
     */
    public float predictNextPrice(Market market) {
        Vesta.MARKETS.put(market.getSymbol(), market);
        Pair<float[][][], float[]> pair = BuilderData.fullBuild(List.of(market.getSymbol()));

        // Solo necesitamos X (la última ventana completa)
        float[][][] Xraw = pair.getKey();
        if (Xraw.length == 0) {
            throw new RuntimeException("No se pudo construir la secuencia de entrada");
        }

        // Tomar la última ventana (que debería ser la única si build() está bien configurado)
        float[][][] window = new float[][][]{Xraw[Xraw.length - 1]};

        // DEBUG: Verificar dimensiones
        Vesta.info("Dimensión de entrada: " + window[0][0].length + " características");

        // Hacer predicción
        return predict(window);
    }

    /**
     * Método mejorado para cargar todo: modelo + normalizadores
     */
    @Contract("_ -> new")
    public static @NotNull PredictionResult loadFullModel(String modelName) throws IOException {
        Device device = Device.gpu();

        // 1. Cargar modelo
        Model model = IOdata.loadModel();

        // 2. Cargar normalizadores
        Pair<RobustNormalizer, MultiSymbolNormalizer> normalizers = IOdata.loadNormalizers();

        // 3. Obtener metadatos (lookback y features)
        int lookBack = VestaEngine.LOOK_BACK;
        int features = 7; // OHLCV + 2 símbolo (ajusta según tu caso)

        Vesta.info("✅ Sistema completo cargado:");
        Vesta.info("  Modelo: " + modelName);
        Vesta.info("  Lookback: " + lookBack);
        Vesta.info("  Features: " + features);

        return new PredictionResult(model, normalizers.getKey(),
                normalizers.getValue(), lookBack, features, device);
    }

    /**
     * Clase para mantener todo junto
     */
    @Getter
    public static class PredictionResult {
        public final Model model;
        public final RobustNormalizer xNormalizer;
        public final MultiSymbolNormalizer yNormalizer;
        public final int lookBack;
        public final int features;
        public final Device device;

        public PredictionResult(Model model, RobustNormalizer xNormalizer,
                                MultiSymbolNormalizer yNormalizer, int lookBack,
                                int features, Device device) {
            this.model = model;
            this.xNormalizer = xNormalizer;
            this.yNormalizer = yNormalizer;
            this.lookBack = lookBack;
            this.features = features;
            this.device = device;
        }

        public void close() {
            if (model != null) {
                model.close();
            }
        }
    }
}