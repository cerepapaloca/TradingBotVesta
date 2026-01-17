package xyz.cereshost.file;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.Main;
import xyz.cereshost.VestaEngine;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.packet.PacketHandler;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;

public class IOdata {

    public static final Path MODEL_DIR = Path.of("models");
    public static final String NORMALIZER_DIR = "normalizers";

    public static void saveOut(@NotNull Path path, String json, String name) throws IOException {
        Path file = path.resolve(name + ".json");
        Files.writeString(
                file,
                json,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    public static void loadMarkets(List<String> symbols) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(symbols.size());
        for (String s : symbols){
            Vesta.info("📡 Enviado solicitud de datos del mercado: " + s);
            PacketHandler.sendPacket(new RequestMarketClient(s), MarketDataServer.class).thenAccept(packet -> {
                Vesta.MARKETS.put(s, packet.getMarket());
                latch.countDown();
                Vesta.info("✅ Datos del mercado " + s + " recibidos (" + (symbols.size() - latch.getCount()) + "/" + symbols.size() + ")");
            });
        }
        latch.await();
    }

    public static void loadMarkets(String... symbols) throws InterruptedException {
        loadMarkets(Arrays.asList(symbols));
    }

    static {
        // Crear directorios si no existen
        new File(MODEL_DIR.toUri()).mkdirs();
        new File(NORMALIZER_DIR).mkdirs();
    }

    /**
     * Guardar normalizador X (RobustNormalizer)
     */
    public static void saveXNormalizer(RobustNormalizer normalizer) throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR);
        String s = Utils.GSON.toJson(normalizer);
        saveOut(normPath, s, "Normalizer_x");
        Vesta.info("✅ Normalizador X guardado en: " + normPath);
    }

    /**
     * Guardar normalizador Y (MultiSymbolNormalizer)
     */
    public static void saveYNormalizer(MultiSymbolNormalizer normalizer) throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR);
        String s = Utils.GSON.toJson(normalizer);
        saveOut(normPath, s, "Normalizer_y");
        Vesta.info("✅ Normalizador Y guardado en: " + normPath);
    }

    /**
     * Cargar normalizador X
     */
    public static RobustNormalizer loadXNormalizer() throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR, "Normalizer_x.json");
        if (!Files.exists(normPath)) throw new FileNotFoundException("Normalizador X no encontrado: " + normPath);
        return Utils.GSON.fromJson(Files.readString(normPath), RobustNormalizer.class);
    }

    /**
     * Cargar normalizador Y
     */
    public static MultiSymbolNormalizer loadYNormalizer() throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR, "Normalizer_y.json");
        if (!Files.exists(normPath)) throw new FileNotFoundException("Normalizador Y no encontrado: " + normPath);
        return Utils.GSON.fromJson(Files.readString(normPath), MultiSymbolNormalizer.class);
    }

    /**
     * Cargar ambos normalizadores
     */
    public static Pair<RobustNormalizer, MultiSymbolNormalizer> loadNormalizers() throws IOException {
        RobustNormalizer xNorm = loadXNormalizer();
        MultiSymbolNormalizer yNorm = loadYNormalizer();
        return new Pair<>(xNorm, yNorm);
    }
    /**
     * Cargar modelo simple (backward compatibility)
     */
    public static Model loadModel() throws IOException {
        Device device = Device.gpu();
        Path modelDir = MODEL_DIR.toAbsolutePath();

        Vesta.info("📂 Cargando modelo desde: " + modelDir);

        if (!Files.exists(modelDir)) {
            throw new FileNotFoundException("Directorio del modelo no encontrado: " + modelDir);
        }

        try {
            // Crear instancia del modelo
            Model model = Model.newInstance(Main.NAME_MODEL, device, "PyTorch");

            // Asignar la arquitectura (IMPORTANTE)
            model.setBlock(VestaEngine.getSequentialBlock());

            // Cargar parámetros
            model.load(modelDir, Main.NAME_MODEL);

            Vesta.info("✅ Modelo cargado exitosamente");
            Vesta.info("  Parámetros cargados: " + model.getBlock().getParameters().size());

            return model;

        } catch (Exception e) {
            throw new IOException("Error cargando modelo: " + e.getMessage(), e);
        }
    }

    /**
     * Guardar modelo
     */
    public static void saveModel(Model model) throws IOException {
        String modelName = model.getName();
        Path modelDir = MODEL_DIR.resolve(modelName);

        // Crear directorio
        Files.createDirectories(modelDir);

        // Guardar modelo
        model.save(modelDir, modelName);

        // Guardar propiedades
        saveModelProperties(modelDir, model);

        Vesta.info("✅ Modelo guardado en: " + modelDir);
    }

    /**
     * Guardar propiedades del modelo
     */
    private static void saveModelProperties(Path modelDir, Model model) throws IOException {
        Properties props = new Properties();
        props.setProperty("model.name", model.getName());
        props.setProperty("engine", "PyTorch");
        props.setProperty("lookback", String.valueOf(VestaEngine.LOOK_BACK));
        props.setProperty("timestamp", String.valueOf(System.currentTimeMillis()));

        Path propsPath = modelDir.resolve("model.properties");
        try (OutputStream os = Files.newOutputStream(propsPath)) {
            props.store(os, "Model properties");
        }
    }


}
