package xyz.cereshost.io;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.Main;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.utils.XNormalizer;
import xyz.cereshost.utils.YNormalizer;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.Properties;
import java.util.UUID;

public class IOdata {

    public static final Path MODEL_DIR = Path.of("models");
    public static final String NORMALIZER_DIR = "normalizers";
    public static final int TRAINING_CACHE_MAGIC = 0x54425631;

    public static void saveOut(@NotNull Path path, String json, String name) throws IOException {
        Path file = path.resolve(name + ".json");
        Files.writeString(
                file,
                json,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    public static Path createTrainingCacheDir() throws IOException {
        Path dir = Path.of("data", "cache", UUID.randomUUID().toString());
        Files.createDirectories(dir);
        return dir;
    }

    public static Path saveTrainingCache(Path dir, String symbol, int month, float[][][] X, float[][] y) throws IOException {
        if (X == null || X.length == 0 || y == null || y.length == 0) {
            throw new IllegalArgumentException("Empty training cache");
        }
        Files.createDirectories(dir);
        String fileName = String.format(Locale.ROOT, "%s-%d-%s.bin", symbol, month, UUID.randomUUID());
        Path file = dir.resolve(fileName);
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(
                Files.newOutputStream(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING)))) {
            out.writeInt(TRAINING_CACHE_MAGIC);
            out.writeInt(1);
            int xSamples = X.length;
            int seqLen = X[0].length;
            int features = X[0][0].length;
            int ySamples = y.length;
            int yCols = y[0].length;
            out.writeInt(xSamples);
            out.writeInt(seqLen);
            out.writeInt(features);
            out.writeInt(ySamples);
            out.writeInt(yCols);

            for (float[][] seq : X) {
                for (int j = 0; j < seqLen; j++) {
                    float[] row = seq[j];
                    for (int k = 0; k < features; k++) {
                        out.writeFloat(row[k]);
                    }
                }
            }

            for (float[] row : y) {
                for (int j = 0; j < yCols; j++) {
                    out.writeFloat(row[j]);
                }
            }
        }
        return file;
    }

    public static Pair<float[][][], float[][]> loadTrainingCache(Path file) throws IOException {
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file)))) {
            int magic = in.readInt();
            if (magic != TRAINING_CACHE_MAGIC) {
                throw new IOException("Cache invalida: " + file);
            }
            int version = in.readInt();
            if (version != 1) {
                throw new IOException("Cache version invalida: " + version);
            }
            int xSamples = in.readInt();
            int seqLen = in.readInt();
            int features = in.readInt();
            int ySamples = in.readInt();
            int yCols = in.readInt();

            float[][][] X = new float[xSamples][seqLen][features];
            float[][] y = new float[ySamples][yCols];

            for (int i = 0; i < xSamples; i++) {
                float[][] seq = X[i];
                for (int j = 0; j < seqLen; j++) {
                    float[] row = seq[j];
                    for (int k = 0; k < features; k++) {
                        row[k] = in.readFloat();
                    }
                }
            }

            for (int i = 0; i < ySamples; i++) {
                float[] row = y[i];
                for (int j = 0; j < yCols; j++) {
                    row[j] = in.readFloat();
                }
            }
            return new Pair<>(X, y);
        }
    }

    public static void deleteTrainingCache(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException e) {
            Vesta.info("No se pudo borrar cache temporal: " + file + " - " + e.getMessage());
        }
    }



    static {
        // Crear directorios si no existen
        new File(MODEL_DIR.toUri()).mkdirs();
        new File(NORMALIZER_DIR).mkdirs();
    }

    /**
     * Guardar normalizador X (RobustNormalizer)
     */
    public static void saveXNormalizer(XNormalizer normalizer) throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR);
        String s = Utils.GSON.toJson(normalizer);
        saveOut(normPath, s, "Normalizer_x");
        Vesta.info("✅ Normalizador X guardado en: " + normPath);
    }

    /**
     * Guardar normalizador Y (MultiSymbolNormalizer)
     */
    public static void saveYNormalizer(YNormalizer normalizer) throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR);
        String s = Utils.GSON.toJson(normalizer);
        saveOut(normPath, s, "Normalizer_y");
        Vesta.info("✅ Normalizador Y guardado en: " + normPath);
    }

    /**
     * Cargar normalizador X
     */
    public static XNormalizer loadXNormalizer() throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR, "Normalizer_x.json");
        if (!Files.exists(normPath)) throw new FileNotFoundException("Normalizador X no encontrado: " + normPath);
        return Utils.GSON.fromJson(Files.readString(normPath), XNormalizer.class);
    }

    /**
     * Cargar normalizador Y
     */
    public static YNormalizer loadYNormalizer() throws IOException {
        Path normPath = Paths.get(NORMALIZER_DIR, "Normalizer_y.json");
        if (!Files.exists(normPath)) throw new FileNotFoundException("Normalizador Y no encontrado: " + normPath);
        return Utils.GSON.fromJson(Files.readString(normPath), YNormalizer.class);
    }

    /**
     * Cargar ambos normalizadores
     */
    public static Pair<XNormalizer, YNormalizer> loadNormalizers() throws IOException {
        XNormalizer xNorm = loadXNormalizer();
        YNormalizer yNorm = loadYNormalizer();
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



