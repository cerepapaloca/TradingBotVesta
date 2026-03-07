package xyz.cereshost.vesta.core.io;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.util.Pair;
import com.google.gson.JsonIOException;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.Main;
import xyz.cereshost.vesta.common.packet.Utils;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.engine.VestaEngine;
import xyz.cereshost.vesta.core.utils.BuilderData;
import xyz.cereshost.vesta.core.utils.TrainingData;
import xyz.cereshost.vesta.core.utils.XNormalizer;
import xyz.cereshost.vesta.core.utils.YNormalizer;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.zip.Deflater;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import static xyz.cereshost.vesta.core.io.IOMarket.BUFFER_READ_MB;

public class IOdata {

    public static final Path MODEL_DIR = Path.of("models");
    public static final String NORMALIZER_DIR = "normalizers";
    public static final String CACHE_DIR = "E:\\data";
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
        Path dir = new CacheProperties(VestaEngine.LOOK_BACK, BuilderData.FEATURES, 5, Main.SYMBOLS_TRAINING, Main.MAX_MONTH_TRAINING, -1).getPath();
        Files.createDirectories(dir);
        return dir;
    }

    public static Path saveTrainingCache(Path dir, String symbol, int month, float[][][] X, float[][] y, boolean useZip) throws IOException {
        if (X == null || X.length == 0 || y == null || y.length == 0) {
            throw new IllegalArgumentException("Empty training cache");
        }
        Files.createDirectories(dir);
        String extension = useZip ? "zip" : "bin";
        String fileName = String.format(Locale.ROOT, "%s-%d-%s.%s", symbol, month, UUID.randomUUID(), extension);
        Path file = dir.resolve(fileName);
        if (useZip) {
            try (ZipOutputStream zipOut = new ZipOutputStream(new BufferedOutputStream(
                    Files.newOutputStream(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING),
                    (1 << 20) * BUFFER_READ_MB
            ))) {
            zipOut.setMethod(ZipOutputStream.DEFLATED);
            zipOut.setLevel(Deflater.BEST_SPEED);
            zipOut.putNextEntry(new ZipEntry("cache.bin"));
            DataOutputStream out = new DataOutputStream(zipOut);
            writeTrainingCache(X, y, out);
            out.flush();
            Vesta.info("(idx:%d) 📀 Resultado guardado en: %s", month, dir);
            zipOut.closeEntry();
            }
        } else {
            try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(
                    Files.newOutputStream(file, StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING),
                    (1 << 20) * BUFFER_READ_MB
            ))) {
                writeTrainingCache(X, y, out);
                out.flush();
                Vesta.info("(idx:%d) 📀 Resultado guardado en: %s", month, dir);
            }
        }
        return file;
    }

    private static void writeTrainingCache(float[][][] X, float[][] y, DataOutputStream out) throws IOException {
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

    public static Pair<float[][][], float[][]> loadTrainingCache(Path file) throws IOException {
        String name = file.getFileName().toString().toLowerCase(Locale.ROOT);

        if (name.endsWith(".zip")) {
            try (ZipInputStream zipIn = new ZipInputStream(new BufferedInputStream(Files.newInputStream(file), (1 << 20) * BUFFER_READ_MB))) {
                ZipEntry entry = zipIn.getNextEntry();

                if (entry == null) {
                    throw new IOException("Cache zip vacia: " + file);
                }

                DataInputStream in = new DataInputStream(zipIn);
                Pair<float[][][], float[][]> pair = readTrainingCache(in, file);
                zipIn.closeEntry();
                return pair;
            }
        }

        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(file), (1 << 20) * BUFFER_READ_MB))) {
            return readTrainingCache(in, file);
        }
    }

    private static Pair<float[][][], float[][]> readTrainingCache(DataInputStream in, Path file) throws IOException {
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
    public static Pair<XNormalizer, YNormalizer> loadNormalizers() throws IOException, JsonIOException {
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
            VestaEngine.setRootManager(model.getNDManager());
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

    public static void saveCacheProperties(CacheProperties cacheProperties) throws IOException {
        Path cache = cacheProperties.getPath();
        String s = Utils.GSON.toJson(cacheProperties);
        saveOut(cache, s, "cacheProperties");
    }

    @Nullable
    public static CacheProperties loadCacheProperties() throws IOException {
        Path cacheDir = createTrainingCacheDir();
        if (!Files.exists(cacheDir)) return null;
        Path dir = cacheDir.resolve("cacheProperties.json");
        if (!Files.exists(dir)) return null;
        CacheProperties cache = Utils.GSON.fromJson(Files.readString(dir), CacheProperties.class);
        if (cache.symbol.isEmpty()){
            return null;
        }else {
            return cache;
        }
    }

    public record CacheProperties(int lookback, int features, int outputs, List<String> symbol, int monthData, int sizeData) {

        @NotNull
        public UUID getUUID() {
            return UUID.nameUUIDFromBytes(String.format("L: %d F: %d O: %d S: %s M: %d", lookback, features, outputs, symbol, monthData).getBytes(StandardCharsets.UTF_8));
        }

        public Path getPath() {
            return Paths.get(CACHE_DIR, "cache", this.getUUID().toString());
        }
    }

    private static Path getDir() throws IOException {
        try (var stream = Files.list(Paths.get(CACHE_DIR, "cache"))) {
            return stream
                    .filter(Files::isDirectory)
                    .max(Comparator.comparingLong(p -> {
                        try {
                            return Files.getLastModifiedTime(p).toMillis();
                        } catch (IOException e) {
                            return Long.MIN_VALUE;
                        }
                    })).orElse(null);
        }
    }

    public static boolean isBuiltData() throws IOException {
        return loadCacheProperties() != null;
    }

    public static TrainingData getBuiltData() throws IOException {
        CacheProperties cacheProperties = Objects.requireNonNull(loadCacheProperties());
        List<Path> cacheFiles = new ArrayList<>();
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(Path.of(CACHE_DIR, "cache", cacheProperties.getUUID().toString()))) {
            for (Path path : stream) {
                String name = path.getFileName().toString().toLowerCase();
                if (name.endsWith(".zip") || name.endsWith(".bin")) cacheFiles.add(path);
            }
        }
        TrainingData trainingData = new TrainingData(cacheFiles, cacheProperties.sizeData, cacheProperties.lookback, cacheProperties.features, cacheProperties.outputs);
        try {
            Pair<XNormalizer, YNormalizer> normalizer = loadNormalizers();
            trainingData.setXNormalizer(normalizer.getKey());
            trainingData.setYNormalizer(normalizer.getValue());
        }catch (JsonIOException | FileNotFoundException ignore){
            trainingData.prepareNormalize();
        }
        return trainingData;
    }

    public record ApiKeysBinance(String key, String secret){}

    public static ApiKeysBinance loadApiKeysBinance() throws IOException {
        Path path = Paths.get("apiKeys.properties");
        if (Files.exists(path)){
            Properties properties = new Properties();
            properties.load(Files.newInputStream(path));
            return new ApiKeysBinance(properties.getProperty("apiKey"), properties.getProperty("secret"));
        }else {
            Properties props = new Properties();
            props.setProperty("apiKey", "");
            props.setProperty("secret", "");

            try (OutputStream os = Files.newOutputStream(path)) {
                props.store(os, "apiKeys");
            }
            return new ApiKeysBinance("", "");
        }
    }

    public record DiscordConfig(String token, String userRaw){
        public List<String> users(){
            return Arrays.asList(userRaw.split(","));
        }
    }

    public static DiscordConfig loadDiscordConfig() throws IOException {
        Path path = Paths.get("discord.properties");
        if (Files.exists(path)){
            Properties properties = new Properties();
            properties.load(Files.newInputStream(path));
            return new DiscordConfig(properties.getProperty("token"), properties.getProperty("users"));
        }else {
            Properties props = new Properties();
            props.setProperty("token", "");
            props.setProperty("users", "");

            try (OutputStream os = Files.newOutputStream(path)) {
                props.store(os, "configuración de discord");
            }
            return new DiscordConfig("", "");
        }
    }
}



