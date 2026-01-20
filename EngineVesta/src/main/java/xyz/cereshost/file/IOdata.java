package xyz.cereshost.file;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.util.Pair;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.Main;
import xyz.cereshost.TypeData;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Volumen;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.packet.PacketHandler;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicLong;

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

    public static long loadMarkets(boolean forTraining, List<String> symbols) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(symbols.size());
        AtomicLong lastUpdate = new AtomicLong();
        for (String s : symbols){
            if (Main.TYPE_DATA == TypeData.ALL){
                Vesta.info("📡 Enviado solicitud de datos del mercado: " + s);
                PacketHandler.sendPacket(new RequestMarketClient(s, forTraining), MarketDataServer.class).thenAccept(packet -> {
                    Vesta.MARKETS.put(s, packet.getMarket());
                    latch.countDown();
                    lastUpdate.set(packet.getLastUpdate());
                    Vesta.info("✅ Datos del mercado " + s + " recibidos (" + (symbols.size() - latch.getCount()) + "/" + symbols.size() + ")");
                });
            }else {
                // dos semanas en minutos 20160
                String url = Utils.BASE_URL_API + "klines" + "?symbol=" + s + "&interval=1m&limit=" + "20160";
                long time = System.currentTimeMillis();
                Vesta.info("📡 Solicitud de dato a binance del mercado: " + s + " (" + url + ")");
                String raw = Utils.getRequest(url);
                ObjectMapper mapper = new ObjectMapper();
                JsonNode root;
                try {
                    root = mapper.readTree(raw);
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
                ArrayDeque<CandleSimple> deque = new ArrayDeque<>();
                Vesta.info("📂 Datos recibidos de binance del mercado: " + s + " (" + raw.getBytes(StandardCharsets.UTF_8).length / 1024 + "mb)" );
                for (int i = 0; i < 1000; i++) {
                    JsonNode kline = root.get(i);

                    double baseVolume = kline.get(5).asDouble();
                    double quoteVolume = kline.get(7).asDouble();  // USDT
                    double takerBuyQuoteVolume = kline.get(10).asDouble(); // USDT agresivo

                    double sellQuoteVolume = quoteVolume - takerBuyQuoteVolume;
                    double deltaUSDT = takerBuyQuoteVolume - sellQuoteVolume;
                    double buyRatio = takerBuyQuoteVolume / quoteVolume;
                    deque.add(new CandleSimple(
                            kline.get(0).asLong(),
                            kline.get(1).asDouble(), // open
                            kline.get(2).asDouble(), // high
                            kline.get(3).asDouble(), // low
                            kline.get(4).asDouble(), // close
                            new Volumen(quoteVolume, baseVolume, takerBuyQuoteVolume, sellQuoteVolume, deltaUSDT, buyRatio)));
                }
                Market market = new Market(s);
                market.addCandles(deque);
                Vesta.MARKETS.put(s, market);
                Vesta.info("✅ Datos procesado de binance del mercado: %s ( %.2f s)", s, (float) (System.currentTimeMillis() - time)/1000);
                latch.countDown();
            }
        }
        latch.await();
        return lastUpdate.get();
    }


    public static long loadMarkets(boolean forTraining, String... symbols) throws InterruptedException {
        return loadMarkets(forTraining, Arrays.asList(symbols));
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
