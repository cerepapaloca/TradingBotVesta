package xyz.cereshost.file;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.util.Pair;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.jetbrains.annotations.NotNull;
import tech.tablesaw.io.csv.CsvReader;
import xyz.cereshost.Main;
import xyz.cereshost.DataSource;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.builder.YNormalizer;
import xyz.cereshost.builder.XNormalizer;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Volumen;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.packet.PacketHandler;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

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

    public static Market loadMarkets(DataSource data, String s) throws InterruptedException, IOException {
        return loadMarkets(data, s, -1);
    }

    public static Market loadMarkets(DataSource data, String s, int monthCVS) throws InterruptedException, IOException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicLong lastUpdate = new AtomicLong();
        AtomicReference<Market> marketFinal = new AtomicReference<>(null);
        String baseDir = "data";
        switch (data) {
            case LOCAL_NETWORK, LOCAL_NETWORK_MINIMAL -> {

                Vesta.info("📡 Enviado solicitud de datos del mercado: " + s);
                PacketHandler.sendPacket(new RequestMarketClient(s, data == DataSource.LOCAL_NETWORK), MarketDataServer.class).thenAccept(packet -> {
                    marketFinal.set(packet.getMarket());
                    latch.countDown();
                    lastUpdate.set(packet.getLastUpdate());
                    Vesta.info("✅ Datos del mercado " + s + " recibidos de "+  s);
                });
            }
            case BINANCE -> {
                long timeTotal = System.currentTimeMillis();
                Vesta.info("📡 Solicitud de dato a binance del mercado: " + s);
                String raw = Utils.getRequest(Utils.BASE_URL_API + "klines" + "?symbol=" + s + "&interval=1m&limit=" + "1000");
                ObjectMapper mapper1 = new ObjectMapper();
                JsonNode root1;
                try {
                    root1 = mapper1.readTree(raw);
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
                ArrayDeque<CandleSimple> deque = new ArrayDeque<>();
                Vesta.info("📂 Datos recibidos de binance del mercado: " + s + " (" + raw.getBytes(StandardCharsets.UTF_8).length / 1024 + "mb)");
                for (int i = 0; i < 1000; i++) {
                    JsonNode kline = root1.get(i);

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
                ObjectMapper mapper2 = new ObjectMapper();
                JsonNode root2;
                try {
                    root2 = mapper2.readTree(Utils.getRequest(Utils.BASE_URL_API + "trades" + "?symbol=" + s + "&limit=" + 800));
                } catch (JsonProcessingException e) {
                    throw new RuntimeException(e);
                }
                Deque<Trade> trades = new ArrayDeque<>();
                for (JsonNode trade : root2) {
                    double quoteQty = trade.get("quoteQty").asDouble();
                    double price = trade.get("price").asDouble();
                    boolean isBuyerMaker = trade.get("isBuyerMaker").asBoolean();
                    long id = trade.get("id").asLong();
                    long time = trade.get("time").asLong();
                    trades.add(new Trade(id, time, price, quoteQty, isBuyerMaker));
                }

                Market market = new Market(s);
                market.addCandles(deque);
                market.addTrade(trades);
                marketFinal.set(market);
                Vesta.info("✅ Datos procesado de binance del mercado: %s (%.2fss)", s, (float) (System.currentTimeMillis() - timeTotal) / 1000);
                latch.countDown();
            }
            case CSV -> {
                // Normalizar monthCVS a >= 1
                int monthIndex = Math.max(1, monthCVS);
                // offset: 0 -> most recent (Dec 2025), 1 -> one month back (Nov 2025), ...
                int offset = monthIndex - 1;

                int targetYear = 2025 - (offset / 12);
                int targetMonth = 12 - (offset % 12);

                long timeTotal = System.currentTimeMillis();
                Vesta.info("%d/%02d (idx=%d) 📂 Verificando caché local para: %s", targetYear, targetMonth, monthIndex, s);
                Deque<CandleSimple> candles = new ArrayDeque<>();
                Deque<Trade> trades = new ArrayDeque<>();
                File klineFile = ensureFileCached(baseDir, s, "klines", targetYear, targetMonth);
                candles.addAll(parseKlinesFromFile(klineFile));
                File tradeFile = ensureFileCached(baseDir, s, "trades", targetYear, targetMonth);
                trades.addAll(parseTradesFromFile(tradeFile));

                if (candles.isEmpty() || trades.isEmpty()) {
                    Vesta.info("⚠️ Datos incompletos o corruptos para %s en %d/%02d (idx=%d)", s, targetYear, targetMonth, monthIndex);
                    latch.countDown();
                    return marketFinal.get();
                }

                // 3. Lógica de CORTE (Sincronización de tiempos)
                long minTimeCandles = candles.getFirst().openTime();
                long maxTimeCandles = candles.getLast().openTime();

                long minTimeTrades = trades.getFirst().time();
                long maxTimeTrades = trades.getLast().time();

                long commonStart = Math.max(minTimeCandles, minTimeTrades);
                long commonEnd = Math.min(maxTimeCandles, maxTimeTrades);

                Vesta.info("%d/%02d (idx=%d) ✂️ Ajustando %s a ventana común: %d - %d", targetYear, targetMonth, monthIndex, s, commonStart, commonEnd);

                Deque<CandleSimple> finalCandles = candles.stream()
                        .filter(c -> c.openTime() >= commonStart && c.openTime() <= commonEnd)
                        .collect(Collectors.toCollection(ArrayDeque::new));
                candles.clear();
                Deque<Trade> finalTrades = trades.stream()
                        .filter(t -> t.time() >= commonStart && t.time() <= commonEnd)
                        .collect(Collectors.toCollection(ArrayDeque::new));
                trades.clear(); // Esto puede pesar más 20GB de RAM
                System.gc();
                final Market market = new Market(s);
                market.addCandles(finalCandles);
                market.addTrade(finalTrades);
                Vesta.info("%d/%02d (idx=%d) 🔒 Asegurando orden de los datos", targetYear, targetMonth, monthIndex);
                market.sortd();

                marketFinal.set(market);
                lastUpdate.set(System.currentTimeMillis());

                Vesta.info("%d/%02d (idx=%d) ✅ Mercado cargado desde DISCO: %s (C: %d, T: %d) en %.2fs",
                        targetYear, targetMonth, monthIndex, s, finalCandles.size(), finalTrades.size(), (float) (System.currentTimeMillis() - timeTotal) / 1000);

                latch.countDown();
            }
        }
        latch.await();
        return marketFinal.get();
    }


    private static File ensureFileCached(String baseDir, String symbol, String type, int year, int month) throws IOException {
        String monthStr = String.format("%02d", month);
        // Nombre del archivo según convención de Binance
        String fileName = String.format("%s-%s-%d-%s.zip", symbol, (type.equals("klines") ? "1m" : "trades"), year, monthStr);

        // Estructura: ./data/ETHUSDT/klines/ETHUSDT-1m-2025-12.zip
        File dir = new File(baseDir + File.separator + symbol + File.separator + type);
        if (!dir.exists()) {
            dir.mkdirs(); // Crea la estructura de carpetas si no existe
        }

        File targetFile = new File(dir, fileName);

        if (targetFile.exists() && targetFile.length() > 0) {
            // Vesta.info("   -> Usando caché local: " + targetFile.getPath());
            return targetFile;
        }

        // Construir URL de descarga
        String urlTypePath = type.equals("klines") ? "klines" : "trades"; // URL path segment
        String urlInterval = type.equals("klines") ? "/1m" : ""; // Trades no tienen intervalo en la URL

        String urlString = String.format("https://data.binance.vision/data/futures/um/monthly/%s/%s%s/%s",
                urlTypePath, symbol, urlInterval, fileName);

        Vesta.info("⬇️ Descargando nuevo archivo: " + fileName + " (" + urlString + ")");

        try (InputStream in = new URL(urlString).openStream()) {
            Files.copy(in, targetFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            // Si falla la descarga, borramos el archivo vacío/corrupto para no romper ejecuciones futuras
            if(targetFile.exists()) targetFile.delete();
            throw new IOException("Fallo al descargar " + urlString, e);
        }

        return targetFile;
    }

    private static List<CandleSimple> parseKlinesFromFile(File file) {
        List<CandleSimple> list = new ArrayList<>();
        if (!file.exists()) return list;

        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(file))) { // CAMBIO: FileInputStream
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis));
                    String line;
                    while ((line = br.readLine()) != null) {
                        if(Character.isLetter(line.charAt(0))) continue; // Skip Header

                        String[] p = line.split(",");
                        // Parsing idéntico al anterior...
                        double quoteVolume = Double.parseDouble(p[7]);
                        double takerBuyQuoteVolume = Double.parseDouble(p[10]);

                        list.add(new CandleSimple(
                                Long.parseLong(p[0]), // Open time
                                Double.parseDouble(p[1]), // Open
                                Double.parseDouble(p[2]), // High
                                Double.parseDouble(p[3]), // Low
                                Double.parseDouble(p[4]), // Close
                                new Volumen(quoteVolume, Double.parseDouble(p[5]), takerBuyQuoteVolume,
                                        quoteVolume - takerBuyQuoteVolume,
                                        takerBuyQuoteVolume - (quoteVolume - takerBuyQuoteVolume),
                                        (quoteVolume == 0) ? 0 : takerBuyQuoteVolume / quoteVolume)
                        ));
                    }
                }
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Klines locales: " + e.getMessage());
        }
        return list;
    }

    private static List<Trade> parseTradesFromFile(File file) {
        List<Trade> list = new ArrayList<>();
        //if (!file.exists()) return list;

        try (ZipInputStream zis = new ZipInputStream(new FileInputStream(file))) { // CAMBIO: FileInputStream
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis));
                    String line;
                    while ((line = br.readLine()) != null) {
                        if(line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header

                        int p0 = line.indexOf(',');
                        int p1 = line.indexOf(',', p0 + 1);
                        int p2 = line.indexOf(',', p1 + 1);
                        int p3 = line.indexOf(',', p2 + 1);
                        int p4 = line.indexOf(',', p3 + 1);
                        int p5 = line.indexOf(',', p4 + 1);
                        if (p5 == -1) p5 = line.length(); // Por si es la última columna

                        list.add(new Trade(
                                Long.parseLong(line.substring(0, p0)),             // id (col 0)
                                Long.parseLong(line.substring(p3 + 1, p4)),        // time (col 4)
                                Double.parseDouble(line.substring(p0 + 1, p1)),     // price (col 1)
                                Double.parseDouble(line.substring(p1 + 1, p2)),     // qty (col 2)
                                "true".equals(line.substring(p4 + 1, p5)) // isBuyerMaker (col 5)
                        ));
                        CsvReader csvReader = new CsvReader();
                    }
                }
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Trades locales: " + e.getMessage());
        }
        return list;
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
