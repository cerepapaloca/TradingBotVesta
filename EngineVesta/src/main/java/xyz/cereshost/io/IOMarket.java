package xyz.cereshost.io;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.github.luben.zstd.ZstdInputStream;
import com.github.luben.zstd.ZstdOutputStream;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.DataSource;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.packet.PacketHandler;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDate;
import java.time.YearMonth;
import java.time.temporal.ChronoUnit;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;

import static xyz.cereshost.io.KlinesSerializable.parseKlineLine;
import static xyz.cereshost.io.TradeSerializable.parseTradeLine;

@UtilityClass
public class IOMarket {

    public static final int TRADE_BIN_MAGIC = 0x54524431;
    public static final int KLINE_BIN_MAGIC = 0x4B4C4E31;
    static final int BIN_VERSION = 1;
    private static final int BATCH_SIZE = 50_000;
    public static final String STORAGE_DIR = "data";
    public static final int BUFFER_READ_MB = 50;
    private static final int ZSTD_LEVEL = 1;
    private static final String EXT_ZIP = ".zip";
    private static final String EXT_BIN = ".bin";
    private static final String EXT_ZST = ".zst";
    private static final String EXT_BIN_ZST = ".bin.zst";
    // Base de referencia para indices diarios (ajustar si cambia el dataset)
    private static final LocalDate BASE_DATE = LocalDate.of(2025, 12, 31);


    public static Market loadMarkets(DataSource data, String s) throws InterruptedException, IOException {
        return loadMarkets(data, s, -1);
    }

    public static Market loadMarkets(DataSource data, String s, int dayIndex) throws InterruptedException, IOException {
        CountDownLatch latch = new CountDownLatch(1);
        AtomicLong lastUpdate = new AtomicLong();
        AtomicReference<Market> marketFinal = new AtomicReference<>(null);
        String baseDir = STORAGE_DIR.toString();
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
                    trades.add(new Trade(id, time,(float) price, (float) quoteQty, isBuyerMaker));
                }

                Market market = new Market(s);
                market.addCandles(deque);
                market.addTrade(trades);
                marketFinal.set(market);
                Vesta.info("✅ Datos procesado de binance del mercado: %s (%.2fss)", s, (float) (System.currentTimeMillis() - timeTotal) / 1000);
                latch.countDown();
            }
            case LOCAL_ZIP -> {
                // Normalizar dayIndex a >= 1
                int normalizedDayIndex = Math.max(1, dayIndex);
                LocalDate targetDate = resolveDateFromDayIndex(normalizedDayIndex);
                int targetYear = targetDate.getYear();
                int targetMonth = targetDate.getMonthValue();
                int targetDay = targetDate.getDayOfMonth();

                long timeTotal = System.currentTimeMillis();
                Vesta.info("%d/%02d/%02d (idx=%d) 💾 Leyendo zst local de klines", targetYear, targetMonth, targetDay, normalizedDayIndex);
                File klineFile = ensureFileCached(baseDir, s, "klines", targetDate);
                Deque<CandleSimple> candles = parseKlinesFromFile(klineFile);
                Vesta.info("%d/%02d/%02d (idx=%d) 💾 Leyendo zst local de trades", targetYear, targetMonth, targetDay, normalizedDayIndex);
                File tradeFile = ensureFileCached(baseDir, s, "trades", targetDate);
                Deque<Trade> trades = parseTradesFromFile(tradeFile);
                if (candles.isEmpty() || trades.isEmpty()) {
                    Vesta.info("Datos incompletos o corruptos para %s en %d/%02d/%02d (idx=%d)", s, targetYear, targetMonth, targetDay, normalizedDayIndex);
                    latch.countDown();
                    return marketFinal.get();
                }
                int sizeCandles = candles.size();
                int sizeTrades = trades.size();
                Vesta.info("%d/%02d/%02d (idx=%d) 🔒 Asegurando orden de los datos", targetYear, targetMonth, targetDay, normalizedDayIndex);
                LinkedHashSet<CandleSimple> candlesSorted = Market.sortInChunks(candles, 10_000, CandleSimple::openTime);
                LinkedHashSet<Trade> tradeSorted = Market.sortInChunks(trades, 10_000, Trade::time);

                // 3. Lógica de CORTE (Sincronización de tiempos)
                long minTimeCandles = candlesSorted.getFirst().openTime();
                long maxTimeCandles = candlesSorted.getLast().openTime();

                long minTimeTrades = tradeSorted.getFirst().time();
                long maxTimeTrades = tradeSorted.getLast().time();

                long commonStart = Math.max(minTimeCandles, minTimeTrades);
                long commonEnd = Math.min(maxTimeCandles, maxTimeTrades);

                Vesta.info("%d/%02d/%02d (idx=%d) ✂️ Ajustando %s a ventana comun: %d - %d", targetYear, targetMonth, targetDay, normalizedDayIndex, s, commonStart, commonEnd);
                // borrar por inicio
                while (!candlesSorted.isEmpty() && candlesSorted.getFirst().openTime() < commonStart) {
                    candlesSorted.removeFirst();
                }
                // borrar por final
                while (!candlesSorted.isEmpty() && candlesSorted.getLast().openTime() > commonEnd) {
                    candlesSorted.removeLast();
                }

                // mismo para trades
                while (!tradeSorted.isEmpty() && tradeSorted.getFirst().time() < commonStart) {
                    tradeSorted.removeFirst();
                }
                while (!tradeSorted.isEmpty() && tradeSorted.getLast().time() > commonEnd) {
                    tradeSorted.removeLast();
                }

                final Market market = new Market(s);
                market.setCandles(candlesSorted);
                market.setTrade(tradeSorted);
                //market.sortd();

                marketFinal.set(market);
                lastUpdate.set(System.currentTimeMillis());

                Vesta.info("%d/%02d/%02d (idx=%d) 🟩 Mercado cargado desde DISCO: %s (C: %d, T: %d) en %.2fs",
                        targetYear, targetMonth, targetDay, normalizedDayIndex, s, sizeCandles, sizeTrades, (float) (System.currentTimeMillis() - timeTotal) / 1000);

                latch.countDown();
            }
        }
        latch.await();
        return marketFinal.get();
    }

    public static LocalDate resolveDateFromDayIndex(int dayIndex) {
        int normalized = Math.max(1, dayIndex);
        return BASE_DATE.minusDays(normalized - 1L);
    }

    public static int resolveDayIndex(LocalDate date) {
        long days = ChronoUnit.DAYS.between(date, BASE_DATE);
        return (int) days + 1;
    }

    public static YearMonth resolveMonthFromIndex(int monthIndex) {
        int normalized = Math.max(1, monthIndex);
        return YearMonth.from(BASE_DATE).minusMonths(normalized - 1L);
    }


    private static File ensureFileCached(String baseDir, String symbol, String type, LocalDate date) throws IOException {
        String monthStr = String.format("%02d", date.getMonthValue());
        String dayStr = String.format("%02d", date.getDayOfMonth());
        String dateStr = String.format("%d-%s-%s", date.getYear(), monthStr, dayStr);
        // Nombre del archivo segun convencion de Binance (daily)
        String baseName = type.equals("klines")
                ? String.format("%s-1m-%s", symbol, dateStr)
                : String.format("%s-trades-%s", symbol, dateStr);
        String fileNameZip = baseName + EXT_ZIP;
        String fileNameBin = baseName + EXT_BIN;
        String fileNameBinZst = baseName + EXT_BIN_ZST;
        // Estructura: ./data/ETHUSDT/klines/2025-12/ETHUSDT-1m-2025-12-01.bin.zst
        File dir = new File(baseDir + File.separator + symbol + File.separator + type + File.separator + date.getYear() + "-" + monthStr);
        if (!dir.exists()) {
            dir.mkdirs(); // Crea la estructura de carpetas si no existe
        }

        File targetFileZip = new File(dir, fileNameZip);
        File targetFileBin = new File(dir, fileNameBin);
        File targetFileBinZst = new File(dir, fileNameBinZst);

        if (targetFileBin.exists() && targetFileBin.length() > 0) {
            return targetFileBin;
        } else if (targetFileBinZst.exists() && targetFileBinZst.length() > 0) {
            return targetFileBinZst;
        } else if (targetFileZip.exists() && targetFileZip.length() > 0) {
            convertZipToBinZst(targetFileZip, targetFileBinZst, type);
            safeDelete(targetFileZip);
            return targetFileBinZst;
        }

        // Construir URL de descarga
        String urlTypePath = type.equals("klines") ? "klines" : "trades"; // URL path segment
        String urlInterval = type.equals("klines") ? "/1m" : ""; // Trades no tienen intervalo en la URL

        String urlString = String.format("https://data.binance.vision/data/futures/um/daily/%s/%s%s/%s",
                urlTypePath, symbol, urlInterval, fileNameZip);

        Vesta.info("📥 Descargando nuevo archivo: " + fileNameZip + " (" + urlString + ")");

        try (InputStream in = new URL(urlString).openStream()) {
            Files.copy(in, targetFileZip.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            // Si falla la descarga, borramos el archivo vacio/corrupto para no romper ejecuciones futuras
            if (targetFileZip.exists()) targetFileZip.delete();
            throw new IOException("Fallo al descargar " + urlString, e);
        }
        Vesta.info("✅ Descargando completada: " + fileNameZip);
        convertZipToBinZst(targetFileZip, targetFileBinZst, type);
        safeDelete(targetFileZip);

        return targetFileBinZst;
    }

    /**
     * Cambia el nombre de referencia al binario comprimido.
     * @param dataFile el archivo base (zip/bin/bin.zst)
     * @return el nombre del binario (bin o bin.zst)
     */
    public static String binEntryName(File dataFile) {
        String name = dataFile.getName();
        if (name.endsWith(EXT_BIN_ZST)) {
            return name;
        }
        if (name.endsWith(EXT_ZIP)) {
            name = name.substring(0, name.length() - EXT_ZIP.length());
        }
        if (name.endsWith(EXT_ZST)) {
            name = name.substring(0, name.length() - EXT_ZST.length());
        }
        if (name.endsWith(EXT_BIN)) {
            return name;
        }
        return name + EXT_BIN_ZST;
    }

    /**
     * Transforma la referencia de un archivo base a binario.
     * @param dataFile el archivo base
     * @return el archivo con la extension .bin o .bin.zst
     */
    private static File binFileForZip(File dataFile) {
        File parent = dataFile.getParentFile();
        String name = binEntryName(dataFile);
        return parent == null ? new File(name) : new File(parent, name);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    private static Deque<CandleSimple> parseKlinesFromFile(File file) {
        Deque<CandleSimple> list = new ArrayDeque<>();
        KlinesSerializable parse = new KlinesSerializable();

        Deque<CandleSimple> cached1 = parseFromBin(binFileForZip(file), parse);
        if (cached1 != null) {
            return cached1;
        }
        if (!file.exists()) return list;
        if (file.getName().endsWith(EXT_ZIP)) {
            Deque<CandleSimple> cached2 = parseFromBinInZip(file, parse);
            if (cached2 != null) {
                return cached2;
            }
        }

        return parseFromCSVandWriteBin(file, parse, parse);
    }

    private static Deque<Trade> parseTradesFromFile(File file) {
        Deque<Trade> list = new ArrayDeque<>();
        TradeSerializable parseCSV = new TradeSerializable();

        Deque<Trade> cached1 = parseFromBin(binFileForZip(file), parseCSV);
        if (cached1 != null) {
            return cached1;
        }
        if (!file.exists()) return list;
        // En el caso de que .bin exista dentro del .zip (legacy)
        if (file.getName().endsWith(EXT_ZIP)) {
            Deque<Trade> cached2 = parseFromBinInZip(file, parseCSV);
            if (cached2 != null) {
                return cached2;
            }
        }

        return parseFromCSVandWriteBin(file, parseCSV, parseCSV);
    }

    private static <T> @NotNull Deque<T> parseFromCSVandWriteBin(File file, SerializableCSV<T> SerializableCSV, SerializableBin<T> SerializableBin) {
        Deque<T> list = new ArrayDeque<>();
        if (!file.exists()) return list;
        if (!file.getName().endsWith(EXT_ZIP)) return list;


        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
        int maxInFlight = threads * 2;
        int batchSize = BATCH_SIZE;
        Deque<FutureTask<List<T>>> tasks = new ArrayDeque<>(maxInFlight);

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(file), (1 << 20) * BUFFER_READ_MB))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), (1 << 20) * BUFFER_READ_MB);
                    String line;
                    List<String> batch = new ArrayList<>(batchSize);
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header
                        batch.add(line);
                        if (batch.size() >= batchSize) {
                            SerializableCSV.submitBatch(tasks, batch);
                            batch = new ArrayList<>(batchSize);
                            if (tasks.size() >= maxInFlight) {
                                drainTask(tasks, list);
                            }
                        }
                    }
                    if (!batch.isEmpty()) {
                        SerializableCSV.submitBatch(tasks, batch);
                    }
                }
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo csv locales: " + e.getMessage());
        }
        try {
            while (!tasks.isEmpty()) {
                drainTask(tasks, list);
            }
        } catch (Exception e) {
            Vesta.info("Error procesando datos locales: " + e.getMessage());
        }
        boolean wrote = false;
        if (!list.isEmpty()) {
            try {
                SerializableBin.writeBin(file, list);
                wrote = true;
            } catch (Exception e) {
                Vesta.info("Error guardando binario: " + e.getMessage());
            }
        }
        if (wrote) {
            safeDelete(file);
        }
        return list;
    }

    public interface SerializableBin<T> {
        void writeBin(File zipFile, Deque<T> source) throws IOException;
        Deque<T> readBin(DataInputStream in) throws IOException;
    }

    public interface SerializableCSV<T> {
        void submitBatch(Deque<FutureTask<List<T>>> tasks, List<String> batch);
    }

    public <T> void drainTask(Deque<FutureTask<List<T>>> tasks, Deque<T> list) {
        try {
            FutureTask<List<T>> task = tasks.removeFirst();
            List<T> trades = task.get();
            if (trades != null && !trades.isEmpty()) {
                list.addAll(trades);
            }
        } catch (InterruptedException | ExecutionException e) {
            throw new RuntimeException(e);
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////



    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    private static <T> Deque<T> parseFromBin(File binFile, SerializableBin<T> parseMethod) {
        File candidate = binFile;
        if (candidate.getName().endsWith(EXT_BIN_ZST)) {
            File raw = new File(candidate.getParentFile(),
                    candidate.getName().substring(0, candidate.getName().length() - EXT_ZST.length()));
            if (raw.exists() && raw.length() > 0) {
                candidate = raw;
            }
        }
        if (!candidate.exists() || candidate.length() == 0) {
            return null;
        }
        try (DataInputStream in = openBinInputStream(candidate)) {
            return parseMethod.readBin(in);
        } catch (Exception e) {
            Vesta.info("Error leyendo binario: " + e.getMessage());
            return null;
        }
    }

    private static DataInputStream openBinInputStream(File binFile) throws IOException {
        InputStream in = new BufferedInputStream(new FileInputStream(binFile), (1 << 20) * BUFFER_READ_MB);
        if (binFile.getName().endsWith(EXT_ZST)) {
            in = new ZstdInputStream(in);
        }
        return new DataInputStream(in);
    }

    private static DataOutputStream openBinOutputStream(File binFile, boolean useZstd) throws IOException {
        OutputStream out = new BufferedOutputStream(new FileOutputStream(binFile), (1 << 20) * BUFFER_READ_MB);
        if (useZstd) {
            ZstdOutputStream zOut = new ZstdOutputStream(out);
            zOut.setLevel(ZSTD_LEVEL);
            zOut.setWorkers(Math.min(8, Runtime.getRuntime().availableProcessors()));
            out = zOut;
        }
        return new DataOutputStream(out);
    }

    private static <T> Deque<T> parseFromBinInZip(File zipFile, SerializableBin<T> parse) {
        String entryName = binEntryName(zipFile);
        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry entry = zip.getEntry(entryName);
            if (entry == null && entryName.endsWith(EXT_BIN_ZST)) {
                String legacyEntry = entryName.substring(0, entryName.length() - EXT_ZST.length());
                entry = zip.getEntry(legacyEntry);
            }
            if (entry == null) {
                return null;
            }
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(zip.getInputStream(entry), (1 << 20) * BUFFER_READ_MB))) {
                return parse.readBin(in);
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Trades binario: " + e.getMessage());
            return null;
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    @FunctionalInterface
    public interface BinWriter {
        void write(DataOutputStream out) throws IOException;
    }

    public static void rewriteZipWithBinEntry(File zipFile, String entryName, BinWriter writer) throws IOException {
        File parent = zipFile.getParentFile();
        File binFile = parent == null ? new File(entryName) : new File(parent, entryName);
        Path temp = Files.createTempFile(parent == null ? Paths.get(".") : parent.toPath(), binFile.getName(), ".tmp");
        try {
            try (DataOutputStream out = openBinOutputStream(temp.toFile(), binFile.getName().endsWith(EXT_ZST))) {
                writer.write(out);
                out.flush();
            }
            Files.move(temp, binFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    private static void copyStream(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[(1 << 20) * BUFFER_READ_MB];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    private static void convertZipToBinZst(File zipFile, File binZstFile, String type) throws IOException {
        if ("trades".equals(type)) {
            buildTradeBinFromZip(zipFile, binZstFile);
        } else if ("klines".equals(type)) {
            buildKlineBinFromZip(zipFile, binZstFile);
        } else {
            throw new IOException("Tipo no soportado: " + type);
        }
    }

    private static void safeDelete(File file) {
        if (file == null) return;
        try {
            Files.deleteIfExists(file.toPath());
        } catch (IOException ignored) {
            // no-op
        }
    }

    private static void buildTradeBinFromZip(File zipFile, File binFile) throws IOException {
        if (binFile.getParentFile() != null) {
            Files.createDirectories(binFile.getParentFile().toPath());
        }
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), (1 << 20) * BUFFER_READ_MB));
             DataOutputStream out = openBinOutputStream(binFile, binFile.getName().endsWith(EXT_ZST))) {
            out.writeInt(TRADE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), (1 << 20) * BUFFER_READ_MB);
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header
                        Trade trade = parseTradeLine(line);
                        if (trade == null) continue;
                        out.writeLong(trade.id());
                        out.writeLong(trade.time());
                        out.writeDouble(trade.price());
                        out.writeDouble(trade.qty());
                        out.writeBoolean(trade.isBuyerMaker());
                    }
                }
            }
        }
    }

    private static void buildKlineBinFromZip(File zipFile, File binFile) throws IOException {
        if (binFile.getParentFile() != null) {
            Files.createDirectories(binFile.getParentFile().toPath());
        }
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), (1 << 20) * BUFFER_READ_MB));
             DataOutputStream out = openBinOutputStream(binFile, binFile.getName().endsWith(EXT_ZST))) {
            out.writeInt(KLINE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), (1 << 20) * BUFFER_READ_MB);
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header
                        CandleSimple candle = parseKlineLine(line);
                        Volumen vol = candle.volumen();
                        out.writeLong(candle.openTime());
                        out.writeDouble(candle.open());
                        out.writeDouble(candle.high());
                        out.writeDouble(candle.low());
                        out.writeDouble(candle.close());
                        out.writeDouble(vol.quoteVolume());
                        out.writeDouble(vol.baseVolume());
                        out.writeDouble(vol.takerBuyQuoteVolume());
                        out.writeDouble(vol.sellQuoteVolume());
                        out.writeDouble(vol.deltaUSDT());
                        out.writeDouble(vol.buyRatio());
                    }
                }
            }
        }
    }


    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    public static void extractFirstBin(Path folder) throws IOException {
        Vesta.info("Comenzando extraccion en " + folder);
        if (!Files.isDirectory(folder)) {
            throw new IllegalArgumentException("No es una carpeta");
        }
        boolean extracted = false;
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(folder, "*" + EXT_BIN_ZST)) {
            for (Path binPath : stream) {
                extracted = true;
                Vesta.info("Extrayendo de " + binPath.toString());
                extractBinFromZst(binPath);
            }
        }
        if (!extracted) {
            try (DirectoryStream<Path> stream = Files.newDirectoryStream(folder, "*" + EXT_ZIP)) {
                for (Path zipPath : stream) {
                    Vesta.info("Extrayendo de " + zipPath.toString());
                    extractBinFromZip(zipPath);
                }
            }
        }
    }

    private static void extractBinFromZst(Path binZstPath) throws IOException {
        String name = binZstPath.getFileName().toString();
        String outName = name.endsWith(EXT_ZST) ? name.substring(0, name.length() - EXT_ZST.length()) : name;
        Path outputPath = binZstPath.getParent().resolve(outName);
        try (InputStream in = new ZstdInputStream(new BufferedInputStream(Files.newInputStream(binZstPath), (1 << 20) * BUFFER_READ_MB));
             OutputStream out = new BufferedOutputStream(Files.newOutputStream(
                     outputPath,
                     StandardOpenOption.CREATE,
                     StandardOpenOption.TRUNCATE_EXISTING), (1 << 20) * BUFFER_READ_MB)) {
            copyStream(in, out);
        }
        System.out.println("Extraido: " + outputPath);
    }

    private static void extractBinFromZip(Path zipPath) throws IOException {
        try (ZipFile zipFile = new ZipFile(zipPath.toFile())) {
            Enumeration<? extends ZipEntry> entries = zipFile.entries();

            while (entries.hasMoreElements()) {
                ZipEntry entry = entries.nextElement();

                if (!entry.isDirectory() && entry.getName().endsWith(".bin")) {

                    Path outputPath = zipPath.getParent()
                            .resolve(Paths.get(entry.getName()).getFileName());

                    try (InputStream in = zipFile.getInputStream(entry);
                         OutputStream out = Files.newOutputStream(
                                 outputPath,
                                 StandardOpenOption.CREATE,
                                 StandardOpenOption.TRUNCATE_EXISTING)) {

                        in.transferTo(out);
                    }

                    // Solo el primer .bin
                    System.out.println("Extraido: " + outputPath);
                    break;
                }
            }
        }
    }
}


