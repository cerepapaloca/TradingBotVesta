package xyz.cereshost.io;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
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
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

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


    public static Market loadMarkets(DataSource data, String s) throws InterruptedException, IOException {
        return loadMarkets(data, s, -1);
    }

    public static Market loadMarkets(DataSource data, String s, int monthCVS) throws InterruptedException, IOException {
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
                // Normalizar monthCVS a >= 1
                int monthIndex = Math.max(1, monthCVS);
                // offset: 0 -> most recent (Dec 2025), 1 -> one month back (Nov 2025), ...
                int offset = monthIndex - 1;

                int targetYear = 2025 - (offset / 12);
                int targetMonth = 12 - (offset % 12);

                long timeTotal = System.currentTimeMillis();
                Vesta.info("%d/%02d (idx=%d) 💾 Leyendo zip local de klines", targetYear, targetMonth, monthIndex);
                File klineFile = ensureFileCached(baseDir, s, "klines", targetYear, targetMonth);
                Deque<CandleSimple> candles = parseKlinesFromFile(klineFile);
                Vesta.info("%d/%02d (idx=%d) 💾 Leyendo zip local de trades", targetYear, targetMonth, monthIndex);
                File tradeFile = ensureFileCached(baseDir, s, "trades", targetYear, targetMonth);
                Deque<Trade> trades = parseTradesFromFile(tradeFile);
                if (candles.isEmpty() || trades.isEmpty()) {
                    Vesta.info("⚠️ Datos incompletos o corruptos para %s en %d/%02d (idx=%d)", s, targetYear, targetMonth, monthIndex);
                    latch.countDown();
                    return marketFinal.get();
                }
                int sizeCandles = candles.size();
                int sizeTrades = trades.size();
                Vesta.info("%d/%02d (idx=%d) 🔒 Asegurando orden de los datos", targetYear, targetMonth, monthIndex);
                LinkedHashSet<CandleSimple> candlesSorted = Market.sortInChunks(candles, 10_000, CandleSimple::openTime);
                LinkedHashSet<Trade> tradeSorted = Market.sortInChunks(trades, 10_000, Trade::time);

                // 3. Lógica de CORTE (Sincronización de tiempos)
                long minTimeCandles = candlesSorted.getFirst().openTime();
                long maxTimeCandles = candlesSorted.getLast().openTime();

                long minTimeTrades = tradeSorted.getFirst().time();
                long maxTimeTrades = tradeSorted.getLast().time();

                long commonStart = Math.max(minTimeCandles, minTimeTrades);
                long commonEnd = Math.min(maxTimeCandles, maxTimeTrades);

                Vesta.info("%d/%02d (idx=%d) ✂️ Ajustando %s a ventana común: %d - %d", targetYear, targetMonth, monthIndex, s, commonStart, commonEnd);
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

                Vesta.info("%d/%02d (idx=%d) ✅ Mercado cargado desde DISCO: %s (C: %d, T: %d) en %.2fs",
                        targetYear, targetMonth, monthIndex, s, sizeCandles, sizeTrades, (float) (System.currentTimeMillis() - timeTotal) / 1000);

                latch.countDown();
            }
        }
        latch.await();
        return marketFinal.get();
    }


    private static File ensureFileCached(String baseDir, String symbol, String type, int year, int month) throws IOException {
        String monthStr = String.format("%02d", month);
        // Nombre del archivo según convención de Binance
        String baseName = String.format("%s-%s-%d-%s", symbol, (type.equals("klines") ? "1m" : "trades"), year, monthStr);
        String fileNameZip = baseName + ".zip";
        String fileNameBin = baseName + ".bin";
        // Estructura: ./data/ETHUSDT/klines/ETHUSDT-1m-2025-12.zip
        File dir = new File(baseDir + File.separator + symbol + File.separator + type);
        if (!dir.exists()) {
            dir.mkdirs(); // Crea la estructura de carpetas si no existe
        }

        File targetFileZip = new File(dir, fileNameZip);
        File targetFileBin = new File(dir, fileNameBin);

        if (targetFileZip.exists() && targetFileZip.length() > 0) {
            // Vesta.info("   -> Usando caché local: " + targetFile.getPath());
            return targetFileZip;
        }else if (targetFileBin.exists() && targetFileBin.length() > 0) {
            return targetFileBin;
        }

        // Construir URL de descarga
        String urlTypePath = type.equals("klines") ? "klines" : "trades"; // URL path segment
        String urlInterval = type.equals("klines") ? "/1m" : ""; // Trades no tienen intervalo en la URL

        String urlString = String.format("https://data.binance.vision/data/futures/um/monthly/%s/%s%s/%s",
                urlTypePath, symbol, urlInterval, fileNameZip);

        Vesta.info("⬇️ Descargando nuevo archivo: " + fileNameZip + " (" + urlString + ")");

        try (InputStream in = new URL(urlString).openStream()) {
            Files.copy(in, targetFileZip.toPath(), StandardCopyOption.REPLACE_EXISTING);
        } catch (IOException e) {
            // Si falla la descarga, borramos el archivo vacío/corrupto para no romper ejecuciones futuras
            if(targetFileZip.exists()) targetFileZip.delete();
            throw new IOException("Fallo al descargar " + urlString, e);
        }

        try {
            ensureBinCache(targetFileZip, type);
        } catch (Exception e) {
            Vesta.info("Error creando cache binario: " + e.getMessage());
        }

        return targetFileZip;
    }

    /**
     * Cambiar el nombre del archivo que se hace referencia
     * @param zipFile el archivo a cambiar
     * @return el nombre cambiado
     */

    public static String binEntryName(File zipFile) {
        String name = zipFile.getName();
        if (name.endsWith(".zip")) {
            name = name.substring(0, name.length() - 4);
        }
        if (name.endsWith(".bin")) {
            return name;
        }else {
            return name + ".bin";
        }
    }

    /**
     * Transforma la referencia de un archivo .zip a .bin
     * @param zipFile el a cambiar
     * @return el archivo con la extension .bin
     */

    private static File binFileForZip(File zipFile) {
        File parent = zipFile.getParentFile();
        String name = binEntryName(zipFile);
        return parent == null ? new File(name) : new File(parent, name);
    }

    private static boolean hasZipEntry(File zipFile, String entryName) throws IOException {
        try (ZipFile zip = new ZipFile(zipFile)) {
            return zip.getEntry(entryName) != null;
        }
    }

    private static void ensureBinCache(File zipFile, String type) throws IOException {
        String entryName = binEntryName(zipFile);
        if (hasZipEntry(zipFile, entryName)) {
            return;
        }
        Path temp = Files.createTempFile(zipFile.getParentFile().toPath(), entryName, ".tmp");
        File binTemp = temp.toFile();
        try {
            if ("trades".equals(type)) {
                buildTradeBinFromZip(zipFile, binTemp);
            } else if ("klines".equals(type)) {
                buildKlineBinFromZip(zipFile, binTemp);
            }
            rewriteZipWithBinEntry(zipFile, entryName, out -> {
                try (InputStream in = new BufferedInputStream(new FileInputStream(binTemp), (1 << 20) * BUFFER_READ_MB)) {
                    copyStream(in, out);
                }
            });
        } finally {
            if (binTemp.exists()) {
                binTemp.delete();
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////////////////////////////////

    private synchronized static Deque<CandleSimple> parseKlinesFromFile(File file) {
        Deque<CandleSimple> list = new ArrayDeque<>();
        if (!file.exists()) return list;
        KlinesSerializable parse = new KlinesSerializable();

        Deque<CandleSimple> cached1 = parseFromBin(binFileForZip(file), parse);
        if (cached1 != null) {
            return cached1;
        }
        Deque<CandleSimple> cached2 = parseFromBinInZip(file, parse);
        if (cached2 != null) {
            return cached2;
        }

        return parseFromCSVandWriteBin(file, parse, parse);
    }

    private synchronized static Deque<Trade> parseTradesFromFile(File file) {
        Deque<Trade> list = new ArrayDeque<>();
        if (!file.exists()) return list;
        TradeSerializable parseCSV = new TradeSerializable();

        Deque<Trade> cached1 = parseFromBin(binFileForZip(file), parseCSV);
        if (cached1 != null) {
            return cached1;
        }
        // En el caso de que .bin exista dentro del .zip
        Deque<Trade> cached2 = parseFromBinInZip(file, parseCSV);
        if (cached2 != null) {
            return cached2;
        }

        return parseFromCSVandWriteBin(file, parseCSV, parseCSV);
    }

    private static <T> @NotNull Deque<T> parseFromCSVandWriteBin(File file, SerializableCSV<T> SerializableCSV, SerializableBin<T> SerializableBin) {
        Deque<T> list = new ArrayDeque<>();
        if (!file.exists()) return list;


        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
        int maxInFlight = threads * 2;
        int batchSize = BATCH_SIZE;
        Deque<FutureTask<List<T>>> tasks = new ArrayDeque<>(maxInFlight);

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(file), (1 << 20) * BUFFER_READ_MB))) { // CAMBIO: FileInputStream
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
            Vesta.info("Error procesando Klines locales: " + e.getMessage());
        }
        if (!list.isEmpty()) {
            try {
                SerializableBin.writeBin(file, list);
            } catch (Exception e) {
                Vesta.info("Error guardando Klines binario: " + e.getMessage());
            }
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
        if (!binFile.exists() || binFile.length() == 0) {
            return null;
        }
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(new FileInputStream(binFile), (1 << 20) * BUFFER_READ_MB))) {
            return parseMethod.readBin(in);
        } catch (Exception e) {
            Vesta.info("Error leyendo binario: " + e.getMessage());
            return null;
        }
    }

    private static <T> Deque<T> parseFromBinInZip(File zipFile, SerializableBin<T> parse) {
        String entryName = binEntryName(zipFile);
        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry entry = zip.getEntry(entryName);
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
        Path temp = Files.createTempFile(zipFile.getParentFile().toPath(), zipFile.getName(), ".tmp");
        try {
            try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), (1 << 20) * BUFFER_READ_MB));
                 ZipOutputStream zos = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(temp), (1 << 20) * BUFFER_READ_MB))) {
                ZipEntry entry;
                byte[] buffer = new byte[(1 << 20) * BUFFER_READ_MB];
                while ((entry = zis.getNextEntry()) != null) {
                    if (entry.isDirectory()) {
                        continue;
                    }
                    if (entryName.equals(entry.getName())) {
                        continue;
                    }
                    ZipEntry outEntry = new ZipEntry(entry.getName());
                    zos.putNextEntry(outEntry);
                    int read;
                    while ((read = zis.read(buffer)) != -1) {
                        zos.write(buffer, 0, read);
                    }
                    zos.closeEntry();
                }
                ZipEntry binEntry = new ZipEntry(entryName);
                zos.putNextEntry(binEntry);
                DataOutputStream out = new DataOutputStream(zos);
                writer.write(out);
                out.flush();
                zos.closeEntry();
            }
            Files.move(temp, zipFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
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

    private static void buildTradeBinFromZip(File zipFile, File binFile) throws IOException {
        if (binFile.getParentFile() != null) {
            Files.createDirectories(binFile.getParentFile().toPath());
        }
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), (1 << 20) * BUFFER_READ_MB));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(binFile), (1 << 20) * BUFFER_READ_MB))) {
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
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(binFile), (1 << 20) * BUFFER_READ_MB))) {
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
        Vesta.info("ℹ️ Comenzado extracción en " + folder);
        if (!Files.isDirectory(folder)) {
            throw new IllegalArgumentException("No es una carpeta");
        }
        try (DirectoryStream<Path> stream = Files.newDirectoryStream(folder, "*.zip")) {
            for (Path zipPath : stream) {
                Vesta.info("📦 Extrayendo de " + zipPath.toString());
                extractBinFromZip(zipPath);
            }
        }
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
                    System.out.println("Extraído: " + outputPath);
                    break;
                }
            }
        }
    }
}
