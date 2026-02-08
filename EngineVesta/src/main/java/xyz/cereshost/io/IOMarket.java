package xyz.cereshost.io;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.experimental.UtilityClass;
import xyz.cereshost.DataSource;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.common.market.Volumen;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.packet.PacketHandler;

import java.io.*;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

@UtilityClass
public class IOMarket {

    private static final int TRADE_BIN_MAGIC = 0x54524431;
    private static final int KLINE_BIN_MAGIC = 0x4B4C4E31;
    private static final int BIN_VERSION = 1;
    private static final int BATCH_SIZE = 10_000;


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
                Vesta.info("%d/%02d (idx=%d) 💾 Verificando caché local para: %s", targetYear, targetMonth, monthIndex, s);
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

        try {
            ensureBinCache(targetFile, type);
        } catch (Exception e) {
            Vesta.info("Error creando cache binario: " + e.getMessage());
        }

        return targetFile;
    }

    private static String binEntryName(File zipFile) {
        String name = zipFile.getName();
        if (name.endsWith(".zip")) {
            name = name.substring(0, name.length() - 4);
        }
        return name + ".bin";
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
                try (InputStream in = new BufferedInputStream(new FileInputStream(binTemp), 1 << 20)) {
                    copyStream(in, out);
                }
            });
        } finally {
            if (binTemp.exists()) {
                binTemp.delete();
            }
        }
    }

    ///////////////////
    ///////////////////

    private static List<CandleSimple> parseKlinesFromFile(File file) {
        List<CandleSimple> list = new ArrayList<>();
        if (!file.exists()) return list;

        List<CandleSimple> cached = parseKlinesFromBin(file);
        if (cached != null) {
            return cached;
        }

        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
        int maxInFlight = threads * 2;
        int batchSize = BATCH_SIZE;
        Deque<FutureTask<List<CandleSimple>>> tasks = new ArrayDeque<>(maxInFlight);

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(file), 1 << 20))) { // CAMBIO: FileInputStream
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), 1 << 20);
                    String line;
                    List<String> batch = new ArrayList<>(batchSize);
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header
                        batch.add(line);
                        if (batch.size() >= batchSize) {
                            submitKlineBatch(tasks, batch);
                            batch = new ArrayList<>(batchSize);
                            if (tasks.size() >= maxInFlight) {
                                drainKlineTask(tasks, list);
                            }
                        }
                    }
                    if (!batch.isEmpty()) {
                        submitKlineBatch(tasks, batch);
                    }
                }
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Klines locales: " + e.getMessage());
        }
        try {
            while (!tasks.isEmpty()) {
                drainKlineTask(tasks, list);
            }
        } catch (Exception e) {
            Vesta.info("Error procesando Klines locales: " + e.getMessage());
        }
        if (!list.isEmpty()) {
            try {
                writeKlineBinFromList(file, list);
            } catch (Exception e) {
                Vesta.info("Error guardando Klines binario: " + e.getMessage());
            }
        }
        return list;
    }

    private static List<Trade> parseTradesFromFile(File file) {
        List<Trade> list = new ArrayList<>();
        if (!file.exists()) return list;

        List<Trade> cached = parseTradesFromBin(file);
        if (cached != null) {
            return cached;
        }

        int threads = Math.max(1, Runtime.getRuntime().availableProcessors());
        int maxInFlight = threads * 2;
        int batchSize = BATCH_SIZE;
        Deque<FutureTask<List<Trade>>> tasks = new ArrayDeque<>(maxInFlight);

        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(file), 1 << 20))) {
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), 1 << 20);
                    String line;
                    List<String> batch = new ArrayList<>(batchSize);
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header
                        batch.add(line);
                        if (batch.size() >= batchSize) {
                            submitTradeBatch(tasks, batch);
                            batch = new ArrayList<>(batchSize);
                            if (tasks.size() >= maxInFlight) {
                                drainTradeTask(tasks, list);
                            }
                        }
                    }
                    if (!batch.isEmpty()) {
                        submitTradeBatch(tasks, batch);
                    }
                }
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Trades locales: " + e.getMessage());
        }
        try {
            while (!tasks.isEmpty()) {
                drainTradeTask(tasks, list);
            }
        } catch (Exception e) {
            Vesta.info("Error procesando Trades locales: " + e.getMessage());
        }
        if (!list.isEmpty()) {
            try {
                writeTradeBinFromList(file, list);
            } catch (Exception e) {
                Vesta.info("Error guardando Trades binario: " + e.getMessage());
            }
        }
        return list;
    }

    ///////////////////
    ///////////////////

    private static void submitTradeBatch(Deque<FutureTask<List<Trade>>> tasks, List<String> batch) {
        List<String> batchCopy = new ArrayList<>(batch);
        FutureTask<List<Trade>> task = new FutureTask<>(() -> parseTradeBatch(batchCopy));
        tasks.addLast(task);
        VestaEngine.EXECUTOR_READ_SCV.execute(task);
    }

    private static void drainTradeTask(Deque<FutureTask<List<Trade>>> tasks, List<Trade> list)
            throws InterruptedException, ExecutionException {
        FutureTask<List<Trade>> task = tasks.removeFirst();
        List<Trade> trades = task.get();
        if (trades != null && !trades.isEmpty()) {
            list.addAll(trades);
        }
    }

    private static List<Trade> parseTradeBatch(List<String> lines) {
        List<Trade> trades = new ArrayList<>(lines.size());
        for (String line : lines) {
            Trade trade = parseTradeLine(line);
            if (trade != null) {
                trades.add(trade);
            }
        }
        return trades;
    }

    private static Trade parseTradeLine(String line) {
        int p0 = line.indexOf(',');
        int p1 = line.indexOf(',', p0 + 1);
        int p2 = line.indexOf(',', p1 + 1);
        int p3 = line.indexOf(',', p2 + 1);
        int p4 = line.indexOf(',', p3 + 1);
        int p5 = line.indexOf(',', p4 + 1);
        if (p0 <= 0 || p1 <= 0 || p2 <= 0 || p3 <= 0 || p4 <= 0) {
            return null;
        }
        if (p5 == -1) p5 = line.length(); // Por si es la ultima columna

        return new Trade(
                Long.parseLong(line.substring(0, p0)),             // id (col 0)
                Long.parseLong(line.substring(p3 + 1, p4)),        // time (col 4)
                Double.parseDouble(line.substring(p0 + 1, p1)),     // price (col 1)
                Double.parseDouble(line.substring(p1 + 1, p2)),     // qty (col 2)
                "true".equals(line.substring(p4 + 1, p5)) // isBuyerMaker (col 5)
        );
    }

    ///////////////////
    ///////////////////

    private static List<Trade> parseTradesFromBin(File zipFile) {
        String entryName = binEntryName(zipFile);
        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry entry = zip.getEntry(entryName);
            if (entry == null) {
                return null;
            }
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(zip.getInputStream(entry), 1 << 20))) {
                int magic = in.readInt();
                if (magic != TRADE_BIN_MAGIC) {
                    return null;
                }
                int version = in.readInt();
                if (version != BIN_VERSION) {
                    return null;
                }
                List<Trade> list = new ArrayList<>();
                while (true) {
                    try {
                        long id = in.readLong();
                        long time = in.readLong();
                        double price = in.readDouble();
                        double qty = in.readDouble();
                        boolean isBuyerMaker = in.readBoolean();
                        list.add(new Trade(id, time, price, qty, isBuyerMaker));
                    } catch (EOFException eof) {
                        break;
                    }
                }
                return list;
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Trades binario: " + e.getMessage());
            return null;
        }
    }

    private static List<CandleSimple> parseKlinesFromBin(File zipFile) {
        String entryName = binEntryName(zipFile);
        try (ZipFile zip = new ZipFile(zipFile)) {
            ZipEntry entry = zip.getEntry(entryName);
            if (entry == null) {
                return null;
            }
            try (DataInputStream in = new DataInputStream(new BufferedInputStream(zip.getInputStream(entry), 1 << 20))) {
                int magic = in.readInt();
                if (magic != KLINE_BIN_MAGIC) {
                    return null;
                }
                int version = in.readInt();
                if (version != BIN_VERSION) {
                    return null;
                }
                List<CandleSimple> list = new ArrayList<>();
                while (true) {
                    try {
                        long openTime = in.readLong();
                        double open = in.readDouble();
                        double high = in.readDouble();
                        double low = in.readDouble();
                        double close = in.readDouble();
                        double quoteVolume = in.readDouble();
                        double baseVolume = in.readDouble();
                        double takerBuyQuoteVolume = in.readDouble();
                        double sellQuoteVolume = in.readDouble();
                        double deltaUSDT = in.readDouble();
                        double buyRatio = in.readDouble();
                        list.add(new CandleSimple(
                                openTime,
                                open,
                                high,
                                low,
                                close,
                                new Volumen(quoteVolume, baseVolume, takerBuyQuoteVolume, sellQuoteVolume, deltaUSDT, buyRatio)
                        ));
                    } catch (EOFException eof) {
                        break;
                    }
                }
                return list;
            }
        } catch (Exception e) {
            Vesta.info("Error leyendo Klines binario: " + e.getMessage());
            return null;
        }
    }

    ///////////////////
    ///////////////////

    private static void writeTradeBinFromList(File zipFile, List<Trade> trades) throws IOException {
        String entryName = binEntryName(zipFile);
        rewriteZipWithBinEntry(zipFile, entryName, out -> {
            out.writeInt(TRADE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            for (Trade trade : trades) {
                out.writeLong(trade.id());
                out.writeLong(trade.time());
                out.writeDouble(trade.price());
                out.writeDouble(trade.qty());
                out.writeBoolean(trade.isBuyerMaker());
            }
        });
    }

    private static void writeKlineBinFromList(File zipFile, List<CandleSimple> candles) throws IOException {
        String entryName = binEntryName(zipFile);
        rewriteZipWithBinEntry(zipFile, entryName, out -> {
            out.writeInt(KLINE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            for (CandleSimple candle : candles) {
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
        });
    }

    @FunctionalInterface
    private interface BinWriter {
        void write(DataOutputStream out) throws IOException;
    }

    private static void rewriteZipWithBinEntry(File zipFile, String entryName, BinWriter writer) throws IOException {
        Path temp = Files.createTempFile(zipFile.getParentFile().toPath(), zipFile.getName(), ".tmp");
        try {
            try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), 1 << 20));
                 ZipOutputStream zos = new ZipOutputStream(new BufferedOutputStream(Files.newOutputStream(temp), 1 << 20))) {
                ZipEntry entry;
                byte[] buffer = new byte[1 << 20];
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
        byte[] buffer = new byte[1 << 20];
        int read;
        while ((read = in.read(buffer)) != -1) {
            out.write(buffer, 0, read);
        }
    }

    private static void buildTradeBinFromZip(File zipFile, File binFile) throws IOException {
        if (binFile.getParentFile() != null) {
            Files.createDirectories(binFile.getParentFile().toPath());
        }
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), 1 << 20));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(binFile), 1 << 20))) {
            out.writeInt(TRADE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), 1 << 20);
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
        try (ZipInputStream zis = new ZipInputStream(new BufferedInputStream(new FileInputStream(zipFile), 1 << 20));
             DataOutputStream out = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(binFile), 1 << 20))) {
            out.writeInt(KLINE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            ZipEntry entry;
            while ((entry = zis.getNextEntry()) != null) {
                if (!entry.isDirectory() && entry.getName().endsWith(".csv")) {
                    BufferedReader br = new BufferedReader(new InputStreamReader(zis), 1 << 20);
                    String line;
                    while ((line = br.readLine()) != null) {
                        if (line.isEmpty() || Character.isLetter(line.charAt(0))) continue; // Skip Header
                        CandleSimple candle = parseKlineLine(line);
                        if (candle == null) continue;
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


    private static void submitKlineBatch(Deque<FutureTask<List<CandleSimple>>> tasks, List<String> batch) {
        List<String> batchCopy = new ArrayList<>(batch);
        FutureTask<List<CandleSimple>> task = new FutureTask<>(() -> parseKlineBatch(batchCopy));
        tasks.addLast(task);
        VestaEngine.EXECUTOR_READ_SCV.execute(task);
    }

    private static void drainKlineTask(Deque<FutureTask<List<CandleSimple>>> tasks, List<CandleSimple> list)
            throws InterruptedException, ExecutionException {
        FutureTask<List<CandleSimple>> task = tasks.removeFirst();
        List<CandleSimple> candles = task.get();
        if (candles != null && !candles.isEmpty()) {
            list.addAll(candles);
        }
    }

    private static List<CandleSimple> parseKlineBatch(List<String> lines) {
        List<CandleSimple> candles = new ArrayList<>(lines.size());
        for (String line : lines) {
            CandleSimple candle = parseKlineLine(line);
            if (candle != null) {
                candles.add(candle);
            }
        }
        return candles;
    }

    private static CandleSimple parseKlineLine(String line) {
        String[] p = line.split(",");
        double quoteVolume = Double.parseDouble(p[7]);
        double takerBuyQuoteVolume = Double.parseDouble(p[10]);

        return new CandleSimple(
                Long.parseLong(p[0]), // Open time
                Double.parseDouble(p[1]), // Open
                Double.parseDouble(p[2]), // High
                Double.parseDouble(p[3]), // Low
                Double.parseDouble(p[4]), // Close
                new Volumen(quoteVolume, Double.parseDouble(p[5]), takerBuyQuoteVolume,
                        quoteVolume - takerBuyQuoteVolume,
                        takerBuyQuoteVolume - (quoteVolume - takerBuyQuoteVolume),
                        (quoteVolume == 0) ? 0 : takerBuyQuoteVolume / quoteVolume)
        );
    }
}
