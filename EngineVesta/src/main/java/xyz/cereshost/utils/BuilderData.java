package xyz.cereshost.utils;

import ai.djl.util.Pair;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.ta4j.core.*;
import org.ta4j.core.indicators.ATRIndicator;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.bollinger.*;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.supertrend.SuperTrendIndicator;
import org.ta4j.core.indicators.supertrend.SuperTrendLowerBandIndicator;
import org.ta4j.core.indicators.supertrend.SuperTrendUpperBandIndicator;
import org.ta4j.core.indicators.volume.NVIIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;
import xyz.cereshost.FinancialCalculation;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.io.IOMarket;
import xyz.cereshost.io.IOdata;

import java.io.IOException;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.YearMonth;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import static xyz.cereshost.engine.VestaEngine.LOOK_BACK;

public class BuilderData {

    public static final int DEFAULT_FUTURE_WINDOW = 30;

    public static @NotNull TrainingData buildTrainingData(@NotNull List<String> symbols, int maxMonth, int offset) {
        List<PairCache> cacheEntries = new ArrayList<>();
        long time = System.currentTimeMillis();
        List<CompletableFuture<Object>> waitingCacheSave = new ArrayList<>();
        int futureWindow = DEFAULT_FUTURE_WINDOW;
        for (String symbol : symbols) {
            try {
                List<Candle> allCandlesForChart = new ArrayList<>();

                Vesta.info("Procesando símbolo (Relativo): " + symbol);


                // Procesar cada mes por separado SIN acumular
                List<Integer> months = IntStream.rangeClosed(1, maxMonth)
                        .boxed()
                        .toList();

                // Crear lista de futuros para procesar cada mes de forma asincrónica
                AtomicInteger totalDays = new AtomicInteger();;
                AtomicInteger doneDays = new AtomicInteger();
                List<CompletableFuture<MonthMarketCache>> futures = new ArrayList<>();
                for (int month = maxMonth + offset; month > offset; month--) {
                    final int currentMonth = month;
                    YearMonth targetMonth = IOMarket.resolveMonthFromIndex(currentMonth);
                    int daysInMonth = targetMonth.lengthOfMonth();
                    totalDays.addAndGet(daysInMonth);
                    CompletableFuture<MonthMarketCache> future = CompletableFuture.supplyAsync(() -> {
                        try {
                            System.gc();
                            Vesta.info("(idx:%d) Comenzado carga de datos", currentMonth);

                            List<Candle> candlesThisMonth = new ArrayList<>(daysInMonth * 1500);
                            for (int day = 1; day <= daysInMonth; day++) {
                                LocalDate date = targetMonth.atDay(day);
                                int dayIndex = IOMarket.resolveDayIndex(date);
                                Vesta.info("(idx:%d) ⬆️ Cargando dia %s", currentMonth, date);
                                Market market = IOMarket.loadMarkets(Main.DATA_SOURCE_FOR_TRAINING_MODEL, symbol, dayIndex);
                                if (market == null) {
                                    Vesta.warning("(idx:%d) Mercado vacio para %s", currentMonth, date);
                                    continue;
                                }
                                Vesta.info("(idx:%d) 📊 Convirtiendo velas (dia %s, C:%d)", currentMonth, date, market.getCandleSimples().size());
                                List<Candle> candlesDay = BuilderData.to1mCandles(market, futureWindow);
                                market.clear();
                                if (!candlesDay.isEmpty()) {
                                    candlesThisMonth.addAll(candlesDay);
                                }
                                doneDays.getAndIncrement();
                                Vesta.info("(idx:%d) (%d/%d) (%d/%d) ✅ Dia cargado", currentMonth, day, daysInMonth, doneDays.get(), totalDays.get());
                            }
                            if (candlesThisMonth.size() <= LOOK_BACK + 2) {
                                Vesta.warning("(idx:%d) insuficiente historial: " + candlesThisMonth.size() + " velas", currentMonth);
                                return new MonthMarketCache(0, 0, 0, 0, candlesThisMonth, false);
                            }
                            Vesta.info("(idx:%d) 📦 Exportando Pair (C:%d)", currentMonth, candlesThisMonth.size());
                            Pair<float[][][], float[][]> pair = BuilderData.build(candlesThisMonth, LOOK_BACK, futureWindow);
                            AtomicReference<float[][][]> Xraw = new AtomicReference<>(pair.getKey());
                            AtomicReference<float[][]> yraw = new AtomicReference<>(pair.getValue());
                            if (Xraw.get().length == 0) {
                                return new MonthMarketCache(0, 0, 0, 0, candlesThisMonth, false);
                            }


                            int samples = Xraw.get().length;
                            int seqLen = Xraw.get()[0].length;
                            int features = Xraw.get()[0][0].length;
                            int yCols = yraw.get()[0].length;
                            if (maxMonth < 6) {
                                candlesThisMonth.clear();
                            }
                            pair = null;
                            // Guarda el resultado del pair
                            Vesta.info("(idx:%d) 💿 Guardando resultado del Pair", currentMonth);
                            MonthMarketCache cache = new MonthMarketCache(samples, seqLen, features, yCols, candlesThisMonth, true);
                            waitingCacheSave.add(CompletableFuture.supplyAsync(() -> {
                                try {
                                    Path path = IOdata.saveTrainingCache(IOdata.createTrainingCacheDir(), symbol, currentMonth, Xraw.get(), yraw.get(), false);
                                    cache.setCacheFile(path);
                                } catch (IOException ignored) {}
                                Arrays.fill(Xraw.get(), null);
                                Arrays.fill(yraw.get(), null);
                                Xraw.set(null);
                                yraw.set(null);
                                return new Object();
                            }, VestaEngine.EXECUTOR_WRITE_CACHE_BUILD));
                            return cache;
                        } catch (Exception e) {
                            Vesta.error("(idx:%d) Error procesando mes: " + e.getMessage(), currentMonth);
                            e.printStackTrace();
                            return new MonthMarketCache(0, 0, 0, 0, Collections.emptyList(), false);
                        }
                    }, VestaEngine.EXECUTOR_BUILD);

                    futures.add(future);
                }

                // Esperar a que todos los futuros completen y procesar en orden
                for (int i = 0; i < futures.size(); i++) {
                    try {
                        MonthMarketCache result = futures.get(i).get(); // Bloquea para mantener orden
                        int month = months.get(i);

                        if (result.hasData) {
                            if (maxMonth < 6) {
                                allCandlesForChart.addAll(result.candles);
                            }

                            if (result.samples > 0) {
                                // Añadir objeto que indica que se guardó caché del pair
                                cacheEntries.add(result);
                                Vesta.info("(%d/%d) ✅ procesado: %d muestras (%s)", month, maxMonth, result.samples, symbol);
                            }
                        }
                    } catch (InterruptedException | ExecutionException e) {
                        Vesta.error("(idx:%d) Error procesando mes para " + symbol + ": " + e.getMessage(), months.get(i));
                        e.printStackTrace();
                        Thread.currentThread().interrupt();
                    }
                }

                if (!allCandlesForChart.isEmpty()) {
                    ChartUtils.showCandleChart("Mercado", allCandlesForChart, symbol);
                }
            } catch (Exception e) {
                Vesta.error("Error construyendo data para " + symbol + ": " + e.getMessage());
                e.printStackTrace();
            }
        }

        Vesta.info("💤 Esperando que termine de escribir en disco");
        for (CompletableFuture<Object>  future : waitingCacheSave) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }
        if (cacheEntries.isEmpty()) {
            throw new RuntimeException("No se generó data de entrenamiento válida.");
        }

        // Concatenar todos los símbolos
        int totalSamples = cacheEntries.stream().mapToInt(entry -> entry.samples).sum();
        int seqLen = cacheEntries.get(0).seqLen;
        int features = cacheEntries.get(0).features;
        int yCols = cacheEntries.get(0).yCols;

        long delta = (System.currentTimeMillis() - time);
        Vesta.info("✅ Construcción completada de %s (S: %d) +T: %dm %ds", String.join(", ", symbols), totalSamples, delta/60_000,((delta/1000)%60));



        List<Path> paths = new ArrayList<>();


        System.gc();
        // Cargar la cache guardada


//        if (maxMonth > 5){
//        }else {
//            int currentIdx = 0;
//            float[][][] X_final = new float[totalSamples][seqLen][features];
//            float[][] y_final = new float[totalSamples][yCols];
//            for (PairCache entry : cacheEntries) {
//                try {
//                    Pair<float[][][], float[][]> pair = IOdata.loadTrainingCache(entry.cacheFile);
//                    float[][][] xPart = pair.getKey();
//                    float[][] yPart = pair.getValue();
//                    pair = null;
//                    int len = xPart.length;
//
//                    System.arraycopy(xPart, 0, X_final, currentIdx, len);
//                    System.arraycopy(yPart, 0, y_final, currentIdx, len);
//                    xPart = null;
//                    yPart = null;
//                    currentIdx += len;
//                    Vesta.info("💿 Datos recuperados de " + entry.cacheFile);
//                } catch (Exception e) {
//                    throw new RuntimeException("Error cargando cache temporal: " + entry.cacheFile, e);
//                } finally {
//                    IOdata.deleteTrainingCache(entry.cacheFile);
//                }
//            }
//
//            return new TrainingData(new Pair<>(X_final, y_final));
//        }
        return new TrainingData(cacheEntries.stream().map(PairCache::getCacheFile).toList(), totalSamples, seqLen, features, yCols);

    }

    @Getter
    private static class MonthMarketCache extends PairCache {
        final List<Candle> candles;
        final boolean hasData;

        MonthMarketCache(int samples, int seqLen, int features, int yCols, List<Candle> candles, boolean hasData) {
            super(samples, seqLen, features, yCols);
            this.candles = candles;
            this.hasData = hasData;
        }
    }

    @Setter
    @Getter
    private static class PairCache {
        Path cacheFile;
        final int samples;
        final int seqLen;
        final int features;
        final int yCols;

        PairCache(int samples, int seqLen, int features, int yCols) {
            this.samples = samples;
            this.seqLen = seqLen;
            this.features = features;
            this.yCols = yCols;
        }
    }

    /**
     * Construye tensores con features relativas y etiquetas:
     * [upMove, downMove, firstHitFlag, 0, 0]
     * firstHitFlag: 0 si el mínimo se alcanza primero, 1 si el máximo se alcanza primero.
     */
    @Contract("_, _, _ -> new")
    public static @NotNull Pair<float[][][], float[][]> build(@NotNull List<Candle> candles, int lookBack, int futureWindow) {
        int n = candles.size();
        int samples = n - lookBack - futureWindow;

        if (samples <= 0) return new Pair<>(new float[0][0][0], new float[0][0]);

        float[][][] X = new float[samples][lookBack][FEATURES];
        float[][] y = new float[samples][5];

        for (int i = 0; i < samples; i++) {
            // 1. Extraer Features (X)
            for (int j = 0; j < lookBack; j++) {
                X[i][j] = extractFeatures(candles.get(i + j + 1), candles.get(i + j));
            }

            // 2. Definir punto de entrada (Cierre de la ultima vela del lookback)
            double entryPrice = candles.get(i + lookBack).close();

            // --- ESCANEO DEL FUTURO (Max/Min por mecha) ---
            double maxWick = -Double.MAX_VALUE;
            double minWick = Double.MAX_VALUE;
            int maxIndex = -1;
            int minIndex = -1;

            for (int f = 1; f <= futureWindow; f++) {
                Candle future = candles.get(i + lookBack + f);

                if (future.high() > maxWick) {
                    maxWick = future.high();
                    maxIndex = f;
                }
                if (future.low() < minWick) {
                    minWick = future.low();
                    minIndex = f;
                }
            }

            double upMove = Math.abs(maxWick - entryPrice);
            double downMove = Math.abs(entryPrice - minWick);
            float firstHitFlag = (maxIndex <= minIndex) ? 1f : 0f;

            // 3. ASIGNACION DE ETIQUETAS (Y):
            // output 0: upMove (mecha) en porcentaje sobre entryPrice
            y[i][0] = (float) ((entryPrice > 0) ? Math.max(0.0, (upMove / entryPrice)) : 0.0);
            // output 1: downMove (mecha) en porcentaje sobre entryPrice
            y[i][1] = (float) ((entryPrice > 0) ? Math.max(0.0, (downMove / entryPrice)) : 0.0);

            // output 2: 0 si mínimo primero, 1 si máximo primero
            y[i][2] = firstHitFlag;
            y[i][3] = 0f;
            y[i][4] = 0f;
        }

        return new Pair<>(X, y);
    }

    public static @NotNull List<Candle> to1mCandles(@NotNull Market market) {
        return to1mCandles(market, DEFAULT_FUTURE_WINDOW);
    }

    public static @NotNull List<Candle> to1mCandles(@NotNull Market market, int futureWindow) {

        //market.sortd();
        int remove = 0;
        int idx = 0;

        // CandleSimple por minuto
        BaseBarSeries series = new BaseBarSeriesBuilder().build();
        NavigableMap<Long, CandleSimple> simpleByMinute = new TreeMap<>();
        for (CandleSimple cs : market.getCandleSimples()) {
            long minute = (cs.openTime() / 60_000) * 60_000;
            simpleByMinute.put(minute, cs);
            try {
                series.addBar(new BaseBar(Duration.ofSeconds(60),
                        Instant.ofEpochMilli(cs.openTime()),
                        Instant.ofEpochMilli(cs.openTime() + 60_000),
                        DecimalNum.valueOf(cs.open()),
                        DecimalNum.valueOf(cs.high()),
                        DecimalNum.valueOf(cs.low()),
                        DecimalNum.valueOf(cs.close()),
                        DecimalNum.valueOf(cs.volumen().baseVolume()),
                        DecimalNum.valueOf(cs.volumen().quoteVolume()),
                        0
                ));
            } catch (Exception e) {
                e.printStackTrace();
            }
        }

        ClosePriceIndicator indicator = new ClosePriceIndicator(series);

        // Depth por minuto
        NavigableMap<Long, Depth> depthByMinute = new TreeMap<>();
        for (Depth d : market.getDepths()) {
            long minute = (d.getDate() / 60_000) * 60_000;
            depthByMinute.put(minute, d);
        }

        market.buildTradeCache();
        NavigableMap<Long, List<Trade>> tradesByMinute = market.getTradesByMinuteCache();

        List<Candle> candles = new ArrayList<>();
        if (simpleByMinute.isEmpty()) return candles;

        long startMinute = simpleByMinute.firstKey();
        long endMinute = simpleByMinute.lastKey();

        double lastClose = Double.NaN;

        //RSI
        List<Double> closes = new ArrayList<>();
        for (CandleSimple cs : simpleByMinute.values()) {
            closes.add(cs.close());
        }

        RSIIndicator rsi4 = new RSIIndicator(indicator, 4);
        RSIIndicator rsi8 = new RSIIndicator(indicator, 8);
        RSIIndicator rsi16 = new RSIIndicator(indicator, 16);

        // SuperTren
        SuperTrendIndicator superTrend = new SuperTrendIndicator(series, 5, 0.4);


        // MACD
        //MACDIndicator macd = new MACDIndicator(indicator, 12, 26);
//        FinancialCalculation.MACDResult macdRes = FinancialCalculation.computeMACD(closes, 6, 16, 9);
        FinancialCalculation.MACDResult macdRes = FinancialCalculation.computeMACD(closes, 10, 18, 6);
        double[] macdArr = macdRes.macd();
        double[] signalArr = macdRes.signal();
        double[] histArr = macdRes.histogram();

        // NVI
        NVIIndicator nvi = new NVIIndicator(series);

        // Bollinger
        BollingerBandFacade facadeBand = new BollingerBandFacade(indicator, 20, 2);

        // ATR
        ATRIndicator atr14 = new ATRIndicator(series, 14);

        // Volumen Normalizado
        Map<String, double[]> vn = FinancialCalculation.computeVolumeNormalizations(simpleByMinute.values().stream().toList(), 14, atr14.stream().map(Num::doubleValue).toList());

        for (long minute = startMinute; minute <= endMinute; minute += 60_000L) {
            // OHLC + VOLUMEN
            CandleSimple cs = simpleByMinute.get(minute);

            double open, high, low, close;
            double volumeBase = 0;
            double quoteVolume = 0;
            double buyQV = 0;
            double sellQV = 0;
            double deltaUSDT = 0;
            double buyRatio = 0;
            int tradeCount = 0;

            if (cs != null) {
                open = cs.open();
                high = cs.high();
                low = cs.low();
                close = cs.close();
                lastClose = close;
                Volumen v = cs.volumen();
                quoteVolume = v.quoteVolume();
                buyQV = v.takerBuyQuoteVolume();
                sellQV = v.sellQuoteVolume();
                deltaUSDT = v.deltaUSDT();
                buyRatio = v.buyRatio();
                volumeBase = v.baseVolume();
            } else if (!Double.isNaN(lastClose)) {
                open = high = low = close = lastClose;
            } else {
                open = high = low = close = 0.0;
            }

            // Contar trades en este minuto
            if (tradesByMinute != null) {
                List<Trade> minuteTrades = tradesByMinute.get(minute);
                if (minuteTrades != null) {
                    tradeCount = minuteTrades.size();
                }
            }

            // DEPTH
            double bidLiq = 0, askLiq = 0, mid = close, spread = 0;

            Map.Entry<Long, Depth> floor = depthByMinute.floorEntry(minute);
            Depth depth = floor != null ? floor.getValue() : null;

            if (depth != null) {
                bidLiq = depth.getBids().stream()
                        .mapToDouble(o -> o.price() * o.qty())
                        .sum();

                askLiq = depth.getAsks().stream()
                        .mapToDouble(o -> o.price() * o.qty())
                        .sum();

                if (!depth.getBids().isEmpty() && !depth.getAsks().isEmpty()) {
                    double bestBid = depth.getBids().peekFirst().price();
                    double bestAsk = depth.getAsks().peekFirst().price();
                    mid = (bestBid + bestAsk) / 2.0;
                    spread = bestAsk - bestBid;
                }
            }
            double depthImbalance = (bidLiq + askLiq == 0) ? 0 : (bidLiq - askLiq) / (bidLiq + askLiq);

            try {

                // MACD
                double macdVal = checkDouble(macdArr, idx);
                double macdSignal = checkDouble(signalArr, idx);
                double macdHist = checkDouble(histArr, idx);

                candles.add(new Candle(
                        minute,
                        checkDouble(open), checkDouble(high), checkDouble(low), checkDouble(close),

                        calculateCandleDirectionSmooth(open, close, 0.5/100),
                        checkDouble(tradeCount),
                        checkDouble(volumeBase),
                        checkDouble(quoteVolume),
                        checkDouble(buyQV),
                        checkDouble(sellQV),

                        checkDouble(vn.get("ratio")[idx]),
                        checkDouble(vn.get("zscore")[idx]),
                        checkDouble(vn.get("perAtr")[idx]),

                        checkDouble(deltaUSDT),
                        checkDouble(buyRatio),
                        checkDouble(bidLiq),
                        checkDouble(askLiq),
                        checkDouble(depthImbalance),
                        checkDouble(mid),
                        checkDouble(spread),
                        rsi4.getValue(idx).doubleValue(),
                        rsi8.getValue(idx).doubleValue(),
                        rsi16.getValue(idx).doubleValue(),
                        macdVal,
                        macdSignal,
                        macdHist,
                        nvi.getValue(idx).doubleValue(),
                        facadeBand.upper().getValue(idx).doubleValue(),
                        facadeBand.middle().getValue(idx).doubleValue(),
                        facadeBand.lower().getValue(idx).doubleValue(),
                        facadeBand.bandwidth().getValue(idx).doubleValue(),
                        facadeBand.percentB().getValue(idx).doubleValue(),
                        atr14.getValue(idx).doubleValue(),
                        (float) (superTrend.getValue(idx).floatValue() - close)
                ));
            } catch (IllegalArgumentException ignored) {
                remove++;
            }
            idx++;
        }
        closes.clear();
        Vesta.info("Se elimino %d por dar resultado NA o Infinito", remove);
        return candles;
    }


    public static double checkDouble(double[] d, int i) throws IllegalArgumentException{
        if (i < d.length) {
            return checkDouble(d[i]);
        } else throw new IllegalArgumentException("Fuera del index");
    }

    public static double checkDouble(double d) throws IllegalArgumentException{
        if (Double.isInfinite(d) || Double.isNaN(d)) {
            throw new IllegalArgumentException("The input is infinite or NaN");
        }
        return d;
    }

    public static float calculateCandleDirectionSmooth(double open, double close, double maxChangePercent) {
        if (Math.abs(open) < 0.0000001) return 0.0f;

        double changePercent = (close - open) / open;
        double normalized = changePercent / maxChangePercent;

        normalized = Math.max(-2.0, Math.min(2.0, normalized));

        // Aplicar sigmoide ajustada para [-1, 1]
        // Usamos tanh que ya está en el rango [-1, 1]
        return (float) Math.tanh(normalized);
    }


    public final static int FEATURES;

    private static final double SAFE_EPS = 1e-8;

    private static float safeLogRatio(double numerator, double denominator) {
        if (!Double.isFinite(numerator) || !Double.isFinite(denominator)) return 0f;
        if (numerator <= 0.0 || denominator <= 0.0) return 0f;
        double ratio = numerator / denominator;
        if (!Double.isFinite(ratio) || ratio <= 0.0) return 0f;
        double v = Math.log(ratio);
        return Double.isFinite(v) ? (float) v : 0f;
    }

    private static float safeLog1p(double value) {
        if (!Double.isFinite(value)) return 0f;
        if (value <= -1.0) return 0f;
        double v = Math.log1p(Math.max(0.0, value));
        return Double.isFinite(v) ? (float) v : 0f;
    }

    private static float safeDiv(double numerator, double denominator) {
        if (!Double.isFinite(numerator) || !Double.isFinite(denominator)) return 0f;
        if (Math.abs(denominator) < SAFE_EPS) return 0f;
        double v = numerator / denominator;
        return Double.isFinite(v) ? (float) v : 0f;
    }

    private static float safeFloat(double value) {
        return Double.isFinite(value) ? (float) value : 0f;
    }

    public static float @NotNull [] extractFeatures(@NotNull Candle curr, @NotNull Candle prev) {

        List<Float> fList = new ArrayList<>();

        // 1-4: Precios relativos (Log Returns)
        fList.add(safeLogRatio(curr.high(), prev.high()));
        fList.add(safeLogRatio(curr.open(), prev.open()));
        fList.add(safeLogRatio(curr.close(), prev.close()));
        fList.add(safeLogRatio(curr.low(), prev.low()));

        fList.add(safeFloat(curr.direccion()));
        fList.add(safeLog1p(curr.amountTrades()));

        // Volúmenes relativos
        fList.add(safeFloat(curr.volRatioToMean()));
        fList.add(safeFloat(curr.volZscore()));
        fList.add(safeFloat(curr.volPerAtr()));
//        fList.add((float) Math.log(curr.quoteVolume() / prevClose));
//        fList.add((float) Math.log((curr.buyQuoteVolume() - prev.buyQuoteVolume()) / curr.buyQuoteVolume()));// Dan 0
//        fList.add((float) Math.log((curr.sellQuoteVolume() - prev.sellQuoteVolume()) / curr.sellQuoteVolume()));

        // Delta y Buy Ratio
        double totalVol = curr.buyQuoteVolume() + curr.sellQuoteVolume();
        fList.add(safeDiv(curr.deltaUSDT(), totalVol));
        fList.add(safeFloat(curr.buyRatio()));

//        fList.add((float) Math.log(curr.bidLiquidity() / prevClose));
//        fList.add((float) Math.log(curr.askLiquidity() / prevClose));

        // 12-14: Métricas de Orderbook relativas
//        fList.add((float) curr.depthImbalance());
//        fList.add((float) ((curr.midPrice() - curr.close()) / curr.close()));
//        fList.add((float) (curr.spread() / curr.close()));

        // RSI
//        fList.add(safeDiv(curr.rsi4(), 100.0));
        fList.add(safeDiv(curr.rsi8(), 100.0));
        fList.add(safeDiv(curr.rsi16(), 100.0));

        // MACD
        fList.add(safeDiv(curr.macdVal(), curr.close()));
        fList.add(safeDiv(curr.macdSignal(), curr.close()));
        fList.add(safeDiv(curr.macdHist(), curr.close()));

        // NVI
        fList.add(safeDiv(curr.nvi(), curr.close()));

        // Bollinger
        double bbUpper = curr.upperBand();
        double bbLower = curr.lowerBand();
        double bbMiddle = curr.middleBand();

        double bbRange = bbUpper - bbLower;
        float bbBandwidth = safeDiv(bbRange, bbMiddle);
        float bbPos = safeDiv(curr.close() - bbMiddle, bbRange);

        fList.add(bbBandwidth);
        fList.add(bbPos);
        // ATR
        fList.add(safeDiv(curr.atr14(), curr.close()));

        float[] f = new float[fList.size()];
        for (int i = 0; i < fList.size(); i++) {
            float v = fList.get(i);
            f[i] = Float.isFinite(v) ? v : 0f;
        }
        return f;
    }

    /**
     * Añade características 2
     * 1 el mercado que esta los datos
     * 2 cuantos mercados puede haber
     */
    public static float[][][] addSymbolFeature(float[][][] X, String symbol) {
        float[][][] XwithSymbol = new float[X.length][X[0].length][X[0][0].length + 2];

        // Codificación one-hot simplificada del símbolo
        int symbolIndex = Vesta.MARKETS_NAMES.indexOf(symbol);
        float symbolOneHot = symbolIndex / (float) Vesta.MARKETS_NAMES.size();
        float symbolNorm = (float) Math.log(symbolIndex + 1) / (float) Math.log(Vesta.MARKETS_NAMES.size() + 1);

        for (int i = 0; i < X.length; i++) {
            for (int j = 0; j < X[0].length; j++) {
                // Copiar características originales
                System.arraycopy(X[i][j], 0, XwithSymbol[i][j], 0, X[0][0].length);
                // Añadir características del símbolo
                XwithSymbol[i][j][X[0][0].length] = symbolOneHot;
                XwithSymbol[i][j][X[0][0].length + 1] = symbolNorm;
            }
        }
        return XwithSymbol;
    }


    static {
        FEATURES = extractFeatures(
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1, 1,1,1,1,1,
                        1, 1 ,1 ,1, 1,1),
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1,1,1,1,1,1, 1,
                        1,1,1, 1, 1)
        ).length; // más 2 por que tiene sumar el feature del símbolo en el que esta y todos los símbolos que puede estar
    }

}





