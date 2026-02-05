package xyz.cereshost.builder;

import ai.djl.util.Pair;
import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.ta4j.core.*;
import org.ta4j.core.indicators.ATRIndicator;
import org.ta4j.core.indicators.RSIIndicator;
import org.ta4j.core.indicators.bollinger.*;
import org.ta4j.core.indicators.helpers.ClosePriceIndicator;
import org.ta4j.core.indicators.volume.NVIIndicator;
import org.ta4j.core.num.DecimalNum;
import org.ta4j.core.num.Num;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.FinancialCalculation;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.io.IOdata;

import java.time.Duration;
import java.time.Instant;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

import static xyz.cereshost.engine.VestaEngine.LOOK_BACK;

public class BuilderData {

    public static @NotNull Pair<float[][][], float[][]> fullBuild(@NotNull List<String> symbols, int maxMonth, int offset) {
        List<float[][][]> allX = new ArrayList<>();
        List<float[][]> allY = new ArrayList<>();
        for (String symbol : symbols) {
            try {
                List<Candle> allCandlesForChart = new ArrayList<>();

                Vesta.info("Procesando símbolo (Relativo): " + symbol);


                // Procesar cada mes por separado SIN acumular
                List<Integer> months = IntStream.rangeClosed(1, maxMonth)
                        .boxed()
                        .toList();

                // Crear lista de futuros para procesar cada mes de forma asincrónica
                List<CompletableFuture<MonthResult>> futures = new ArrayList<>();

                for (int month = maxMonth + offset; month > offset; month--) {
                    final int currentMonth = month;
                    CompletableFuture<MonthResult> future = CompletableFuture.supplyAsync(() -> {
                        try {
                            Vesta.info("🖥️ Comenzado carga de datos del mes %d", currentMonth);
                            Market market = IOdata.loadMarkets(Main.DATA_SOURCE_FOR_TRAINING_MODEL, symbol, currentMonth);
                            List<Candle> candlesThisMonth = BuilderData.to1mCandles(market);

                            if (candlesThisMonth.size() <= LOOK_BACK + 2) {
                                Vesta.warning("Mes " + currentMonth + " insuficiente historial: " + candlesThisMonth.size() + " velas");
                                return new MonthResult(null, null, candlesThisMonth, false);
                            }

                            Pair<float[][][], float[][]> pair = BuilderData.build(candlesThisMonth, LOOK_BACK, 20);
                            float[][][] Xraw = addSymbolFeature(pair.getKey(), symbol);
                            float[][] yraw = pair.getValue();

                            return new MonthResult(Xraw, yraw, candlesThisMonth, true);
                        } catch (Exception e) {
                            Vesta.error("Error procesando mes " + currentMonth + " para " + symbol + ": " + e.getMessage());
                            return new MonthResult(null, null, Collections.emptyList(), false);
                        }
                    }, VestaEngine.EXECUTOR_BUILD);

                    futures.add(future);
                }

                // Esperar a que todos los futuros completen y procesar en orden
                for (int i = 0; i < futures.size(); i++) {
                    try {
                        MonthResult result = futures.get(i).get(); // Bloquea para mantener orden
                        int month = months.get(i);

                        if (result.hasData) {
                            if (maxMonth < 6) {
                                allCandlesForChart.addAll(result.candles);
                            }

                            if (result.X != null && result.X.length > 0) {
                                allX.add(result.X);
                                allY.add(result.y);
                                Vesta.info("✅ Mes " + month + " procesado: " + result.X.length + " muestras (" + symbol + ")");
                            }
                        }
                        System.gc();
                    } catch (InterruptedException | ExecutionException e) {
                        Vesta.error("Error procesando mes " + months.get(i) + " para " + symbol + ": " + e.getMessage());
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



        if (allX.isEmpty()) {
            throw new RuntimeException("No se generó data de entrenamiento válida.");
        }

        // Concatenar todos los símbolos
        int totalSamples = allX.stream().mapToInt(x -> x.length).sum();
        int seqLen = allX.get(0)[0].length;
        int features = allX.get(0)[0][0].length;

        float[][][] X_final = new float[totalSamples][seqLen][features];
        float[][] y_final = new float[totalSamples][2];

        int currentIdx = 0;
        for (int i = 0; i < allX.size(); i++) {
            float[][][] xPart = allX.get(i);
            float[][] yPart = allY.get(i);
            int len = xPart.length;

            System.arraycopy(xPart, 0, X_final, currentIdx, len);
            System.arraycopy(yPart, 0, y_final, currentIdx, len);
            currentIdx += len;
        }


        return new Pair<>(X_final, y_final);
    }

    @Getter
    private static class MonthResult {
        final float[][][] X;
        final float[][] y;
        final List<Candle> candles;
        final boolean hasData;

        MonthResult(float[][][] X, float[][] y, List<Candle> candles, boolean hasData) {
            this.X = X;
            this.y = y;
            this.candles = candles;
            this.hasData = hasData;
        }
    }

    /**
     * Construye tensores basados EXCLUSIVAMENTE en cambios relativos (Log Returns)
     */
    @Contract("_, _, _ -> new")
    public static @NotNull Pair<float[][][], float[][]> build(@NotNull List<Candle> candles, int lookBack, int futureWindow) {
        int n = candles.size();
        int samples = n - lookBack - futureWindow;

        if (samples <= 0) return new Pair<>(new float[0][0][0], new float[0][0]);

        float[][][] X = new float[samples][lookBack][features];
        float[][] y = new float[samples][5];

        int[] direccionCount = new int[3]; // 0: Long, 1: Short, 2: Neutral

        // El umbral mínimo de beneficio (THRESHOLD_PRICE ej: 0.001 = 0.1%)
        double MIN_GAIN = PredictionEngine.THRESHOLD_PRICE;

        for (int i = 0; i < samples; i++) {
            // 1. Extraer Features (X)
            for (int j = 0; j < lookBack; j++) {
                X[i][j] = extractFeatures(candles.get(i + j + 1), candles.get(i + j));
            }

            // 2. Definir punto de entrada (Cierre de la última vela del lookback)
            double entryPrice = candles.get(i + lookBack).close();

            // Variables para encontrar el mejor trade posible en la ventana futura
            double bestLogTP_Long = 0;
            double bestLogSL_Long = 0;
            boolean longWasValid = false;

            double bestLogTP_Short = 0;
            double bestLogSL_Short = 0;
            boolean shortWasValid = false;

            // --- ESCANEO DEL FUTURO (Smart Labeling) ---
            double highestInWindow = -1;
            double lowestInWindow = Double.MAX_VALUE;

            for (int f = 1; f <= futureWindow; f++) {
                Candle future = candles.get(i + lookBack + f);

                // Actualizar extremos alcanzados hasta este momento en el futuro
                if (future.high() > highestInWindow) highestInWindow = future.high();
                if (future.low() < lowestInWindow) lowestInWindow = future.low();

                // Evaluar potencial LONG en esta vela futura
                double currentMaxGain = Math.log(future.high() / entryPrice);
                if (currentMaxGain >= MIN_GAIN) {
                    double theoreticalSL = currentMaxGain / 3.0; // Queremos ratio 1:2
                    double maxDrawdown = Math.log(entryPrice / lowestInWindow);

                    // Si el drawdown hasta ahora no ha superado el SL teórico, el trade es válido
                    if (maxDrawdown < theoreticalSL) {
                        if (currentMaxGain > bestLogTP_Long) {
                            bestLogTP_Long = currentMaxGain;
                            bestLogSL_Long = theoreticalSL;
                            longWasValid = true;
                        }
                    }
                }

                // Evaluar potencial SHORT en esta vela futura
                double currentMaxFall = Math.log(entryPrice / future.low());
                if (currentMaxFall >= MIN_GAIN) {
                    double theoreticalSL = currentMaxFall / 3.0; // Ratio 1:2
                    double maxRunup = Math.log(highestInWindow / entryPrice);

                    // Si la subida en contra no ha superado el SL teórico, el trade es válido
                    if (maxRunup < theoreticalSL) {
                        if (currentMaxFall > bestLogTP_Short) {
                            bestLogTP_Short = currentMaxFall;
                            bestLogSL_Short = theoreticalSL;
                            shortWasValid = true;
                        }
                    }
                }
            }

            // 3. ASIGNACIÓN DE ETIQUETAS (Y)
            // Decidimos cuál fue la mejor oportunidad (Long vs Short)
            if (longWasValid && (!shortWasValid || bestLogTP_Long > bestLogTP_Short)) {
                y[i][0] = (float) bestLogTP_Long;
                y[i][1] = (float) bestLogSL_Long;
                y[i][2] = 1.0f; // Long
                y[i][3] = 0f;
                y[i][4] = 0f;
                direccionCount[0]++;
            }
            else if (shortWasValid && (!longWasValid || bestLogTP_Short > bestLogTP_Long)) {
                y[i][0] = (float) bestLogTP_Short;
                y[i][1] = (float) bestLogSL_Short;
                y[i][2] = 0f;
                y[i][3] = 0f;
                y[i][4] = 1.0f; // Short
                direccionCount[1]++;
            }
            else {
                // NEUTRAL: O no hubo movimiento suficiente, o la volatilidad sacó ambos SL
                y[i][0] = (float) Math.abs(Math.log(highestInWindow / entryPrice)); // Volatilidad como TP
                y[i][1] = (float) Math.abs(Math.log(entryPrice / lowestInWindow));  // Volatilidad como SL
                y[i][2] = 0f;
                y[i][3] = 1.0f; // Neutral
                y[i][4] = 0f;
                direccionCount[2]++;
            }
        }

        Vesta.info("Build completado -> L: %d | S: %d | N: %d",
                direccionCount[0], direccionCount[1], direccionCount[2]);

        return new Pair<>(X, y);
    }

    public static @NotNull List<Candle> to1mCandles(@NotNull Market market) {

        market.sortd();
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


        // MACD
        //MACDIndicator macd = new MACDIndicator(indicator, 12, 26);
        FinancialCalculation.MACDResult macdRes = FinancialCalculation.computeMACD(closes, 12, 26, 9);
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
                        atr14.getValue(idx).doubleValue()
                ));
            } catch (IllegalArgumentException ignored) {
                remove++;
            }
            idx++;
        }
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


    @Getter
    private static int features;

    public static float @NotNull [] extractFeatures(@NotNull Candle curr, @NotNull Candle prev) {

        double prevClose = prev.close() <= 0 ? 1.0 : prev.close();
        List<Float> fList = new ArrayList<>();

        // 1-4: Precios relativos (Log Returns)
        fList.add((float) Math.log(Math.clamp(curr.high() / prev.high(), 1e-7, Double.POSITIVE_INFINITY)));
        fList.add((float) Math.log(Math.clamp(curr.open() / prev.open(), 1e-7, Double.POSITIVE_INFINITY)));
        fList.add((float) Math.log(Math.clamp(curr.close() / prev.close(), 1e-7, Double.POSITIVE_INFINITY)));
        fList.add((float) Math.log(Math.clamp(curr.low() / prev.low(), 1e-7, Double.POSITIVE_INFINITY)));

        fList.add(curr.direccion());
        fList.add((float) Math.log(curr.amountTrades()));

        // Volúmenes relativos
        fList.add((float) curr.volRatioToMean());
        fList.add((float) curr.volZscore());
        fList.add((float) curr.volPerAtr());
//        fList.add((float) Math.log(curr.quoteVolume() / prevClose));
//        fList.add((float) Math.log((curr.buyQuoteVolume() - prev.buyQuoteVolume()) / curr.buyQuoteVolume()));// Dan 0
//        fList.add((float) Math.log((curr.sellQuoteVolume() - prev.sellQuoteVolume()) / curr.sellQuoteVolume()));

        // Delta y Buy Ratio
        double totalVol = curr.buyQuoteVolume() + curr.sellQuoteVolume();
        fList.add((totalVol == 0) ? 0 : (float) (curr.deltaUSDT() / totalVol));
        fList.add((float) curr.buyRatio());

//        fList.add((float) Math.log(curr.bidLiquidity() / prevClose));
//        fList.add((float) Math.log(curr.askLiquidity() / prevClose));

        // 12-14: Métricas de Orderbook relativas
//        fList.add((float) curr.depthImbalance());
//        fList.add((float) ((curr.midPrice() - curr.close()) / curr.close()));
//        fList.add((float) (curr.spread() / curr.close()));

        // RSI
        fList.add((float) curr.rsi4()/100);
        fList.add((float) curr.rsi8()/100);
        fList.add((float) curr.rsi16()/100);

        // MACD
        fList.add((float) (curr.macdVal() / curr.close()));
        fList.add((float) (curr.macdSignal() / curr.close()));
        fList.add((float) (curr.macdHist() / curr.close()));

        // NVI
        fList.add((float) (curr.nvi() / curr.close()));

        // Bollinger
        double bbUpper = curr.upperBand();
        double bbLower = curr.lowerBand();
        double bbMiddle = curr.middleBand();

        double bbRange = bbUpper - bbLower;
        float bbBandwidth = 0f;
        if (bbMiddle > 0 && bbRange > 0) {
            bbBandwidth = (float) (bbRange / bbMiddle);
        }
        float bbPos = 0f;
        if (bbRange > 0) {
            bbPos = (float) ((curr.close() - bbMiddle) / bbRange);
        }

        fList.add(bbBandwidth);
        fList.add(bbPos);
        // ATR
        fList.add((float) (curr.atr14() / curr.close()));

        float[] f = new float[fList.size()];
        for (int i = 0; i < fList.size(); i++) {
            f[i] = fList.get(i);
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
        features = extractFeatures(
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1, 1,1,1,1,1,
                        1, 1 ,1 ,1, 1),
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1,1,1,1,1,1, 1,
                        1,1,1, 1)
        ).length + 2; // más 2 por que tiene sumar el feature del símbolo en el que esta y todos los símbolos que puede estar
    }

}