package xyz.cereshost.builder;

import ai.djl.util.Pair;
import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.FinancialCalculation;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.file.IOdata;

import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

import static xyz.cereshost.engine.VestaEngine.LOOK_BACK;

public class BuilderData {

    @SuppressWarnings("ConstantValue")
    public static @NotNull Pair<float[][][], float[][]> fullBuild(@NotNull List<String> symbols) {
        List<float[][][]> allX = new ArrayList<>();
        List<float[][]> allY = new ArrayList<>();
        for (String symbol : symbols) {
            try {
                List<Candle> allCandlesForChart = new ArrayList<>();

                Vesta.info("Procesando símbolo (Relativo): " + symbol);
                int maxMonth = Main.MAX_MONTH_TRAINING;

                // Procesar cada mes por separado SIN acumular
                List<Integer> months = IntStream.rangeClosed(1, maxMonth)
                        .boxed()
                        .toList();

                // Crear lista de futuros para procesar cada mes de forma asincrónica
                List<CompletableFuture<MonthResult>> futures = new ArrayList<>();

                for (int month = maxMonth; month > 0; month--) {
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

                            Pair<float[][][], float[][]> pair = BuilderData.build(candlesThisMonth, LOOK_BACK, 10);
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
                    double theoreticalSL = currentMaxGain / 2.0; // Queremos ratio 1:2
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
                    double theoreticalSL = currentMaxFall / 2.0; // Ratio 1:2
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
        NavigableMap<Long, CandleSimple> simpleByMinute = new TreeMap<>();
        for (CandleSimple cs : market.getCandleSimples()) {
            long minute = (cs.openTime() / 60_000) * 60_000;
            simpleByMinute.put(minute, cs);
        }

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

        double[] rsi4Arr = FinancialCalculation.computeRSI(closes, 4);
        double[] rsi8Arr = FinancialCalculation.computeRSI(closes, 8);
        double[] rsi16Arr = FinancialCalculation.computeRSI(closes, 16);


        // MACD
        FinancialCalculation.MACDResult macdRes = FinancialCalculation.computeMACD(closes, 12, 26, 9);
        double[] macdArr = macdRes.macd();
        double[] signalArr = macdRes.signal();
        double[] histArr = macdRes.histogram();

        // NVI
        double[] nviArr = FinancialCalculation.computeNVI(closes, simpleByMinute.values().stream().map(c -> c.volumen().quoteVolume()).toList(), 1000.0);

        // Bollinger
        FinancialCalculation.BollingerBandsResult bollingerBandsResult = FinancialCalculation.computeBollingerBands(closes, 20, 2);
        double[] upperBandArr = bollingerBandsResult.upperBand();
        double[] middleBandArr = bollingerBandsResult.middleBand();
        double[] lowerBandArr = bollingerBandsResult.lowerBand();
        double[] bandwidthArr = bollingerBandsResult.bandwidth();
        double[] percentBArr = bollingerBandsResult.percentB();

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
                // RSI
                double rsi4 = checkDouble(rsi4Arr, idx);
                double rsi8 = checkDouble(rsi8Arr, idx);
                double rsi16 = checkDouble(rsi16Arr, idx);

                // MACD
                double macdVal = checkDouble(macdArr, idx);
                double macdSignal = checkDouble(signalArr, idx);
                double macdHist = checkDouble(histArr, idx);

                // NVI
                double nvi = checkDouble(nviArr, idx);

                double upperBand = checkDouble(upperBandArr, idx);
                double middleBand = checkDouble(middleBandArr, idx);
                double lowerBand = checkDouble(lowerBandArr, idx);
                double bandwidth = checkDouble(bandwidthArr, idx);
                double percentB = checkDouble(percentBArr, idx);

                candles.add(new Candle(
                        minute,
                        checkDouble(open), checkDouble(high), checkDouble(low), checkDouble(close),
                        calculateCandleDirectionSmooth(open, close, 0.5/100),
                        checkDouble(tradeCount),
                        checkDouble(volumeBase),
                        checkDouble(quoteVolume),
                        checkDouble(buyQV),
                        checkDouble(sellQV),
                        checkDouble(deltaUSDT),
                        checkDouble(buyRatio),
                        checkDouble(bidLiq),
                        checkDouble(askLiq),
                        checkDouble(depthImbalance),
                        checkDouble(mid),
                        checkDouble(spread),
                        rsi4,
                        rsi8,
                        rsi16,
                        macdVal,
                        macdSignal,
                        macdHist,
                        nvi,
                        upperBand,
                        middleBand,
                        lowerBand,
                        bandwidth,
                        percentB
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

        // Umbral de doji
//        if (Math.abs(changePercent) < PredictionEngine.THRESHOLD/5) {
//            return 0.0;
//        }

        // Normalizar al rango [-1, 1] considerando maxChangePercent
        double normalized = changePercent / maxChangePercent;

        // Limitar a [-2, 2] para que la sigmoide cubra bien el rango
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
        fList.add((float) Math.log(curr.open() / prevClose));
        fList.add((float) Math.log(curr.high() / prevClose));
        fList.add((float) Math.log(curr.low() / prevClose));
        fList.add((float) Math.log(curr.close() / prevClose));

        fList.add(curr.direccion());
        fList.add((float) Math.log(curr.amountTrades()));

        // Volúmenes relativos
        fList.add((float) Math.log(curr.volumeBase() / prevClose));
//        fList.add((float) Math.log(curr.quoteVolume() / prevClose));
        fList.add((float) Math.log(curr.buyQuoteVolume() / prevClose));
        fList.add((float) Math.log(curr.sellQuoteVolume() / prevClose));

        // Delta y Buy Ratio
        double totalVol = curr.buyQuoteVolume() + curr.sellQuoteVolume();
        fList.add((totalVol == 0) ? 0 : (float) (curr.deltaUSDT() / totalVol));
        fList.add((float) curr.buyRatio());

//        fList.add((float) Math.log(curr.bidLiquidity() / prevClose));
//        fList.add((float) Math.log(curr.askLiquidity() / prevClose));
//
//        // 12-14: Métricas de Orderbook relativas
//        fList.add((float) curr.depthImbalance());
//        fList.add((float) ((curr.midPrice() - curr.close()) / curr.close()));
//        fList.add((float) (curr.spread() / curr.close()));

        // RSI
        fList.add((float) curr.rsi4());
        fList.add((float) curr.rsi8());
        fList.add((float) curr.rsi16());

        // MACD
        fList.add((float) curr.macdVal());
        fList.add((float) curr.macdSignal());
        fList.add((float) curr.macdHist());

        // NVI
        fList.add((float) curr.nvi());

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
                1,1,1, 1, 1, 1, 1,1,1,1,1, 1),
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1,1,1,1,1,1, 1)
        ).length + 2; // más 2 por que tiene sumar el feature del símbolo en el que esta y todos los símbolos que puede estar
    }

}