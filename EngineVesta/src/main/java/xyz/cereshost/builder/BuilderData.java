package xyz.cereshost.builder;

import ai.djl.util.Pair;
import lombok.Getter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.FinancialCalculation;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;
import xyz.cereshost.engine.PredictionEngine;

import java.util.*;

import static xyz.cereshost.engine.VestaEngine.LOOK_BACK;

public class BuilderData {

    public static Pair<float[][][], float[][]> combineDatasets(@NotNull List<float[][][]> allX, List<float[][]> allY) {
        // ... código estándar de combinación ...
        // Asegúrate de copiar las 3 columnas de Y
        int totalSamples = allX.stream().mapToInt(x -> x.length).sum();
        float[][][] Xcombined = new float[totalSamples][allX.get(0)[0].length][allX.get(0)[0][0].length];
        float[][] ycombined = new float[totalSamples][3]; // <--- IMPORTANTE: 3 Columnas

        int idx = 0;
        for(int k=0; k<allX.size(); k++){
            float[][] y = allY.get(k);
            float[][][] X = allX.get(k);
            for(int i=0; i<X.length; i++){
                Xcombined[idx] = X[i];
                ycombined[idx][0] = y[i][0];
                ycombined[idx][1] = y[i][1];
                ycombined[idx][2] = y[i][2]; // Copiar dirección sin tocar
                idx++;
            }
        }
        return new Pair<>(Xcombined, ycombined);
    }

    public static @NotNull Pair<float[][][], float[][]> fullBuild(@NotNull List<String> symbols) {
        List<float[][][]> allX = new ArrayList<>();
        List<float[][]> allY = new ArrayList<>();

        for (String symbol : symbols) {
            try {
                Vesta.info("Procesando símbolo (Relativo): " + symbol);
                List<Candle> candles = BuilderData.to1mCandles(Vesta.MARKETS.get(symbol));
                ChartUtils.CandleChartUtils.showCandleChart("Mercado", candles, symbol);

                candles.sort(Comparator.comparingLong(Candle::openTime));

                // Necesitamos al menos LOOK_BACK + 2 (uno extra para el cálculo relativo inicial)
                if (candles.size() <= LOOK_BACK + 2) {
                    Vesta.error("Símbolo " + symbol + " insuficiente historial.");
                    continue;
                }

                Pair<float[][][], float[][]> pair = BuilderData.build(candles, LOOK_BACK, 10);
                float[][][] Xraw = addSymbolFeature(pair.getKey(), symbol);
                float[][] yraw = pair.getValue();

                if (Xraw.length > 0) {
                    allX.add(Xraw);
                    allY.add(yraw);
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

    /**
     * Construye tensores basados EXCLUSIVAMENTE en cambios relativos (Log Returns)
     */
    @Contract("_, _, _ -> new")
    public static @NotNull Pair<float[][][], float[][]> build(@NotNull List<Candle> candles, int lookBack, int futureWindow) {
        int n = candles.size();
        // Ahora restamos futureWindow para asegurar que siempre haya datos hacia adelante
        int samples = n - lookBack - futureWindow;

        if (samples <= 0) return new Pair<>(new float[0][0][0], new float[0][0]);

        float[][][] X = new float[samples][lookBack][features];
        float[][] y = new float[samples][3];

        int[] direccionCount = new int[3];
        double MIN_GAIN = 0.001;
        for (int i = 0; i < samples; i++) {
            for (int j = 0; j < lookBack; j++) {
                X[i][j] = extractFeatures(candles.get(i + j + 1), candles.get(i + j));
            }

            double entryPrice = candles.get(i + lookBack).close();

            double bestPriceLong = -Double.MAX_VALUE;
            double worstPriceLong = Double.MAX_VALUE;
            double bestPriceShort = Double.MAX_VALUE;
            double worstPriceShort = -Double.MAX_VALUE;

            for (int f = 1; f <= futureWindow; f++) {
                Candle future = candles.get(i + lookBack + f);
                bestPriceLong = Math.max(bestPriceLong, future.close());
                worstPriceLong = Math.min(worstPriceLong, future.low());
                bestPriceShort = Math.min(bestPriceShort, future.close());
                worstPriceShort = Math.max(worstPriceShort, future.high());
            }

            double finalCloseInWindow = candles.get(i + lookBack + futureWindow).close();
            double totalMovementLog = Math.log(finalCloseInWindow / entryPrice);

            boolean conditionMet = false;

            if (totalMovementLog > PredictionEngine.THRESHOLD) {
                // --- Lógica LONG ---
                double logTP = Math.log(bestPriceLong / entryPrice);
                double logSL = Math.max(0.00001, Math.log(entryPrice / worstPriceLong)); // Evitamos división por cero

                if (Double.isInfinite(logTP) || Double.isNaN(logTP)) logTP = 0;
                if (Double.isInfinite(logSL) || Double.isNaN(logSL)) logSL = 0;

                // Filtros: Ganancia >= 0.1% Y Ratio RR >= 1:1 (TP >= SL)
                if (logTP >= MIN_GAIN && logTP >= logSL) {
                    y[i][0] = (float) Math.abs(logTP);
                    y[i][1] = (float) Math.abs(logSL);
                    y[i][2] = 1.0f; // LONG
                    direccionCount[0]++;
                    conditionMet = true;
                }
            }else if (totalMovementLog < -PredictionEngine.THRESHOLD) {
                // --- Lógica SHORT ---
                double logTP = Math.log(entryPrice / bestPriceShort);
                double logSL = Math.max(0.00001, Math.log(worstPriceShort / entryPrice)); // Evitamos división por cero

                if (Double.isInfinite(logTP) || Double.isNaN(logTP)) logTP = 0;
                if (Double.isInfinite(logSL) || Double.isNaN(logSL)) logSL = 0;

                // Filtros: Ganancia >= 0.1% Y Ratio RR >= 1:1 (TP >= SL)
                if (logTP >= MIN_GAIN && logTP*2 >= logSL) {
                    y[i][0] = (float) Math.abs(logTP);
                    y[i][1] = (float) Math.abs(logSL);
                    y[i][2] = -1.0f; // SHORT
                    direccionCount[1]++;
                    conditionMet = true;
                }
            }

            // Si no cumple el THRESHOLD o no pasó los filtros de RR/Ganancia Mínima -> NEUTRAL
            if (!conditionMet) {
                double volatilityUp = Math.abs(Math.log(bestPriceLong / entryPrice));
                double volatilityDown = Math.abs(Math.log(worstPriceLong / entryPrice));

                if (Double.isInfinite(volatilityUp) || Double.isNaN(volatilityUp)) volatilityUp = 0;
                if (Double.isInfinite(volatilityDown) || Double.isNaN(volatilityDown)) volatilityDown = 0;

                y[i][0] = (float) volatilityUp;
                y[i][1] = (float) volatilityDown;
                y[i][2] = 0.0f; // NEUTRAL
                direccionCount[2]++;
            }
        }
        Vesta.info("Direcciones: " +  direccionCount[0] + "L " +  direccionCount[1] + "S " + direccionCount[2] + "N");
        return new Pair<>(X, y);
    }

    public static @NotNull List<Candle> to1mCandles(@NotNull Market market) {

        market.sortd();
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

            // RSI
            double rsi4 = idx < rsi4Arr.length ? rsi8Arr[idx] : Double.NaN;
            double rsi8 = idx < rsi8Arr.length ? rsi8Arr[idx] : Double.NaN;
            double rsi16 = idx < rsi16Arr.length ? rsi16Arr[idx] : Double.NaN;

            // MACD
            double macdVal = idx < macdArr.length ? macdArr[idx] : Double.NaN;
            double macdSignal = idx < signalArr.length ? signalArr[idx] : Double.NaN;
            double macdHist = idx < histArr.length ? histArr[idx] : Double.NaN;

            // NVI
            double nvi = idx < nviArr.length ? nviArr[idx] : Double.NaN;

            double upperBand = idx < upperBandArr.length ? upperBandArr[idx] : Double.NaN;
            double middleBand = idx < middleBandArr.length ? middleBandArr[idx] : Double.NaN;
            double lowerBand = idx < lowerBandArr.length ? lowerBandArr[idx] : Double.NaN;
            double bandwidth = idx < bandwidthArr.length ? bandwidthArr[idx] : Double.NaN;
            double percentB = idx < percentBArr.length ? percentBArr[idx] : Double.NaN;

            candles.add(new Candle(
                    minute,
                    open, high, low, close,
                    tradeCount,
                    volumeBase,
                    quoteVolume,
                    buyQV,
                    sellQV,
                    deltaUSDT,
                    buyRatio,
                    bidLiq,
                    askLiq,
                    depthImbalance,
                    mid,
                    spread,
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
            idx++;
        }
        return candles;
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
        fList.add((float)  curr.rsi4());
        fList.add((float)  curr.rsi8());
        fList.add((float)  curr.rsi16());

        // MACD
        fList.add((float)  curr.macdVal());
        fList.add((float)  curr.macdSignal());
        fList.add((float)  curr.macdHist());

        // NVI
        fList.add((float)  curr.nvi());

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
        for (int i = 0; i < fList.size(); i++){
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
                1,1,1, 1, 1, 1, 1,1,1,1,1),
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1,1,1,1,1,1)
        ).length + 2; // más 2 por que tiene sumar el feature del símbolo en el que esta y todos los símbolos que puede estar
    }

}