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

        for (int i = 0; i < samples; i++) {
            // 1. Construir ventana de LookBack (Entrada del modelo)
            for (int j = 0; j < lookBack; j++) {
                X[i][j] = extractFeatures(candles.get(i + j + 1), candles.get(i + j));
            }

            // El precio de entrada para el cálculo de retorno es el cierre de la última vela del input
            double entryPrice = candles.get(i + lookBack).close();

            // 2. Escanear ventana futura (X velas) para los Targets
            double bestPriceLong = -Double.MAX_VALUE;  // Cierre más alto
            double worstPriceLong = Double.MAX_VALUE;  // Mecha más baja

            double bestPriceShort = Double.MAX_VALUE;  // Cierre más bajo
            double worstPriceShort = -Double.MAX_VALUE; // Mecha más alta

            for (int f = 1; f <= futureWindow; f++) {
                Candle future = candles.get(i + lookBack + f);

                // Para LONG
                bestPriceLong = Math.max(bestPriceLong, future.close());
                worstPriceLong = Math.min(worstPriceLong, future.low());

                // Para SHORT
                bestPriceShort = Math.min(bestPriceShort, future.close());
                worstPriceShort = Math.max(worstPriceShort, future.high());
            }

            // 3. Determinar Dirección Real dominante en esa ventana
            // Usamos el cierre de la última vela de la ventana vs entrada
            // 3. Calcular el Retorno Total de la ventana (Cierre Final vs Entrada)
            double finalCloseInWindow = candles.get(i + lookBack + futureWindow).close();

            // Usamos log return para ser consistentes con el resto de la lógica
            double totalMovementLog = Math.log(finalCloseInWindow / entryPrice);

            double logTP;
            double logSL;

            if (totalMovementLog > PredictionEngine.THRESHOLD) {
                // Alcista
                logTP = Math.log(bestPriceLong / entryPrice);
                logSL = Math.log(entryPrice / worstPriceLong);

                y[i][0] = (float) Math.abs(logTP);
                y[i][1] = (float) Math.abs(logSL);
                y[i][2] = 1.0f; // Clase 1: LONG

            } else if (totalMovementLog < -PredictionEngine.THRESHOLD) {
                // Bajista
                logTP = Math.log(entryPrice / bestPriceShort);
                logSL = Math.log(worstPriceShort / entryPrice);

                y[i][0] = (float) Math.abs(logTP);
                y[i][1] = (float) Math.abs(logSL);
                y[i][2] = -1.0f; // Clase -1: SHORT

            } else {
                // Lateral
                double volatilityUp = Math.abs(Math.log(bestPriceLong / entryPrice));
                double volatilityDown = Math.abs(Math.log(worstPriceLong / entryPrice));

                y[i][0] = (float) volatilityUp;   // TP potencial (aunque no operemos)
                y[i][1] = (float) volatilityDown; // SL potencial (riesgo)
                y[i][2] = 0.0f; // Clase 0: NEUTRAL
            }

            // Limpieza de seguridad
//            for (int k = 0; k < 2; k++) {
//                if (Float.isNaN(y[i][k]) || Float.isInfinite(y[i][k]) || y[i][k] < 0) {
//                    y[i][k] = 0.000f;
//                }
//            }
        }
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

        double[] nviArr = FinancialCalculation.computeNVI(closes, simpleByMinute.values().stream().map(c -> c.volumen().quoteVolume()).toList(), 1000.0);

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
                    nvi
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
                1,1,1, 1, 1, 1),
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1, 1, 1, 1)
        ).length + 2; // más 2 por que tiene sumar el feature del símbolo en el que esta y todos los símbolos que puede estar
    }

}