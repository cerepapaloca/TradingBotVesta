package xyz.cereshost.builder;

import ai.djl.util.Pair;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.op.core.Max;
import xyz.cereshost.*;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;

import java.util.*;

import static xyz.cereshost.VestaEngine.LOOK_BACK;

public class BuilderData {


    public static @NotNull Pair<float[][][], float[]> fullBuild(@NotNull List<String> symbols) {
        List<float[][][]> allX = new ArrayList<>();
        List<float[]> allY = new ArrayList<>();

        for (String symbol : symbols) {
            try {
                Vesta.info("Procesando símbolo (Relativo): " + symbol);
                List<Candle> candles = BuilderData.to1mCandles(Vesta.MARKETS.get(symbol));
                candles.sort(Comparator.comparingLong(Candle::openTime));

                // Necesitamos al menos LOOK_BACK + 2 (uno extra para el cálculo relativo inicial)
                if (candles.size() <= LOOK_BACK + 2) {
                    Vesta.error("Símbolo " + symbol + " insuficiente historial.");
                    continue;
                }

                Pair<float[][][], float[]> pair = BuilderData.build(candles, LOOK_BACK);
                float[][][] Xraw = pair.getKey();
                float[] yraw = pair.getValue();

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
        float[] y_final = new float[totalSamples];

        int currentIdx = 0;
        for (int i = 0; i < allX.size(); i++) {
            float[][][] xPart = allX.get(i);
            float[] yPart = allY.get(i);
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
    public static Pair<float[][][], float[]> build(List<Candle> candles, int lookBack) {
        // Empezamos desde 1 porque necesitamos candle[i-1] para calcular el cambio relativo
        int n = candles.size();
        int samples = n - lookBack - 1; // -1 extra por el cálculo de retorno futuro

        if (samples <= 0) return new Pair<>(new float[0][0][0], new float[0]);

        float[][][] X = new float[samples][lookBack][features];
        float[] y = new float[samples];

        // i representa el inicio de la ventana de predicción
        // Empezamos en 1 para garantizar que siempre exista una vela anterior 'prev'
        for (int i = 0; i < samples; i++) {
            // Construir ventana de LookBack
            for (int j = 0; j < lookBack; j++) {
                X[i][j] = extractFeatures(candles.get(i + j + 1), candles.get(i + j));
            }

            // TARGET (Y): Log return de la SIGUIENTE vela (t+1)
            int targetIdx = i + lookBack + 1;
            Candle targetCandle = candles.get(targetIdx);
            Candle lastInputCandle = candles.get(i + lookBack);

            // Predicción: ¿Cuánto cambia el precio logarítmicamente?
            y[i] = (float) Math.log(targetCandle.close() / lastInputCandle.close());
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

        double[] rsi8Arr = FinancialCalculation.computeRSI(closes, 8);
        double[] rsi16Arr = FinancialCalculation.computeRSI(closes, 16);

        // MACD
        FinancialCalculation.MACDResult macdRes = FinancialCalculation.computeMACD(closes, 12, 26, 9);
        double[] macdArr = macdRes.macd();
        double[] signalArr = macdRes.signal();
        double[] histArr = macdRes.histogram();


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
            double rsi8 = idx < rsi8Arr.length ? rsi8Arr[idx] : Double.NaN;
            double rsi16 = idx < rsi16Arr.length ? rsi16Arr[idx] : Double.NaN;

            // MACD
            double macdVal = idx < macdArr.length ? macdArr[idx] : Double.NaN;
            double macdSignal = idx < signalArr.length ? signalArr[idx] : Double.NaN;
            double macdHist = idx < histArr.length ? histArr[idx] : Double.NaN;

            candles.add(new Candle(
                    minute,
                    open, high, low, close,
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
                    rsi8,
                    rsi16,
                    macdVal,
                    macdSignal,
                    macdHist
            ));
            idx++;
        }
        ChartUtils.CandleChartUtils.showCandleChart("Mercado", candles, market.getSymbol());
        return candles;
    }

    private static int features = 0;

    public static float[] extractFeatures(Candle curr, Candle prev) {

        double prevClose = prev.close() <= 0 ? 1.0 : prev.close();
        List<Float> fList = new ArrayList<>();

        // 1-4: Precios relativos (Log Returns)
        fList.add((float) Math.log(curr.open() / prevClose));
        fList.add((float) Math.log(curr.high() / prevClose));
        fList.add((float) Math.log(curr.low() / prevClose));
        fList.add((float) Math.log(curr.close() / prevClose));

        // 5-8: Volúmenes (Log1p)
        fList.add((float) Math.log(curr.volumeBase() / prevClose));
        fList.add((float) Math.log(curr.quoteVolume() / prevClose));
        fList.add((float) Math.log(curr.buyQuoteVolume() / prevClose));
        fList.add((float) Math.log(curr.sellQuoteVolume() / prevClose));

        // 9-10: Delta y Buy Ratio
        double totalVol = curr.buyQuoteVolume() + curr.sellQuoteVolume();
        fList.add((totalVol == 0) ? 0 : (float) (curr.deltaUSDT() / totalVol));
        fList.add((float) curr.buyRatio());

        if (Main.TYPE_DATA.equals(TypeData.ALL)){
            fList.add((float) Math.log1p(curr.bidLiquidity()));
            fList.add((float) Math.log1p(curr.askLiquidity()));

            // 12-14: Métricas de Orderbook relativas
            fList.add((float) curr.depthImbalance());
            fList.add((float) ((curr.midPrice() - curr.close()) / curr.close()));
            fList.add((float) (curr.spread() / curr.close()));
        }

        fList.add((float)  curr.rsi8());
        fList.add((float)  curr.resi16());
        fList.add((float)  curr.macdVal());
        fList.add((float)  curr.macdSignal());
        fList.add((float)  curr.macdHist());

        float[] f = new float[fList.size()];
        for (int i = 0; i < fList.size(); i++){
            f[i] = fList.get(i);
        }
        if (features == 0) features = fList.size();
        return f;
    }

    public static void updateFeatures() {
        extractFeatures(
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1),
                new Candle(
                1,1,1,1,1,1,1,1,1,
                1,1,1,1,1,1,1,1,1,
                1,1,1)
        );
    }

}