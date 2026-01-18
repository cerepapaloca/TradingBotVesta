package xyz.cereshost.builder;

import ai.djl.util.Pair;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.tensorflow.op.core.Max;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.EngineUtils;
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

        int features = 17;
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

            double depthImbalance =
                    (bidLiq + askLiq == 0) ? 0 : (bidLiq - askLiq) / (bidLiq + askLiq);

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
                    spread
            ));
        }
        ChartUtils.CandleChartUtils.showCandleChart("Mercado", candles, market.getSymbol());
        return candles;
    }

    public static float[] extractFeatures(Candle curr, Candle prev) {
        float[] f = new float[17];
        double prevClose = prev.close() <= 0 ? 1.0 : prev.close();

        // 1-4: Precios relativos (Log Returns)
        f[0] = (float) Math.log(curr.open() / prevClose);
        f[1] = (float) Math.log(curr.high() / prevClose);
        f[2] = (float) Math.log(curr.low() / prevClose);
        f[3] = (float) Math.log(curr.close() / prevClose);

        // 5-8: Volúmenes (Log1p)
        f[4] = (float) Math.log1p(curr.volumeBase());
        f[5] = (float) Math.log1p(curr.quoteVolume());
        f[6] = (float) Math.log1p(curr.buyQuoteVolume());
        f[7] = (float) Math.log1p(curr.sellQuoteVolume());

        // 9-10: Delta y Buy Ratio
        double totalVol = curr.buyQuoteVolume() + curr.sellQuoteVolume();
        f[8] = (totalVol == 0) ? 0 : (float) (curr.deltaUSDT() / totalVol);
        f[9] = (float) curr.buyRatio();

        // 11-12: Placeholders (Para futuros indicadores como RSI o EMA)
        f[10] = 0;
        f[11] = 0;

        // 13-14: Liquidez (Log1p)
        f[12] = (float) Math.log1p(curr.bidLiquidity());
        f[13] = (float) Math.log1p(curr.askLiquidity());

        // 15-17: Métricas de Orderbook relativas
        f[14] = (float) curr.depthImbalance();
        f[15] = (float) ((curr.midPrice() - curr.close()) / curr.close());
        f[16] = (float) (curr.spread() / curr.close());

        return f;
    }

}