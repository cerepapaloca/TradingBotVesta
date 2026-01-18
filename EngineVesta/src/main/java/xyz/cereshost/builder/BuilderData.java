package xyz.cereshost.builder;

import ai.djl.util.Pair;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.ChartUtils;
import xyz.cereshost.EngineUtils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;

import java.util.*;
import java.util.stream.Stream;

import static xyz.cereshost.VestaEngine.LOOK_BACK;

public class BuilderData {

    public static @NotNull Pair<float[][][], float[]> fullBuild(@NotNull List<String> symbols) {
        // Combinar datos de todos los símbolos
        List<float[][][]> allX = new ArrayList<>();
        List<float[]> allY = new ArrayList<>(); // Cambiado: float[][] -> float[]

        for (String symbol : symbols) {
            try {
                Vesta.info("Procesando símbolo: " + symbol);
                List<Candle> candles = BuilderData.to1mCandles(Vesta.MARKETS.get(symbol));
                candles.sort(Comparator.comparingLong(Candle::openTime));
                if (candles.size() <= LOOK_BACK + 1) {
                    Vesta.error("Símbolo " + symbol + " no tiene suficientes velas: " + candles.size());
                    continue;
                }

                Pair<float[][][], float[]> pair = BuilderData.build(candles, LOOK_BACK); // Cambiado
                float[][][] Xraw = pair.getKey();
                float[] yraw = pair.getValue(); // Cambiado

                if (Xraw.length > 0) {
                    // Añadir símbolo como característica adicional
                    float[][][] XwithSymbol = EngineUtils.addSymbolFeature(Xraw, symbol, symbols);
                    allX.add(XwithSymbol);
                    allY.add(yraw); // Cambiado
                    Vesta.info("Añadidas " + Xraw.length + " muestras");
                }
                ChartUtils.CandleChartUtils.showCandleChart("Datos Originales", candles, symbol);
            } catch (Exception e) {
                Vesta.error("Error procesando símbolo " + symbol + ": " + e.getMessage());
            }
        }

        if (allX.isEmpty()) {
            throw new RuntimeException("No hay datos suficientes de ningún símbolo");
        }

        return EngineUtils.combineDatasets(allX, allY);
    }

    public static Pair<float[][][], float[]> build(List<Candle> candles, int lookback) {
        if (candles == null) {
            throw new IllegalArgumentException("candles == null");
        }
        int n = candles.size();
        if (n <= lookback) {
            return new Pair<>(new float[0][][], new float[0]);  // Cambiado: float[0][] -> float[0]
        }

        List<float[][]> X = new ArrayList<>();
        List<Float> y = new ArrayList<>();  // Cambiado: float[] -> Float

        final int NUM_FEATURES = 8;

        for (int i = lookback; i < n; i++) {
            // construir ventana: índices [i-lookback, ..., i-1]
            float[][] seq = new float[lookback][NUM_FEATURES];
            boolean skip = false;

            for (int j = 0; j < lookback; j++) {
                Candle c = candles.get(i - lookback + j);
                double[] f = extractFeatures(c);
                if (f.length != NUM_FEATURES) {
                    skip = true;
                    break;
                }
                for (int k = 0; k < NUM_FEATURES; k++) {
                    double v = f[k];
                    if (Double.isNaN(v) || Double.isInfinite(v)) {
                        skip = true;
                        break;
                    }
                    seq[j][k] = (float) v;
                }
                if (skip) break;
            }
            if (skip) continue;

            // Target: Retorno Logarítmico
            Candle targetC = candles.get(i);
            Candle prevC = candles.get(i - 1); // La última vela de tu ventana de entrada

            double currentClose = targetC.close();
            double prevClose = prevC.close();

            // Evitar log(0) o división por cero
            if (prevClose <= 0 || currentClose <= 0) continue;

            // Usamos Log Return porque es simétrico y sumable
            double logReturn = Math.log(currentClose / prevClose);

            y.add((float) logReturn);
            X.add(seq);
        }

        float[][][] Xarr = X.toArray(new float[0][][]);

        // Convertir List<Float> a float[]
        float[] yarr = new float[y.size()];
        for (int i = 0; i < y.size(); i++) {
            yarr[i] = y.get(i);
        }

        return new Pair<>(Xarr, yarr);
    }

    @Contract("_ -> new")
    public static double @NotNull [] extractFeatures(@NotNull Candle c) {
        return new double[]{
                c.close(),
                c.quoteVolume(),
                c.deltaUSDT(),
                c.buyRatio(),
                c.bidLiquidity(),
                c.askLiquidity(),
                c.depthImbalance(),
                c.spread()
        };
    }

    // Este método ya no es necesario, pero lo dejo por compatibilidad
    @Contract("_ -> new")
    public static double @NotNull [] extractTarget(@NotNull Candle next) {
        return new double[]{next.close()};  // Solo el cierre
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

        return candles;
    }

}