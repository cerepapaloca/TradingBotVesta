package xyz.cereshost.builder;

import ai.djl.util.Pair;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;

import java.util.*;

public class DatasetBuilder {

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

            // target = SOLO el cierre de la vela i
            Candle targetC = candles.get(i);
            double close = targetC.close();

            if (Double.isNaN(close)) continue;

            // Opcional: convertir target a retorno relativo
            // double prevClose = candles.get(i-1).close();
            // double returnValue = (close - prevClose) / prevClose;
            // y.add((float) returnValue);

            // Por defecto: target en valor absoluto (precio de cierre)
            y.add((float) close);
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
        // ... (código sin cambios) ...
        // Agrupar trades por minuto (tree map para iterar en orden)
        NavigableMap<Long, List<Trade>> tradesByMinute = new TreeMap<>();
        for (Trade t : market.getTrades()) {
            long minute = (t.time() / 60_000) * 60_000;
            tradesByMinute.computeIfAbsent(minute, k -> new ArrayList<>()).add(t);
        }

        // Indexar tickMarkers por minuto en un TreeMap para floorKey / ceilingKey
        NavigableMap<Long, TickMarket> tickByMinute = new TreeMap<>();
        for (TickMarket tm : market.getTickMarkers()) {
            long minute = (tm.getDepth().getDate() / 60_000) * 60_000;
            tickByMinute.put(minute, tm);
        }

        List<Candle> candles = new ArrayList<>();

        if (tradesByMinute.isEmpty()) {
            // no trades -> devolver lista vacía
            return candles;
        }

        // Rango completo de minutos (desde el primer trade hasta el último trade)
        long startMinute = tradesByMinute.firstKey();
        long endMinute = tradesByMinute.lastKey();

        // diagnostico opcional
        Vesta.info("to1mCandles: startMinute=" + startMinute + " endMinute=" + endMinute
                + " minutes=" + ((endMinute - startMinute) / 60000 + 1)
                + " tradesMinutes=" + tradesByMinute.size()
                + " tickSnapshots=" + tickByMinute.size());

        double lastClose = Double.NaN; // para forward-fill si no hay trades en un minuto

        // recorrer EVERY minute entre start y end
        for (long minute = startMinute; minute <= endMinute; minute += 60_000L) {
            List<Trade> trades = tradesByMinute.get(minute);

            double open, high, low, close;
            double volumeBase = 0;
            double quoteVolume = 0;
            double buyQV = 0;
            double sellQV = 0;

            if (trades == null || trades.isEmpty()) {
                // minuto SIN trades -> crear vela "vacía" usando lastClose si existe
                if (Double.isNaN(lastClose)) {
                    // si no hay lastClose (primeros minutos sin trades) no podemos crear OHLC real:
                    // opcional: saltar o usar 0. Aquí uso 0 para evitar NaNs.
                    open = high = low = close = 0.0;
                } else {
                    open = high = low = close = lastClose;
                }
                // volúmenes quedan en 0
            } else {
                // hay trades -> ordenar por time para OHLC
                trades.sort(Comparator.comparingLong(Trade::time));
                open = trades.get(0).price();
                close = trades.get(trades.size() - 1).price();
                high = trades.stream().mapToDouble(Trade::price).max().orElse(open);
                low = trades.stream().mapToDouble(Trade::price).min().orElse(open);

                for (Trade t : trades) {
                    double q = t.quoteQty();
                    volumeBase += t.qty();
                    quoteVolume += q;
                    if (t.isBuyerMaker()) sellQV += q; // sale agresiva
                    else buyQV += q; // compra agresiva
                }
                lastClose = close;
            }

            double deltaUSDT = buyQV - sellQV;
            double buyRatio = quoteVolume == 0.0 ? 0.0 : buyQV / quoteVolume;

            // Depth: usa el snapshot más cercano por floorKey(minute)
            double bidLiq = 0, askLiq = 0, mid = close, spread = 0;
            Map.Entry<Long, TickMarket> floor = tickByMinute.floorEntry(minute);
            TickMarket tick = (floor != null) ? floor.getValue() : null;

            if (tick != null && tick.getDepth() != null) {
                Depth depth = tick.getDepth();
                // sumar liq (por ejemplo primeros 10 niveles)
                bidLiq = depth.getBids().stream().mapToDouble(o -> o.price() * o.qty()).sum();
                askLiq = depth.getAsks().stream().mapToDouble(o -> o.price() * o.qty()).sum();
                if (!depth.getBids().isEmpty() && !depth.getAsks().isEmpty()) {
                    double bestBid = depth.getBids().peekFirst().price();
                    double bestAsk = depth.getAsks().peekFirst().price();
                    mid = (bestBid + bestAsk) / 2.0;
                    spread = bestAsk - bestBid;
                }
            }

            double depthImbalance = (bidLiq + askLiq == 0) ? 0 : (bidLiq - askLiq) / (bidLiq + askLiq);

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