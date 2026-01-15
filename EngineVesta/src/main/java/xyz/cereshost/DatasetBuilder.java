package xyz.cereshost;

import ai.djl.util.Pair;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.market.*;

import java.util.*;

import static xyz.cereshost.Main.extractFeatures;
import static xyz.cereshost.Main.extractTarget;

public class DatasetBuilder {

    public static Pair<float[][][], float[][]> build(
            List<Candle> candles,
            int lookback
    ) {
        List<float[][]> X = new ArrayList<>();
        List<float[]> y = new ArrayList<>();

        for (int i = lookback; i < candles.size() - 1; i++) {

            float[][] seq = new float[lookback][8];

            for (int j = 0; j < lookback; j++) {
                double[] f = extractFeatures(candles.get(i - lookback + j));
                for (int k = 0; k < f.length; k++) {
                    seq[j][k] = (float) f[k];
                }
            }

            double[] t = extractTarget(candles.get(i + 1));
            y.add(new float[]{(float) t[0], (float) t[1]});
            X.add(seq);
        }

        return new Pair<>(
                X.toArray(new float[0][][]),
                y.toArray(new float[0][])
        );
    }

    public static @NotNull List<Candle> to1mCandles(@NotNull Market market) {
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
        System.out.println("to1mCandles: startMinute=" + startMinute + " endMinute=" + endMinute
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

