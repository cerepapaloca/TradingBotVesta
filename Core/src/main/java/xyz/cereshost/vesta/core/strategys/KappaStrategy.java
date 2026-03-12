package xyz.cereshost.vesta.core.strategys;

import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.time.Instant;
import java.time.LocalTime;
import java.time.ZoneOffset;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

public class KappaStrategy implements TradingStrategy {

    private static final LocalTime SESSION_START_UTC = LocalTime.of(13, 0);
    private static final LocalTime SESSION_END_UTC = LocalTime.of(14, 0);
    private static final long MILLIS_PER_DAY = 86_400_000L;

    private static final int EMA_LENGTH = 150;
    private static final int DMI_LENGTH = 10;
    private static final int ADX_SMOOTHING = 16;
    private static final double ADX_THRESHOLD = 10;
    private static final double RSI_LONG_THRESHOLD = 50.0;
    private static final double RSI_SHORT_THRESHOLD = 40.0;

    private static final double TP_POINTS = 3;
    private static final double SL_POINTS = 2;
    private static final double TICKS_PER_POINT = 19;
    private static final double PRICE_TICK_SIZE = 0.01;

    private static final boolean USE_TRAILING_STOP = true;
    private static final double TRAIL_POINTS_MULTIPLIER = 0.55;
    private static final double TRAIL_OFFSET_TICKS = 50.0;

    private static final int LEVERAGE = 4;
    private static final double ORDER_BALANCE_FRACTION = 1.0;
    private static final double MIN_ORDER_NOTIONAL = 8.0;

    private Double sessionHigh;
    private Double sessionLow;
    private boolean previousInSession;
    private long previousDay = Long.MIN_VALUE;
    private double previousClose = Double.NaN;
    private double previousSessionHigh = Double.NaN;
    private double previousSessionLow = Double.NaN;

    private final Map<UUID, TrailingState> trailingStates = new ConcurrentHashMap<>();

    @Override
    public void executeStrategy(PredictionEngine.PredictionResult pred, List<Candle> visibleCandles, TradingManager operations) {
        if (visibleCandles == null || visibleCandles.size() < 2) {
            return;
        }

        Candle current = visibleCandles.getLast();
        long currentDay = Math.floorDiv(current.openTime(), MILLIS_PER_DAY);
        if (previousDay != Long.MIN_VALUE && currentDay != previousDay) {
            sessionHigh = null;
            sessionLow = null;
        }

        boolean inSession = isInSessionUtc(current.openTime());
        boolean isSessionStart = inSession && !previousInSession;
        if (inSession) {
            if (isSessionStart || sessionHigh == null || sessionLow == null) {
                sessionHigh = current.high();
                sessionLow = current.low();
            } else {
                sessionHigh = Math.max(sessionHigh, current.high());
                sessionLow = Math.min(sessionLow, current.low());
            }
        }

        syncTrailingState(operations);
        if (USE_TRAILING_STOP) {
            applyTrailingStop(operations, current);
        }

        double ema = computeLatestEma(visibleCandles, EMA_LENGTH);
        double rsi = current.rsi8();
        double adx = computeLatestAdx(visibleCandles, DMI_LENGTH, ADX_SMOOTHING);

        boolean canTrade = !inSession
                && sessionHigh != null
                && sessionLow != null
                && Double.isFinite(previousClose)
                && Double.isFinite(previousSessionHigh)
                && Double.isFinite(previousSessionLow)
                && Double.isFinite(ema)
                && Double.isFinite(rsi)
                && Double.isFinite(adx);

        boolean crossHigh = canTrade && crossover(previousClose, previousSessionHigh, current.close(), sessionHigh);
        boolean crossLow = canTrade && crossunder(previousClose, previousSessionLow, current.close(), sessionLow);

        boolean buyCondition = crossHigh
                && current.close() > ema
                && adx > ADX_THRESHOLD
                && rsi > RSI_LONG_THRESHOLD;

        boolean sellCondition = crossLow
                && current.close() < ema
                && adx > ADX_THRESHOLD
                && rsi < RSI_SHORT_THRESHOLD;

        if (!operations.hasOpenOperation()) {
            if (buyCondition) {
                openOperation(operations, DireccionOperation.LONG, current.close());
            } else if (sellCondition) {
                openOperation(operations, DireccionOperation.SHORT, current.close());
            }
        }

        previousDay = currentDay;
        previousInSession = inSession;
        previousClose = current.close();
        previousSessionHigh = sessionHigh != null ? sessionHigh : Double.NaN;
        previousSessionLow = sessionLow != null ? sessionLow : Double.NaN;
    }

    @Override
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {
        trailingStates.remove(closeOperation.getUuid());
    }

    private void openOperation(TradingManager operations, DireccionOperation direction, double entryPrice) {
        if (!Double.isFinite(entryPrice) || entryPrice <= 0.0) {
            return;
        }

        double availableBalance = operations.getAvailableBalance();
        if (!Double.isFinite(availableBalance) || availableBalance <= 0.0) {
            return;
        }

        double minMarginUsdt = MIN_ORDER_NOTIONAL / Math.max(LEVERAGE, 1);
        double amountUsdt = Math.max(availableBalance * ORDER_BALANCE_FRACTION, minMarginUsdt);
        amountUsdt = Math.min(amountUsdt, availableBalance);
        if (amountUsdt * LEVERAGE < MIN_ORDER_NOTIONAL) {
            operations.log("Balance insuficiente para abrir KappaStrategy");
            return;
        }

        double tpTicks = ((TP_POINTS/100) * entryPrice) * TICKS_PER_POINT;
        double slTicks = ((SL_POINTS/100) * entryPrice) * TICKS_PER_POINT;
        double tpPercent = priceDistanceToPercent(entryPrice, ticksToPrice(tpTicks));
        double slPercent = priceDistanceToPercent(entryPrice, ticksToPrice(slTicks));

        TradingManager.OpenOperation open = operations.open(tpPercent, slPercent, direction, amountUsdt, LEVERAGE);
        if (open != null) {
            trailingStates.put(open.getUuid(), new TrailingState(open.getEntryPrice()));
        }
    }

    private void applyTrailingStop(TradingManager operations, Candle current) {
        double trailDistancePrice = ticksToPrice((TP_POINTS * TICKS_PER_POINT) * TRAIL_POINTS_MULTIPLIER);
        double trailOffsetPrice = ticksToPrice(TRAIL_OFFSET_TICKS);
        if (!Double.isFinite(trailDistancePrice) || !Double.isFinite(trailOffsetPrice)
                || trailDistancePrice <= 0.0 || trailOffsetPrice <= 0.0) {
            return;
        }

        for (TradingManager.OpenOperation open : operations.getOpens()) {
            TrailingState state = trailingStates.computeIfAbsent(open.getUuid(), id -> new TrailingState(open.getEntryPrice()));
            if (open.isUpDireccion()) {
                state.highestPrice = Math.max(state.highestPrice, current.high());
                if (!state.activated && state.highestPrice >= (open.getEntryPrice() + trailOffsetPrice)) {
                    state.activated = true;
                }
                if (!state.activated) {
                    continue;
                }

                double trailStopPrice = state.highestPrice - trailDistancePrice;
                if (!Double.isFinite(trailStopPrice)) {
                    continue;
                }
                if (trailStopPrice > open.getSlPrice()) {
                    double newSlPercent = ((open.getEntryPrice() - trailStopPrice) / open.getEntryPrice()) * 100.0;
                    if (Double.isFinite(newSlPercent)) {
                        open.setSlPercent(newSlPercent);
                    }
                }
            } else {
                state.lowestPrice = Math.min(state.lowestPrice, current.low());
                if (!state.activated && state.lowestPrice <= (open.getEntryPrice() - trailOffsetPrice)) {
                    state.activated = true;
                }
                if (!state.activated) {
                    continue;
                }

                double trailStopPrice = state.lowestPrice + trailDistancePrice;
                if (!Double.isFinite(trailStopPrice)) {
                    continue;
                }
                if (trailStopPrice < open.getSlPrice()) {
                    double newSlPercent = ((trailStopPrice - open.getEntryPrice()) / open.getEntryPrice()) * 100.0;
                    if (Double.isFinite(newSlPercent)) {
                        open.setSlPercent(newSlPercent);
                    }
                }
            }
        }
    }

    private void syncTrailingState(TradingManager operations) {
        Set<UUID> active = new HashSet<>();
        for (TradingManager.OpenOperation open : operations.getOpens()) {
            active.add(open.getUuid());
            trailingStates.computeIfAbsent(open.getUuid(), id -> new TrailingState(open.getEntryPrice()));
        }
        trailingStates.keySet().removeIf(uuid -> !active.contains(uuid));
    }

    private static boolean isInSessionUtc(long openTimeMillis) {
        LocalTime time = Instant.ofEpochMilli(openTimeMillis).atZone(ZoneOffset.UTC).toLocalTime();
        return !time.isBefore(SESSION_START_UTC) && time.isBefore(SESSION_END_UTC);
    }

    private static boolean crossover(double prevA, double prevB, double currA, double currB) {
        return prevA <= prevB && currA > currB;
    }

    private static boolean crossunder(double prevA, double prevB, double currA, double currB) {
        return prevA >= prevB && currA < currB;
    }

    private static double ticksToPrice(double ticks) {
        return ticks * PRICE_TICK_SIZE;
    }

    private static double priceDistanceToPercent(double referencePrice, double priceDistance) {
        if (!Double.isFinite(referencePrice) || referencePrice <= 0.0 || !Double.isFinite(priceDistance) || priceDistance <= 0.0) {
            return Double.NaN;
        }
        return (priceDistance / referencePrice) * 100.0;
    }

    private static double computeLatestEma(List<Candle> candles, int period) {
        if (candles == null || period <= 0 || candles.size() < period) {
            return Double.NaN;
        }

        double sum = 0.0;
        for (int i = 0; i < period; i++) {
            sum += candles.get(i).close();
        }

        double ema = sum / period;
        double alpha = 2.0 / (period + 1.0);
        for (int i = period; i < candles.size(); i++) {
            double close = candles.get(i).close();
            ema = alpha * (close - ema) + ema;
        }
        return ema;
    }

    private static double computeLatestRsi(List<Candle> candles, int period) {
        if (candles == null || period <= 0 || candles.size() <= period) {
            return Double.NaN;
        }

        double gains = 0.0;
        double losses = 0.0;
        for (int i = 1; i <= period; i++) {
            double delta = candles.get(i).close() - candles.get(i - 1).close();
            if (delta >= 0) {
                gains += delta;
            } else {
                losses -= delta;
            }
        }

        double avgGain = gains / period;
        double avgLoss = losses / period;
        for (int i = period + 1; i < candles.size(); i++) {
            double delta = candles.get(i).close() - candles.get(i - 1).close();
            double gain = Math.max(delta, 0.0);
            double loss = Math.max(-delta, 0.0);
            avgGain = ((avgGain * (period - 1)) + gain) / period;
            avgLoss = ((avgLoss * (period - 1)) + loss) / period;
        }

        if (avgLoss == 0.0) {
            return avgGain == 0.0 ? 50.0 : 100.0;
        }
        double rs = avgGain / avgLoss;
        return 100.0 - (100.0 / (1.0 + rs));
    }

    private static double computeLatestAdx(List<Candle> candles, int period, int adxSmoothing) {
        if (candles == null || period <= 0 || adxSmoothing <= 0 || candles.size() <= (period + adxSmoothing)) {
            return Double.NaN;
        }

        int n = candles.size();
        double[] tr = new double[n];
        double[] plusDm = new double[n];
        double[] minusDm = new double[n];

        for (int i = 1; i < n; i++) {
            Candle prev = candles.get(i - 1);
            Candle curr = candles.get(i);

            double upMove = curr.high() - prev.high();
            double downMove = prev.low() - curr.low();
            plusDm[i] = (upMove > downMove && upMove > 0.0) ? upMove : 0.0;
            minusDm[i] = (downMove > upMove && downMove > 0.0) ? downMove : 0.0;

            double range1 = curr.high() - curr.low();
            double range2 = Math.abs(curr.high() - prev.close());
            double range3 = Math.abs(curr.low() - prev.close());
            tr[i] = Math.max(range1, Math.max(range2, range3));
        }

        if (n <= period) {
            return Double.NaN;
        }

        double smTr = 0.0;
        double smPlusDm = 0.0;
        double smMinusDm = 0.0;
        for (int i = 1; i <= period; i++) {
            smTr += tr[i];
            smPlusDm += plusDm[i];
            smMinusDm += minusDm[i];
        }

        double[] dx = new double[n];
        for (int i = period; i < n; i++) {
            if (i > period) {
                smTr = smTr - (smTr / period) + tr[i];
                smPlusDm = smPlusDm - (smPlusDm / period) + plusDm[i];
                smMinusDm = smMinusDm - (smMinusDm / period) + minusDm[i];
            }

            if (smTr <= 0.0) {
                dx[i] = 0.0;
                continue;
            }
            double plusDi = 100.0 * (smPlusDm / smTr);
            double minusDi = 100.0 * (smMinusDm / smTr);
            double diSum = plusDi + minusDi;
            dx[i] = diSum == 0.0 ? 0.0 : 100.0 * (Math.abs(plusDi - minusDi) / diSum);
        }

        int firstAdxIndex = period + adxSmoothing - 1;
        if (firstAdxIndex >= n) {
            return Double.NaN;
        }

        double adx = 0.0;
        for (int i = period; i <= firstAdxIndex; i++) {
            adx += dx[i];
        }
        adx /= adxSmoothing;

        for (int i = firstAdxIndex + 1; i < n; i++) {
            adx = ((adx * (adxSmoothing - 1)) + dx[i]) / adxSmoothing;
        }
        return adx;
    }

    private static final class TrailingState {
        private double highestPrice;
        private double lowestPrice;
        private boolean activated;

        private TrailingState(double entryPrice) {
            this.highestPrice = entryPrice;
            this.lowestPrice = entryPrice;
            this.activated = false;
        }
    }
}
