package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.List;

import static java.lang.Double.isFinite;
import static java.lang.Math.clamp;
import static xyz.cereshost.vesta.core.utils.StrategyUtils.*;

public class DeltaStrategy implements TradingStrategy {

    private static final int MAX_OPEN_CANDLES = 60;
    private static final int LOOKBACK_CANDLES = 80;
    private static final int PIVOT_LEFT = 2;
    private static final int PIVOT_RIGHT = 2;
    private static final double MIN_PRICE_DELTA_PCT = 0.05;
    private static final double MIN_RSI_DELTA = 1.2;
    private static final double RISK_REWARD = 2.2;

    private static final double MIN_TP_PCT = 0.20;
    private static final double MAX_TP_PCT = 1.50;
    private static final double MIN_SL_PCT = 0.20;
    private static final double MAX_SL_PCT = 1.50;

    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        if (visibleCandles == null || visibleCandles.size() < (PIVOT_LEFT + PIVOT_RIGHT + 6)) return;

        Candle curr = visibleCandles.getLast();
        double close = curr.close();
        if (!isFinite(close) || close <= 0) return;

        Trading.DireccionOperation signal = detectRsiDivergenceSignal(visibleCandles);

        for (Trading.OpenOperation o : operations.getOpens()) {
            if (o.getMinutesOpen() >= MAX_OPEN_CANDLES) {
                operations.close(Trading.ExitReason.TIMEOUT, o);
                continue;
            }

            if (signal == Trading.DireccionOperation.LONG && o.getDireccion() == Trading.DireccionOperation.SHORT) {
                operations.close(Trading.ExitReason.STRATEGY_INVERSION, o);
                continue;
            }
            if (signal == Trading.DireccionOperation.SHORT && o.getDireccion() == Trading.DireccionOperation.LONG) {
                operations.close(Trading.ExitReason.STRATEGY_INVERSION, o);
                continue;
            }
            boolean inversion = o.getFlags().contains("inversion");
            boolean margenTakeProfit = o.getRoiRaw() > o.getSlPercent() || (inversion && o.getRoiRaw() > 0);
            if (o.isUpDireccion()){
                boolean b = isHigh(visibleCandles, inversion ? 15 : 50);
                if (!isPeekClose) isPeekClose = b;
                if (isPeekClose && !b && margenTakeProfit) {
                    isPeekClose = false;
                    o.getFlags().add("inversion");
                    operations.close(Trading.ExitReason.STRATEGY, o);
                }
            }else {
                boolean b = isLow(visibleCandles, inversion ? 15 : 50);
                if (!isPeekClose) isPeekClose = b;
                if (isPeekClose && !b && margenTakeProfit) {
                    isPeekClose = false;
                    o.getFlags().add("inversion");
                    operations.close(Trading.ExitReason.STRATEGY, o);
                }
            }
        }

        if (operations.hasOpenOperation()) return;
        if (signal == Trading.DireccionOperation.NEUTRAL) return;

        double slPercent = calcSlPercent(curr);
        double tpPercent = clamp(slPercent * RISK_REWARD, MIN_TP_PCT, MAX_TP_PCT);

        if (longBan) if (signal == Trading.DireccionOperation.LONG) {
            shortBan = false;
            return;
        }
        if (shortBan)  if (signal == Trading.DireccionOperation.SHORT) {
            longBan = false;
            return;
        }

        operations.open(
                tpPercent,
                slPercent,
                signal,
                operations.getAvailableBalance() / 2,
                4
        );
    }

    private boolean longBan = false;
    private boolean shortBan = false;
    private boolean isPeekClose = false;

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {
        isPeekClose = false;
        Trading.ExitReason reason = closeOperation.getReason();
        if (closeOperation.getOpenOperation().getFlags().contains("inversion") && (reason == Trading.ExitReason.STRATEGY)) {
            Trading.OpenOperation op = operations.open(
                    0.3,
                    0.1,
                    closeOperation.getOpenOperation().isUpDireccion()
                            ? Trading.DireccionOperation.SHORT : Trading.DireccionOperation.LONG,
                    operations.getAvailableBalance() / 2, 4
            );
            if (op != null){
                op.getFlags().add("inversion");
            }
        }else if (reason.isStopLoss()) {
            if (reason.equals(Trading.ExitReason.LONG_STOP_LOSS)){
                longBan = true;
            }
            if (reason.equals(Trading.ExitReason.SHORT_STOP_LOSS)){
                shortBan = true;
            }
        };
    }

    private static Trading.DireccionOperation detectRsiDivergenceSignal(List<Candle> candles) {
        int size = candles.size();
        int from = Math.max(PIVOT_LEFT, size - LOOKBACK_CANDLES);
        int to = size - 1 - PIVOT_RIGHT;
        if (to - from < 4) {
            return Trading.DireccionOperation.NEUTRAL;
        }

        int lowPrev = -1;
        int lowLast = -1;
        int highPrev = -1;
        int highLast = -1;

        for (int i = from; i <= to; i++) {
            if (isPivotLow(candles, i)) {
                lowPrev = lowLast;
                lowLast = i;
            }
            if (isPivotHigh(candles, i)) {
                highPrev = highLast;
                highLast = i;
            }
        }

        boolean bullish = isBullishDivergence(candles, lowPrev, lowLast);
        boolean bearish = isBearishDivergence(candles, highPrev, highLast);

        if (bullish == bearish) {
            return Trading.DireccionOperation.NEUTRAL;
        }
        return bullish ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;
    }

    private static boolean isBullishDivergence(List<Candle> candles, int prevIdx, int lastIdx) {
        if (prevIdx < 0 || lastIdx < 0 || lastIdx <= prevIdx) return false;
        Candle prev = candles.get(prevIdx);
        Candle last = candles.get(lastIdx);

        double price1 = prev.close();
        double price2 = last.close();
        double rsi1 = prev.rsi8();
        double rsi2 = last.rsi8();

        if (!isFinite(price1) || !isFinite(price2) || price1 <= 0 || price2 <= 0) return false;
        if (!isFinite(rsi1) || !isFinite(rsi2)) return false;

        double minPrice2 = price1 * (1.0 - MIN_PRICE_DELTA_PCT / 100.0);
        return price2 < minPrice2 && (rsi2 - rsi1) >= MIN_RSI_DELTA;
    }

    private static boolean isBearishDivergence(List<Candle> candles, int prevIdx, int lastIdx) {
        if (prevIdx < 0 || lastIdx < 0 || lastIdx <= prevIdx) return false;
        Candle prev = candles.get(prevIdx);
        Candle last = candles.get(lastIdx);

        double price1 = prev.close();
        double price2 = last.close();
        double rsi1 = prev.rsi8();
        double rsi2 = last.rsi8();

        if (!isFinite(price1) || !isFinite(price2) || price1 <= 0 || price2 <= 0) return false;
        if (!isFinite(rsi1) || !isFinite(rsi2)) return false;

        double maxPrice2 = price1 * (1.0 + MIN_PRICE_DELTA_PCT / 100.0);
        return price2 > maxPrice2 && (rsi1 - rsi2) >= MIN_RSI_DELTA;
    }

    private static double calcSlPercent(Candle candle) {
        double close = candle.close();
        double atr = candle.atr14();
        double sl;

        if (isFinite(atr) && atr > 0 && isFinite(close) && close > 0) {
            sl = (atr / close) * 100.0;
        } else {
            sl = 0.30;
        }
        return clamp(sl, MIN_SL_PCT, MAX_SL_PCT);
    }

    private static boolean isPivotLow(List<Candle> candles, int idx) {
        double p = candles.get(idx).close();
        if (!isFinite(p)) return false;

        for (int i = idx - PIVOT_LEFT; i <= idx + PIVOT_RIGHT; i++) {
            if (i == idx) continue;
            double x = candles.get(i).close();
            if (!isFinite(x) || p >= x) return false;
        }
        return true;
    }

    private static boolean isPivotHigh(List<Candle> candles, int idx) {
        double p = candles.get(idx).close();
        if (!isFinite(p)) return false;

        for (int i = idx - PIVOT_LEFT; i <= idx + PIVOT_RIGHT; i++) {
            if (i == idx) continue;
            double x = candles.get(i).close();
            if (!isFinite(x) || p <= x) return false;
        }
        return true;
    }
}
