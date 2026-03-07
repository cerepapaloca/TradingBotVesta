package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;
import xyz.cereshost.vesta.core.utils.StrategyUtils;

import java.util.List;

import static xyz.cereshost.vesta.core.utils.StrategyUtils.isHigh;
import static xyz.cereshost.vesta.core.utils.StrategyUtils.isLow;

public class EpsilonStrategy implements TradingStrategy {

    private static final int MIN_VISIBLE_CANDLES = 140;
    private static final int MAX_MINUTE_OPEN = 90;

    private static final int PIVOT_LEFT = 2;
    private static final int PIVOT_RIGHT = 2;
    private static final int STRUCTURE_LOOKBACK = 200;
    private static final double MIN_BREAK_PERCENT = 0.06;

    private static final double ATR_MULTIPLIER = 1.15;
    private static final double RISK_REWARD = 6;
    private static final double MIN_SL_PERCENT = 0.30;
    private static final double MAX_SL_PERCENT = 2.0;
    private static final double MIN_TP_PERCENT = 0.40;
    private static final double MAX_TP_PERCENT = 3.80;

    private static final int STOP_LOSS_COOLDOWN_CANDLES = 16;
    private static final int INVERSION_COOLDOWN_CANDLES = 8;

    private long lastTriggeredPivotTime = -1L;
    private int longCooldown = 0;
    private int shortCooldown = 0;

    private boolean isPeekClose = false;
    @NotNull
    private Trading.DireccionOperation direccionOperationWindow = Trading.DireccionOperation.NEUTRAL;
    private int candlesValid;

    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        if (visibleCandles == null || visibleCandles.size() < MIN_VISIBLE_CANDLES) {
            return;
        }

        tickCooldowns();
        Candle curr = visibleCandles.getLast();
        Candle prev = visibleCandles.get(visibleCandles.size() - 2);
        StrategyUtils.BosChochSignal signal = StrategyUtils.detectBosChoch(
                visibleCandles,
                PIVOT_LEFT,
                PIVOT_RIGHT,
                STRUCTURE_LOOKBACK,
                MIN_BREAK_PERCENT
        );

        for (Trading.OpenOperation open : operations.getOpens()) {
            if (open.getMinutesOpen() >= MAX_MINUTE_OPEN) {
                operations.close(Trading.ExitReason.TIMEOUT, open);
                continue;
            }
//            if (shouldCloseByInversion(open, signal)) {
//                open.getFlags().add("inversion");
//                operations.close(Trading.ExitReason.STRATEGY_INVERSION, open);
//            }
            boolean inversion = open.getFlags().contains("inversion");
            boolean margenTakeProfit = open.getRoiRaw() > open.getTpPercent()/2 || (inversion && open.getRoiRaw() > 0);
            if (open.isUpDireccion()){
                boolean b = isHigh(visibleCandles, inversion ? 15 : 60);
                if (!isPeekClose) isPeekClose = b;
                if (isPeekClose && !b && margenTakeProfit) {
                    isPeekClose = false;
                    open.getFlags().add("inversion");
                    operations.close(Trading.ExitReason.STRATEGY, open);
                }
            }else {
                boolean b = isLow(visibleCandles, inversion ? 15 : 60);
                if (!isPeekClose) isPeekClose = b;
                if (isPeekClose && !b && margenTakeProfit) {
                    isPeekClose = false;
                    open.getFlags().add("inversion");
                    operations.close(Trading.ExitReason.STRATEGY, open);
                }
            }
        }

        boolean crossUp = (prev.macdVal() <= prev.macdSignal() && curr.macdVal() > curr.macdSignal());
        boolean crossDown = prev.macdVal() >= prev.macdSignal() && curr.macdVal() < curr.macdSignal();
        boolean validMACD = crossUp || crossDown;

        if (operations.hasOpenOperation() || !signal.valid()) {
            System.out.println("A "  + operations.hasOpenOperation());
            return;
        }

        if (!validMACD) {
            return;
        }

//        Trading.DireccionOperation dirMACD;
//        if (candlesValid != 0){
//            candlesValid--;
//        }else {
//            direccionOperationWindow = Trading.DireccionOperation.NEUTRAL;
//        }
//
//        if (!validMACD) {
//            if (direccionOperationWindow ==  Trading.DireccionOperation.NEUTRAL) {
//                operations.log("MACD no cumple con la condición para operar");
//                return;
//            }else {
//                dirMACD = direccionOperationWindow;
//            }
//        }else {
//            candlesValid = 4;
//            dirMACD = crossDown ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;
//        }




        Trading.DireccionOperation direction = signal.direction() > 0
                ? Trading.DireccionOperation.LONG
                : Trading.DireccionOperation.SHORT;

        if (direction == Trading.DireccionOperation.LONG && prev.macdHist() > -0.0012) {
            return;
        }
        if (direction == Trading.DireccionOperation.SHORT && prev.macdHist() < 0.0012) {
            return;
        }

        if (isDirectionInCooldown(direction)) {
            return;
        }
        if (signal.pivotIndex() < 0 || signal.pivotIndex() >= visibleCandles.size()) {
            return;
        }
        long pivotTime = visibleCandles.get(signal.pivotIndex()).openTime();
        if (pivotTime == lastTriggeredPivotTime) {
            return;
        }

        Candle current = visibleCandles.getLast();
//        if (!passesFilters(current, direction)) {
//            return;
//        }

        double slPercent = calculateSlPercent(current);
        double tpPercent = calculateTpPercent(slPercent, signal);

        Trading.OpenOperation open = operations.open(
                tpPercent,
                slPercent,
                direction,
                operations.getAvailableBalance() / 2.0,
                4
        );
        if (open != null) {
            lastTriggeredPivotTime = pivotTime;
            open.getFlags().add(signal.choch() ? "choch" : "bos");
        }

    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {
        isPeekClose = false;
        Trading.OpenOperation open = closeOperation.getOpenOperation();
        Trading.ExitReason reason = closeOperation.getReason();

        if (reason == Trading.ExitReason.LONG_STOP_LOSS) {
            longCooldown = STOP_LOSS_COOLDOWN_CANDLES;
        } else if (reason == Trading.ExitReason.SHORT_STOP_LOSS) {
            shortCooldown = STOP_LOSS_COOLDOWN_CANDLES;
        } else if (reason == Trading.ExitReason.STRATEGY_INVERSION) {
            if (open.isUpDireccion()) {
                longCooldown = Math.max(longCooldown, INVERSION_COOLDOWN_CANDLES);
            } else {
                shortCooldown = Math.max(shortCooldown, INVERSION_COOLDOWN_CANDLES);
            }
        }
        if (closeOperation.getOpenOperation().getFlags().contains("inversion") && (reason == Trading.ExitReason.STRATEGY)) {
            Trading.OpenOperation op = operations.open(
                    0.3,
                    0.05,
                    closeOperation.getOpenOperation().isUpDireccion()
                            ? Trading.DireccionOperation.SHORT : Trading.DireccionOperation.LONG,
                    operations.getAvailableBalance() / 2, 4
            );
            if (op != null){
                op.getFlags().add("inversion");
            }
        }
    }

    private static boolean shouldCloseByInversion(Trading.OpenOperation open, StrategyUtils.BosChochSignal signal) {
        if (!signal.valid() || !signal.choch()) {
            return false;
        }

        if (open.isUpDireccion()) {
            return signal.direction() < 0;
        }
        return signal.direction() > 0;
    }

    private static boolean passesFilters(Candle candle, Trading.DireccionOperation direction) {
        double rsi = candle.rsi16();
        if (!Double.isFinite(rsi)) return false;

        float stMedium = candle.superTrendMedium();
        float stSlow = candle.superTrendSlow();
        boolean stLongBias = stMedium < 0 || stSlow < 0;
        boolean stShortBias = stMedium > 0 || stSlow > 0;

        if (direction == Trading.DireccionOperation.LONG) {
            if (!stLongBias) return false;
            return rsi >= 50.0;
        }
        if (direction == Trading.DireccionOperation.SHORT) {
            if (!stShortBias) return false;
            return rsi <= 50.0;
        }
        return false;
    }

    private static double calculateSlPercent(Candle candle) {
        double close = candle.close();
        double atr = candle.atr14();
        if (!Double.isFinite(close) || close <= 0 || !Double.isFinite(atr) || atr <= 0) {
            return 0.45;
        }
        double sl = (atr / close) * 100.0 * ATR_MULTIPLIER;
        return clamp(sl, MIN_SL_PERCENT, MAX_SL_PERCENT);
    }

    private static double calculateTpPercent(double slPercent, StrategyUtils.BosChochSignal signal) {
        double rr = signal.choch() ? RISK_REWARD * 0.95 : RISK_REWARD;
        double tp = slPercent * rr;
        return clamp(tp, MIN_TP_PERCENT, MAX_TP_PERCENT);
    }

    private boolean isDirectionInCooldown(Trading.DireccionOperation direction) {
        if (direction == Trading.DireccionOperation.LONG) {
            return longCooldown > 0;
        }
        if (direction == Trading.DireccionOperation.SHORT) {
            return shortCooldown > 0;
        }
        return true;
    }

    private void tickCooldowns() {
        if (longCooldown > 0) longCooldown--;
        if (shortCooldown > 0) shortCooldown--;
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

}
