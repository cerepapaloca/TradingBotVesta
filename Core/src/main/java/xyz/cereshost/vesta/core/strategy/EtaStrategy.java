package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.List;

public class EtaStrategy implements TradingStrategy {
    private static final long MILLIS_PER_DAY = 86_400_000L;
    private static final boolean INVERT_GAP_DIRECTION = false;
    private static final CloseWhen CLOSE_WHEN = CloseWhen.NEW_SESSION;
    private static final int LEVERAGE = 1;
    private static final double ORDER_BALANCE_FRACTION = 0.50;
    private static final double MIN_ORDER_NOTIONAL = 5.0;
    private static final double MIN_PROTECTION_PERCENT = 0.10;
    private static final double MAX_PROTECTION_PERCENT = 8.0;

    @Nullable
    private PendingEntry pendingEntry;


    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        if (visibleCandles == null || visibleCandles.size() < 2) {
            return;
        }


        Candle previous = visibleCandles.get(visibleCandles.size() - 2);
        Candle current = visibleCandles.getLast();

        boolean newSession = isNewSession(previous, current);
        GapSetup gapSetup = newSession ? detectGap(previous, current) : GapSetup.none();

        for (Trading.OpenOperation open : operations.getOpens()) {
            if (shouldCloseOpen(open, newSession, gapSetup)) {
                if (gapSetup.valid()) {
                    pendingEntry = new PendingEntry(gapSetup.direction(), gapSetup.targetPrice());
                }
                operations.close(getCloseReason(open, gapSetup), open);
            }
        }

        if (operations.hasOpenOperation() || !gapSetup.valid()) {
            return;
        }

        openGapTrade(operations, gapSetup.direction(), gapSetup.targetPrice());

    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {
        if (pendingEntry == null || operations.hasOpenOperation()) {
            return;
        }

        PendingEntry setup = pendingEntry;
        pendingEntry = null;
        openGapTrade(operations, setup.direction(), setup.targetPrice());
    }

    private void openGapTrade(@NotNull Trading operations,
                              @NotNull Trading.DireccionOperation direction,
                              double targetPrice) {
        double currentPrice = operations.getCurrentPrice();
        if (!Double.isFinite(currentPrice) || currentPrice <= 0 || !Double.isFinite(targetPrice) || targetPrice <= 0) {
            return;
        }

        double tpPercent = Math.abs((targetPrice - currentPrice) / currentPrice) * 100.0;
        if (!Double.isFinite(tpPercent) || tpPercent <= 0) {
            return;
        }

        double slPercent = clamp(tpPercent, MIN_PROTECTION_PERCENT, MAX_PROTECTION_PERCENT);
        double minMarginUsdt = MIN_ORDER_NOTIONAL / Math.max(LEVERAGE, 1);
        double availableBalance = operations.getAvailableBalance();
        double amountUsdt = Math.max(availableBalance * ORDER_BALANCE_FRACTION, minMarginUsdt);
        amountUsdt = Math.min(amountUsdt, availableBalance);

        if (amountUsdt * LEVERAGE < MIN_ORDER_NOTIONAL) {
            operations.log("Balance insuficiente para operar gap fill");
            return;
        }

        operations.open(
                tpPercent,
                slPercent,
                direction,
                amountUsdt,
                LEVERAGE
        );
    }

    private static boolean shouldCloseOpen(@NotNull Trading.OpenOperation open, boolean newSession, @NotNull GapSetup gapSetup) {
        return switch (CLOSE_WHEN) {
            case NEW_SESSION -> newSession;
            case NEW_GAP -> gapSetup.valid();
            case REVERSE_POSITION -> gapSetup.valid() && open.getDireccion() != gapSetup.direction();
        };
    }

    @NotNull
    private static Trading.ExitReason getCloseReason(@NotNull Trading.OpenOperation open, @NotNull GapSetup gapSetup) {
        if (CLOSE_WHEN == CloseWhen.REVERSE_POSITION && gapSetup.valid() && open.getDireccion() != gapSetup.direction()) {
            return Trading.ExitReason.STRATEGY_INVERSION;
        }
        return Trading.ExitReason.STRATEGY;
    }

    private static boolean isNewSession(@NotNull Candle previous, @NotNull Candle current) {
        long previousDay = Math.floorDiv(previous.openTime(), MILLIS_PER_DAY);
        long currentDay = Math.floorDiv(current.openTime(), MILLIS_PER_DAY);
        return previousDay != currentDay;
    }

    @NotNull
    private static GapSetup detectGap(@NotNull Candle previous, @NotNull Candle current) {
        double prevBodyHigh = Math.max(previous.open(), previous.close());
        double prevBodyLow = Math.min(previous.open(), previous.close());
        double currentBodyHigh = Math.max(current.open(), current.close());
        double currentBodyLow = Math.min(current.open(), current.close());

        boolean upGap = current.open() > previous.high() && currentBodyLow > prevBodyHigh;
        if (upGap) {
            return new GapSetup(
                    true,
                    INVERT_GAP_DIRECTION ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT,
                    prevBodyHigh
            );
        }

        boolean downGap = current.open() < previous.low() && prevBodyLow > currentBodyHigh;
        if (downGap) {
            return new GapSetup(
                    true,
                    INVERT_GAP_DIRECTION ? Trading.DireccionOperation.SHORT : Trading.DireccionOperation.LONG,
                    prevBodyLow
            );
        }

        return GapSetup.none();
    }

    private static double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    private enum CloseWhen {
        NEW_SESSION,
        NEW_GAP,
        REVERSE_POSITION
    }

    private record GapSetup(boolean valid, Trading.DireccionOperation direction, double targetPrice) {
        private static GapSetup none() {
            return new GapSetup(false, Trading.DireccionOperation.NEUTRAL, Double.NaN);
        }
    }

    private record PendingEntry(Trading.DireccionOperation direction, double targetPrice) {

    }
}
