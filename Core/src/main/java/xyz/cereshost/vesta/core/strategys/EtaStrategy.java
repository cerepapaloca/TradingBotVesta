package xyz.cereshost.vesta.core.strategys;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

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
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, TradingManager operations) {
        if (visibleCandles == null || visibleCandles.size() < 2) {
            return;
        }


        Candle previous = visibleCandles.get(visibleCandles.size() - 2);
        Candle current = visibleCandles.getLast();

        boolean newSession = isNewSession(previous, current);
        GapSetup gapSetup = newSession ? detectGap(previous, current) : GapSetup.none();

        for (TradingManager.OpenOperation open : operations.getOpens()) {
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
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {
        if (pendingEntry == null || operations.hasOpenOperation()) {
            return;
        }

        PendingEntry setup = pendingEntry;
        pendingEntry = null;
        openGapTrade(operations, setup.direction(), setup.targetPrice());
    }

    private void openGapTrade(@NotNull TradingManager operations,
                              @NotNull DireccionOperation direction,
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

    private static boolean shouldCloseOpen(@NotNull TradingManager.OpenOperation open, boolean newSession, @NotNull GapSetup gapSetup) {
        return switch (CLOSE_WHEN) {
            case NEW_SESSION -> newSession;
            case NEW_GAP -> gapSetup.valid();
            case REVERSE_POSITION -> gapSetup.valid() && open.getDireccion() != gapSetup.direction();
        };
    }

    @NotNull
    private static TradingManager.ExitReason getCloseReason(@NotNull TradingManager.OpenOperation open, @NotNull GapSetup gapSetup) {
        if (CLOSE_WHEN == CloseWhen.REVERSE_POSITION && gapSetup.valid() && open.getDireccion() != gapSetup.direction()) {
            return TradingManager.ExitReason.STRATEGY_INVERSION;
        }
        return TradingManager.ExitReason.STRATEGY;
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
                    INVERT_GAP_DIRECTION ? DireccionOperation.LONG : DireccionOperation.SHORT,
                    prevBodyHigh
            );
        }

        boolean downGap = current.open() < previous.low() && prevBodyLow > currentBodyHigh;
        if (downGap) {
            return new GapSetup(
                    true,
                    INVERT_GAP_DIRECTION ? DireccionOperation.SHORT : DireccionOperation.LONG,
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

    private record GapSetup(boolean valid, DireccionOperation direction, double targetPrice) {
        private static GapSetup none() {
            return new GapSetup(false, DireccionOperation.NEUTRAL, Double.NaN);
        }
    }

    private record PendingEntry(DireccionOperation direction, double targetPrice) {

    }
}
