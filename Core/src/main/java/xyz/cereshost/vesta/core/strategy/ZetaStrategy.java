package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.List;

import static xyz.cereshost.vesta.core.utils.StrategyUtils.isHigh;
import static xyz.cereshost.vesta.core.utils.StrategyUtils.isLow;

public class ZetaStrategy implements TradingStrategy {
    private static final double GRID_POINT = 1.5;
    private static final double BASE_ORDER_UNITS = 1.8;
    private static final double BASE_ORDER_MARGIN_USDT = 10.0;
    private static final double MARTINGALE_MULTIPLIER = 2;
    private static final boolean ANTI_MARTINGALE = false;
    private static final int LEVERAGE = 4;
    private static final double MIN_ORDER_NOTIONAL = 5.0;

    @Nullable
    private Double baseline;
    private double orderUnits = BASE_ORDER_UNITS;
    @NotNull
    private Trading.DireccionOperation pendingSignal = Trading.DireccionOperation.NEUTRAL;
    private double pendingDistancePercent;

    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        if (visibleCandles == null || visibleCandles.isEmpty()) {
            return;
        }

        lastVisibleCandles = visibleCandles;

        Candle current = visibleCandles.getLast();
        double close = current.close();
        if (!Double.isFinite(close) || close <= 0) {
            operations.log("EL cierre de la vela dio infinito");
            return;
        }

        if (baseline == null || !Double.isFinite(baseline)) {
            baseline = close;
            operations.log("EL baseline dio infinito");
            return;
        }

        double previousBaseline = baseline;
//        double baselineMovePercent = Math.abs(((close - previousBaseline) / previousBaseline) * 100.0);
//        if (!Double.isFinite(baselineMovePercent) || baselineMovePercent <= 0) {
//            operations.log("EL baselineMovePercent dio infinito");
//            return;
//        }

        if (close > previousBaseline + GRID_POINT || close < previousBaseline - GRID_POINT) {
            baseline = close;
        }

//        if (baselineMovePercent >= GRID_PERCENT) {
//            baseline = close;
//        }

        Trading.DireccionOperation signal = getSignal(baseline, previousBaseline);
        if (signal == Trading.DireccionOperation.NEUTRAL) {
            operations.log("Momento no optimo para operar baseline: " + baseline + " previousBaseline: " + previousBaseline);
            return;
        }

        double distancePercent = ((GRID_POINT / close) * 100.0) -((lastVisibleCandles.getLast().atr14()-0.01)*0.1);// baselineMovePercent;
        if (!Double.isFinite(distancePercent) || distancePercent <= 0) {
            operations.log("EL distancePercent dio infinito");
            return;
        }

        for (Trading.OpenOperation open : operations.getOpens()) {
            if (open.getDireccion() != signal) {
                pendingSignal = signal;
                pendingDistancePercent = distancePercent;
                operations.close(Trading.ExitReason.STRATEGY_INVERSION, open);
            }
        }

        if (operations.hasOpenOperation()) {
            operations.log("Ya hay una operacion abierta");
            return;
        }

        operations.open(
                distancePercent,
                distancePercent,
                signal,
                operations.getAvailableBalance()/2,
                LEVERAGE
        );
    }

    private List<Candle> lastVisibleCandles;
    private int lossesCount = 0;
    private int cooldownMinutes = 0;

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {
        double pnlPercent = getClosedPnlPercent(closeOperation);
        if (cooldownMinutes > 0) return;
        if (pnlPercent > 0) {
            orderUnits = ANTI_MARTINGALE ? orderUnits * MARTINGALE_MULTIPLIER : BASE_ORDER_UNITS;
        } else if (pnlPercent < 0) {
            orderUnits = ANTI_MARTINGALE ? BASE_ORDER_UNITS : orderUnits * MARTINGALE_MULTIPLIER;
        }

        if (!Double.isFinite(orderUnits) || orderUnits <= 0) {
            orderUnits = BASE_ORDER_UNITS;
        }

        if (closeOperation.getReason() != Trading.ExitReason.STRATEGY_INVERSION
                || pendingSignal == Trading.DireccionOperation.NEUTRAL
                || !Double.isFinite(pendingDistancePercent)
                || pendingDistancePercent <= 0) {
            return;
        }

        double minMarginUsdt = MIN_ORDER_NOTIONAL / Math.max(LEVERAGE, 1);
        double amountUsdt = Math.max(orderUnits * BASE_ORDER_MARGIN_USDT, minMarginUsdt);
        amountUsdt = Math.min(amountUsdt, operations.getAvailableBalance());
        Trading.DireccionOperation signal = pendingSignal;
        double distancePercent = pendingDistancePercent;
        pendingSignal = Trading.DireccionOperation.NEUTRAL;
        pendingDistancePercent = 0;

        if (amountUsdt * LEVERAGE < MIN_ORDER_NOTIONAL) {
            return;
        }

        operations.open(
                distancePercent,
                distancePercent,
                signal,
                amountUsdt,
                LEVERAGE
        );
    }

    @NotNull
    private static Trading.DireccionOperation getSignal(double baseline, double previousBaseline) {
        if (baseline > previousBaseline) {
            return Trading.DireccionOperation.LONG;
        }
        if (baseline < previousBaseline) {
            return Trading.DireccionOperation.SHORT;
        }
        return Trading.DireccionOperation.NEUTRAL;
    }

    private static double getClosedPnlPercent(@NotNull Trading.CloseOperation closeOperation) {
        Trading.OpenOperation open = closeOperation.getOpenOperation();
        double entry = open.getEntryPrice();
        double exit = closeOperation.getExitPrice();
        if (!Double.isFinite(entry) || !Double.isFinite(exit) || entry <= 0) {
            return 0;
        }

        double movePercent = ((exit - entry) / entry) * 100.0;
        return open.isUpDireccion() ? movePercent : -movePercent;
    }
}
