package xyz.cereshost.vesta.core.strategys;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.util.List;

public class ZetaStrategy implements TradingStrategy {
    private static final double GRID_POINT_PERCENT = 0.3;
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
    private DireccionOperation pendingSignal = DireccionOperation.NEUTRAL;
    private double pendingDistancePercent;

    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, TradingManager operations) {
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
        if (!Double.isFinite(previousBaseline) || previousBaseline <= 0) {
            baseline = close;
            operations.log("EL baseline dio infinito");
            return;
        }

        double movePercent = ((close - previousBaseline) / previousBaseline) * 100.0;
        if (!Double.isFinite(movePercent)) {
            operations.log("EL movePercent dio infinito");
            return;
        }

        if (Math.abs(movePercent) >= GRID_POINT_PERCENT) {
            baseline = close;
        }

        DireccionOperation signal = getSignal(baseline, previousBaseline);
        if (signal == DireccionOperation.NEUTRAL) {
            operations.log("Momento no optimo para operar baseline: " + baseline + " previousBaseline: " + previousBaseline);
            return;
        }
        double rsi = current.rsi8();
        double strong = Math.abs(rsi -50)/100;
        double atrPercent = (lastVisibleCandles.getLast().atr14() / close) * 100.0;
        if (atrPercent < 0.15) return;
        double distancePercent = GRID_POINT_PERCENT + ((strong) * GRID_POINT_PERCENT);//* ((atrPercent - 0.01) * 0.1);// baselineMovePercent;
        if (!Double.isFinite(distancePercent) || distancePercent <= 0) {
            operations.log("EL distancePercent dio infinito");
            return;
        }

//        for (TradingManager.OpenOperation open : operations.getOpens()) {
//            if (open.getDireccion() != signal) {
//                pendingSignal = signal;
//                pendingDistancePercent = distancePercent;
//                operations.close(TradingManager.ExitReason.STRATEGY_INVERSION, open);
//            }
//        }

        if (operations.hasOpenOperation()) {
            operations.log("Ya hay una operacion abierta");
            return;
        }

        operations.open(
                distancePercent,
                distancePercent*0.9,
                signal,
                operations.getAvailableBalance()/2,
                LEVERAGE
        );
    }

    private List<Candle> lastVisibleCandles;

    @Override
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {
        double pnlPercent = getClosedPnlPercent(closeOperation);
        if (pnlPercent > 0) {
            orderUnits = ANTI_MARTINGALE ? orderUnits * MARTINGALE_MULTIPLIER : BASE_ORDER_UNITS;
        } else if (pnlPercent < 0) {
            orderUnits = ANTI_MARTINGALE ? BASE_ORDER_UNITS : orderUnits * MARTINGALE_MULTIPLIER;
        }

        if (!Double.isFinite(orderUnits) || orderUnits <= 0) {
            orderUnits = BASE_ORDER_UNITS;
        }

        if (closeOperation.getReason() != TradingManager.ExitReason.STRATEGY_INVERSION
                || pendingSignal == DireccionOperation.NEUTRAL
                || !Double.isFinite(pendingDistancePercent)
                || pendingDistancePercent <= 0) {
            return;
        }

        double minMarginUsdt = MIN_ORDER_NOTIONAL / Math.max(LEVERAGE, 1);
        double amountUsdt = Math.max(orderUnits * BASE_ORDER_MARGIN_USDT, minMarginUsdt);
        amountUsdt = Math.min(amountUsdt, operations.getAvailableBalance());
        DireccionOperation signal = pendingSignal;
        double distancePercent = pendingDistancePercent;
        pendingSignal = DireccionOperation.NEUTRAL;
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
    private static DireccionOperation getSignal(double baseline, double previousBaseline) {
        if (baseline > previousBaseline) {
            return DireccionOperation.LONG;
        }
        if (baseline < previousBaseline) {
            return DireccionOperation.SHORT;
        }
        return DireccionOperation.NEUTRAL;
    }

    private static double getClosedPnlPercent(@NotNull TradingManager.CloseOperation closeOperation) {
        TradingManager.OpenOperation open = closeOperation.getOpenOperation();
        double entry = open.getEntryPrice();
        double exit = closeOperation.getExitPrice();
        if (!Double.isFinite(entry) || !Double.isFinite(exit) || entry <= 0) {
            return 0;
        }

        double movePercent = ((exit - entry) / entry) * 100.0;
        return open.isUpDireccion() ? movePercent : -movePercent;
    }
}
