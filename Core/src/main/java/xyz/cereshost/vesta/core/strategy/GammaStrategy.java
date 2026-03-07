package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.ArrayList;
import java.util.List;

import static xyz.cereshost.vesta.core.utils.StrategyUtils.isHigh;
import static xyz.cereshost.vesta.core.utils.StrategyUtils.isLow;

public class GammaStrategy implements TradingStrategy {

    private static final int MAX_OPEN_CANDLES = 30;
    private static final double RISK_REWARD = 3; // 2:1
    private static final double MIN_SL_PCT = 0.05;
    private static final double MAX_SL_PCT = 6;

    private static final double TP_TREND = 0;
    private static final double SL_TREND = 0;

    @NotNull
    private Trading.DireccionOperation direccionOperationWindow = Trading.DireccionOperation.NEUTRAL;
    private int candlesValid;

    boolean isPeekClose = false;
    boolean longBan = false;
    boolean shortBan = false;


    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        if (coolDown != 0) {
            coolDown--;
            return;
        }

        Candle prev = visibleCandles.get(visibleCandles.size() - 2);
        Candle curr = visibleCandles.get(visibleCandles.size() - 1);
        if (!isFinite(prev.macdVal()) || !isFinite(prev.macdSignal())
                || !isFinite(curr.macdVal()) || !isFinite(curr.macdSignal())) {
            operations.log("MACD dio infinito");
            return;
        }

        boolean crossUp = (prev.macdVal() <= prev.macdSignal() && curr.macdVal() > curr.macdSignal());
        boolean crossDown = prev.macdVal() >= prev.macdSignal() && curr.macdVal() < curr.macdSignal();
        boolean validMACD = crossUp || crossDown;
        for (Trading.OpenOperation o : operations.getOpens()) {
            if (o.getMinutesOpen() >= MAX_OPEN_CANDLES && o.getFlags().contains("inversion") && o.getRoiRaw() > 0) {
                operations.close(Trading.ExitReason.TIMEOUT, o);
                continue;
            }

//            if ((o.getCountCandles() % 5) == 0){
//                double tp = Math.max(o.getTpPercent() - 0.6, o.getOriginalSlPercent());
//                double roi = o.getRoiRaw();
//                // Margen de seguridad
//                if (roi > tp - 0.04) {
//                    operations.close(Trading.ExitReason.STRATEGY, o);
//                }else {
//                    o.setTpPercent(tp);
//                }
//            }
//            if (o.getCountCandles() > 60){
//                if (o.getRoiRaw() > operations.getMarket().getFeedPercent() + 0.1){
//                    operations.close(Trading.ExitReason.STRATEGY, o);
//                }
//            }
//            if ((o.getCountCandles() % 5) == 0 && o.getCountCandles() > 15){
//                double sl = Math.max(o.getSlPercent() - 0.05, 0.01);
//                double roi = o.getRoiRaw();
//                // Margen de seguridad
//                if (roi > sl + 0.02) {
//                    operations.close(Trading.ExitReason.STRATEGY, o);
//                }else {
//                    o.setSlPercent(sl);
//                }
//            }
//            if (o.getFlags().contains("trend")) return;
            boolean inversion = o.getFlags().contains("inversion");
            boolean margenTakeProfit = o.getRoiRaw() > 0.3 || (inversion && o.getRoiRaw() > 0.1);
            if (o.isUpDireccion()){
                boolean b = isHigh(visibleCandles, inversion ? 15 : 75);
                if (!isPeekClose) isPeekClose = b;
                if (isPeekClose && !b && margenTakeProfit) {//
                    o.getFlags().add("inversion");
                    operations.close(Trading.ExitReason.STRATEGY, o);
                }
            }else {
                boolean b = isLow(visibleCandles, inversion ? 15 : 75);
                if (!isPeekClose) isPeekClose = b;
                if (isPeekClose && !b && margenTakeProfit) {
                    o.getFlags().add("inversion");
                    operations.close(Trading.ExitReason.STRATEGY, o);
                }
            }
        }

        if (operations.hasOpenOperation()) {
            operations.log("Ya hay una operación abierta");
            return;
        }

        double slPercent = calcSlPercent(curr);
        double tpPercent = slPercent * RISK_REWARD;

        trend = getTrend(visibleCandles);
        double deltaMae = curr.emaSlow() - prev.emaSlow();
        Trading.DireccionOperation dirSuperTrendFast = curr.superTrendFast() > 0 ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;
        Trading.DireccionOperation dirSuperTrendMedium = curr.superTrendMedium() > 0 ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;

        switch (trend){
            case LONG -> {
//                if (dirSuperTrendFast != Trading.DireccionOperation.LONG) return;
                Trading.OpenOperation op = operations.open(
                        tpPercent + TP_TREND + (deltaMae),
                        slPercent + SL_TREND + (deltaMae*1.5),
                        Trading.DireccionOperation.LONG, operations.getAvailableBalance() / 2, 4);
                if (op != null){
                    op.getFlags().add("trend");
                }
            }
            case SHORT -> {
//                if (dirSuperTrendFast != Trading.DireccionOperation.SHORT) return;
                Trading.OpenOperation op = operations.open(
                        tpPercent + TP_TREND + (deltaMae),
                        slPercent + SL_TREND + (deltaMae*1.5),
                        Trading.DireccionOperation.SHORT, operations.getAvailableBalance() / 2, 4);
                if (op != null){
                    op.getFlags().add("trend");
                }
            }
            case NEUTRAL -> {
//                if (Math.abs(curr.macdVal()) < 0.00005 || Math.abs(curr.macdSignal()) < 0.00005) return;
                Trading.DireccionOperation dirSuperTrendSlow = curr.superTrendSlow() > 0 ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;

                Trading.DireccionOperation dirMACD;

                if (candlesValid != 0){
                    candlesValid--;
                }else {
                    direccionOperationWindow = Trading.DireccionOperation.NEUTRAL;
                }

                if (!validMACD) {
                    if (direccionOperationWindow ==  Trading.DireccionOperation.NEUTRAL) {
                        operations.log("MACD no cumple con la condición para operar");
                        return;
                    }else {
                        dirMACD = direccionOperationWindow;
                    }
                }else {
                    candlesValid = 4;
                    dirMACD = crossUp ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;
                }

                if (dirMACD == Trading.DireccionOperation.LONG && prev.macdHist() > 0) {
                    return;
                }
                if (dirMACD == Trading.DireccionOperation.SHORT && prev.macdHist() < 0) {
                    return;
                }

                direccionOperationWindow = dirSuperTrendSlow;
//        if (!dirMACD.equals(dirSuperTrendSlow) &&
//                !dirSuperTrendFast.equals(dirSuperTrendSlow) &&
//                !dirSuperTrendMedium.equals(dirSuperTrendSlow)
//        ) {
//            operations.log("SuperTrend: " + dirSuperTrendSlow + " != MACD: " + dirMACD);
//            return;
//        }

                boolean min = false;
                boolean max = false;

                if (longBan) if (dirMACD == Trading.DireccionOperation.LONG) {
                    shortBan = false;
                    return;
                }
                if (shortBan)  if (dirMACD == Trading.DireccionOperation.SHORT) {
                    longBan = false;
                    return;
                }
//        List<Candle> candlesHeavy = new ArrayList<>(visibleCandles);
//        for (int back = 0; back < 15; back++) {
//            candlesHeavy.removeLast();
//            if (!min) min = isLow(candlesHeavy,40);
//            if (!max) max = isHigh(candlesHeavy, 40);
//        }
                List<Candle> candles = new ArrayList<>(visibleCandles);

                for (int back = 0; back < 2; back++) {
                    candles.removeLast();
                    if (!min) min = isLow(candles,30);
                    if (!max) max = isHigh(candles, 30);
                }

                if (max && (dirMACD == Trading.DireccionOperation.SHORT)) return;
                if (min && (dirMACD == Trading.DireccionOperation.LONG)) return;

        double rsi8 = curr.rsi8();
        double rsi16 = curr.rsi16();
        if (!isFinite(rsi8) || !isFinite(rsi16)) {
            return;
        }

        boolean longSignal = rsi8 <= 2;
        boolean shortSignal = rsi8 >= 98;
        if (longSignal) if (dirMACD != Trading.DireccionOperation.LONG) {
            operations.log("El SuperTrend cancela Long");
            return;
        }
        if (shortSignal) if (dirMACD != Trading.DireccionOperation.SHORT) {
            operations.log("El SuperTrend cancela Short");
            return;
        }
//        if (curr.superTrendSlow() > 0) if (dirMACD != Trading.DireccionOperation.LONG) {
//            operations.log("El SuperTrend cancela Long");
//            return;
//        }
//        if (curr.superTrendSlow() < 0) if (dirMACD != Trading.DireccionOperation.SHORT) {
//            operations.log("El SuperTrend cancela Short");
//            return;
//        }
//        if (longSignal) if (dir != Trading.DireccionOperation.LONG) return;
//        if (shortSignal) if (dir != Trading.DireccionOperation.SHORT) return; && tpPercent > 0.4
                if (dirSuperTrendSlow == dirSuperTrendMedium && dirSuperTrendMedium == dirSuperTrendFast && dirSuperTrendFast == dirMACD) {
                    operations.open(tpPercent, slPercent + Math.abs(curr.macdSignal()), dirMACD, operations.getAvailableBalance() / 2, 4);
                }
            }
        }


    }

    private int coolDown = 0;
    private Trading.DireccionOperation trend = Trading.DireccionOperation.NEUTRAL;

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {
        isPeekClose = false;
        Trading.ExitReason reason = closeOperation.getReason();
        if (closeOperation.getOpenOperation().getFlags().contains("inversion") && (reason == Trading.ExitReason.STRATEGY)) {
            Trading.DireccionOperation direccion = closeOperation.getOpenOperation().isUpDireccion()
                    ? Trading.DireccionOperation.SHORT : Trading.DireccionOperation.LONG;
            Trading.OpenOperation op = operations.open(
                    closeOperation.getOpenOperation().getTpPercent(),
                    closeOperation.getOpenOperation().getSlPercent(),
                    direccion,
                    operations.getAvailableBalance() / 2, 4
            );
            if (op != null){
                op.getFlags().add("inversion");
            }
        }else if (reason.isStopLoss()) {
            coolDown = 45;
            if (reason.equals(Trading.ExitReason.LONG_STOP_LOSS)){
                longBan = true;
            }
            if (reason.equals(Trading.ExitReason.SHORT_STOP_LOSS)){
                shortBan = true;
            }
        };
    }

    private static Trading.DireccionOperation getTrend(List<Candle> candles) {
        Candle prev = candles.get(candles.size() - 2);
        Candle curr = candles.getLast();
        double deltaMae = curr.emaSlow() - prev.emaSlow();
        if (deltaMae > 0.0008){
            return Trading.DireccionOperation.LONG;
        }
        if (deltaMae < -0.0008){
            return Trading.DireccionOperation.SHORT;
        }
        return Trading.DireccionOperation.NEUTRAL;
    }

    private static double calcSlPercent(Candle candle) {
        double close = candle.close();
        double atr = candle.atr14();
        double sl;
        if (isFinite(atr) && atr > 0 && isFinite(close) && close > 0) {
            sl = (atr / close) * 100.0;
        } else {
            sl = 0.10;
        }
        return clamp(sl, MIN_SL_PCT, MAX_SL_PCT);
    }

    private static boolean isFinite(double v) {
        return Double.isFinite(v);
    }

    private static double clamp(double v, double min, double max) {
        return Math.max(min, Math.min(max, v));
    }
}
