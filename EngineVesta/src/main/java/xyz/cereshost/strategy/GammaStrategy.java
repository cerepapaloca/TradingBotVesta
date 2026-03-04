package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

import java.util.List;

public class GammaStrategy implements TradingStrategy {

    private static final int MAX_OPEN_CANDLES = 40;
    private static final double RISK_REWARD = 2.6; // 2:1
    private static final double MIN_SL_PCT = 0.4;
    private static final double MAX_SL_PCT = 2;

    @NotNull
    private Trading.DireccionOperation direccionOperationWindow = Trading.DireccionOperation.NEUTRAL;
    private int candlesValid;

    int i = 5;


    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
//        if (i == 0) return;
        Candle prev = visibleCandles.get(visibleCandles.size() - 2);
        Candle curr = visibleCandles.get(visibleCandles.size() - 1);
        if (!isFinite(prev.macdVal()) || !isFinite(prev.macdSignal())
                || !isFinite(curr.macdVal()) || !isFinite(curr.macdSignal())) {
            operations.log("MACD dio infinito");
            return;
        }

        boolean crossUp = (prev.macdVal() <= prev.macdSignal() && curr.macdVal() > curr.macdSignal());
        boolean crossDown = prev.macdVal() >= prev.macdSignal() && curr.macdVal() < curr.macdSignal();
        boolean ValidMACD = crossUp || crossDown;
        for (Trading.OpenOperation o : operations.getOpens()) {
            if (o.getCountCandles() >= MAX_OPEN_CANDLES) {
                operations.close(Trading.ExitReason.TIMEOUT, o.getUuid());
                continue;
            }

            if (ValidMACD && o.getRoiRaw() > operations.getMarket().getFeedPercent()){
                if (crossDown && o.isUpDireccion()) {
                    operations.close(Trading.ExitReason.STRATEGY, o.getUuid());
                }
                if (crossUp && !o.isUpDireccion()) {
                    operations.close(Trading.ExitReason.STRATEGY, o.getUuid());
                }
            }

            if ((o.getCountCandles() % 5) == 0){
                double tp = Math.max(o.getTpPercent() - 0.2, o.getOriginalSlPercent());
                double roi = o.getRoiRaw();
                // Margen de seguridad
                if (roi > tp - 0.05) {
                    operations.close(Trading.ExitReason.STRATEGY, o.getUuid());
                    continue;
                }else {
                    o.setTpPercent(tp);
                }
            }
            if (o.getCountCandles() > 20){
                if (o.getRoiRaw() > operations.getMarket().getFeedPercent() + 0.05){
                    operations.close(Trading.ExitReason.STRATEGY, o.getUuid());
                }
            }
        }

        if (operations.hasOpenOperation()) {
            operations.log("Ya hay una operación abierta");
            return;
        }
        if (visibleCandles == null || visibleCandles.size() < 2) return;

        double slPercent = calcSlPercent(curr);
        double tpPercent = slPercent * RISK_REWARD;

//        if (Math.abs(curr.macdVal()) < 0.00005 || Math.abs(curr.macdSignal()) < 0.00005) return;
        Trading.DireccionOperation dirSuperTrend = curr.superTrend() < 0 ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;
        Trading.DireccionOperation dirMACD;

        if (candlesValid != 0){
            candlesValid--;
        }else {
            direccionOperationWindow = Trading.DireccionOperation.NEUTRAL;
        }

        if (!ValidMACD) {
            if (direccionOperationWindow ==  Trading.DireccionOperation.NEUTRAL) {
                operations.log("MACD no cumple con la condición para operar");
                return;
            }else {
                dirMACD = direccionOperationWindow;
            }
        }else {
            candlesValid = 5;
            dirMACD = crossUp ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;
        }

        if (dirMACD == Trading.DireccionOperation.LONG && prev.macdHist() > 0) {
            return;
        }
        if (dirMACD == Trading.DireccionOperation.SHORT && prev.macdHist() < 0) {
            return;
        }

        direccionOperationWindow = dirSuperTrend;
        if (!dirMACD.equals(dirSuperTrend)) {
            operations.log("SuperTrend: " + dirSuperTrend + " != MACD: " + dirMACD);
            return;
        }

        double rsi8 = curr.rsi8();
        double rsi16 = curr.rsi16();
        if (!isFinite(rsi8) || !isFinite(rsi16)) {
            return;
        }

//        boolean longSignal = rsi8 <= 10;
//        boolean shortSignal = rsi8 >= 90;
//        if (longSignal) if (dirMACD != Trading.DireccionOperation.LONG) {
//            operations.log("El SuperTrend cancela Long");
//            return;
//        }
//        if (shortSignal) if (dirMACD != Trading.DireccionOperation.SHORT) {
//            operations.log("El SuperTrend cancela Short");
//            return;
//        }
//        if (curr.superTrend() > 0) if (dirMACD != Trading.DireccionOperation.LONG) {
//            operations.log("El SuperTrend cancela Long");
//            return;
//        }
//        if (curr.superTrend() < 0) if (dirMACD != Trading.DireccionOperation.SHORT) {
//            operations.log("El SuperTrend cancela Short");
//            return;
//        }
//        if (longSignal) if (dir != Trading.DireccionOperation.LONG) return;
//        if (shortSignal) if (dir != Trading.DireccionOperation.SHORT) return; && tpPercent > 0.4
        System.out.println(dirMACD + "\n" + curr.superTrend());
        operations.open(tpPercent, slPercent, dirMACD, operations.getAvailableBalance() / 2, 4);
        i--;
    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation) {

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
