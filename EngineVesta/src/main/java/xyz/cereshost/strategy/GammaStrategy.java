package xyz.cereshost.strategy;

import org.jetbrains.annotations.Nullable;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

import java.util.List;

public class GammaStrategy implements TradingStrategy {

    private static final int MAX_OPEN_CANDLES = 60;
    private static final double RISK_REWARD = 4; // 2:1
    private static final double MIN_SL_PCT = 0.2;
    private static final double MAX_SL_PCT = 0.4;


    @Override
    public void executeStrategy(PredictionEngine.@Nullable PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        for (Trading.OpenOperation o : operations.getOpens()){
            if (o.getCountCandles() >= MAX_OPEN_CANDLES) operations.close(Trading.ExitReason.TIMEOUT, o.getUuid());
        }


        if (operations.openSize() != 0) return;
        if (visibleCandles == null || visibleCandles.size() < 2) return;

        Candle prev = visibleCandles.get(visibleCandles.size() - 2);
        Candle curr = visibleCandles.get(visibleCandles.size() - 1);

        if (!isFinite(prev.macdVal()) || !isFinite(prev.macdSignal())
                || !isFinite(curr.macdVal()) || !isFinite(curr.macdSignal())) {
            return;
        }

        boolean crossUp = prev.macdVal() <= prev.macdSignal() && curr.macdVal() > curr.macdSignal();
        boolean crossDown = prev.macdVal() >= prev.macdSignal() && curr.macdVal() < curr.macdSignal();

        double slPercent = calcSlPercent(curr);
        double tpPercent = slPercent * RISK_REWARD;

//        if (Math.abs(curr.macdVal()) < 0.00005 || Math.abs(curr.macdSignal()) < 0.00005) return;
        Trading.DireccionOperation dir = curr.macdVal() < 0 ? Trading.DireccionOperation.LONG : Trading.DireccionOperation.SHORT;

        if (!crossUp && !crossDown) return;

        double rsi8 = curr.rsi8();
        double rsi16 = curr.rsi16();
        if (!isFinite(rsi8) || !isFinite(rsi16)) {
            return;
        }

        boolean longSignal = /*rsi8 <= 15 &&*/ rsi16 <= 40;
        boolean shortSignal = /*rsi8 >= 85 &&*/ rsi16 >= 60;
        if (curr.superTrend() > 0) if (dir != Trading.DireccionOperation.LONG) return;
        if (curr.superTrend() < 0) if (dir != Trading.DireccionOperation.SHORT) return;
//        if (longSignal) if (dir != Trading.DireccionOperation.LONG) return;
//        if (shortSignal) if (dir != Trading.DireccionOperation.SHORT) return; && tpPercent > 0.4

        if (operations.openSize() == 0 && (tpPercent < 0.9)) {
            operations.open(tpPercent, slPercent, dir, operations.getAvailableBalance()/2, 1);
        }
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
