package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

public class BetaStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, Trading operations) {
        for (Trading.OpenOperation o : operations.getOpens()){
            if (o.getCountCandles() >= 60) operations.close(Trading.ExitReason.TIMEOUT, o.getUuid());
        }
        if (pred.confident() > 0.7f) {
            if (operations.openSize() == 0 && pred.directionOperation() != Trading.DireccionOperation.NEUTRAL) {
                operations.open(pred.getTpPercent() +0.2, pred.getSlPercent() +0.2, pred.directionOperation(), operations.getAvailableBalance(), 1);
            }
        }
    }
}
