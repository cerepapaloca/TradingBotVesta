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
        if ((pred.getRatio() > 1 && pred.getTpPercent() > 0.15)) {
            if (operations.openSize() == 0 && pred.direction() !=  Trading.DireccionOperation.NEUTRAL) {
                operations.open(pred.getTpPercent() +0.2, pred.getSlPercent() +0.2, pred.direction(), operations.getAvailableBalance(), 1);
            }
        }
    }
}
