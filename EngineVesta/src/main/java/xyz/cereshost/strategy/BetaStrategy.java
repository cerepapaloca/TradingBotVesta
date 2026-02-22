package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

import java.util.List;

public class BetaStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        for (Trading.OpenOperation o : operations.getOpens()){
            if (o.getCountCandles() >= 60) operations.close(Trading.ExitReason.TIMEOUT, o.getUuid());
        }
        if (operations.openSize() == 0 && pred.directionOperation() != Trading.DireccionOperation.NEUTRAL) {
            operations.open(pred.getTpPercent(), pred.getSlPercent(), pred.directionOperation(), operations.getAvailableBalance()/2, 1);
        }
    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation) {

    }
}
