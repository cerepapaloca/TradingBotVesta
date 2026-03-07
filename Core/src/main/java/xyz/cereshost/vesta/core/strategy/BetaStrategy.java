package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.List;

public class BetaStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, List<Candle> visibleCandles, Trading operations) {
        for (Trading.OpenOperation o : operations.getOpens()){
            if (o.getMinutesOpen() >= 60) operations.close(Trading.ExitReason.TIMEOUT, o);
        }
        if (operations.openSize() == 0 && pred.directionOperation() != Trading.DireccionOperation.NEUTRAL) {
            operations.open(pred.getTpPercent(), pred.getSlPercent(), pred.directionOperation(), operations.getAvailableBalance()/2, 1);
        }
    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {

    }
}
