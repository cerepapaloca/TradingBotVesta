package xyz.cereshost.strategy;

import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

import java.util.List;

public class TestStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.PredictionResult prediction, List<Candle> visibleCandles, Trading openOperations) {
        if (openOperations.hasOpenOperation()){
            for (Trading.OpenOperation operation : openOperations.getOpens()){
                openOperations.close(Trading.ExitReason.STRATEGY, operation.getUuid());
            }
        }else {
            openOperations.open(0.9, 0.001, Trading.DireccionOperation.SHORT, openOperations.getAvailableBalance()/2, 1);
        }
    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation) {

    }
}
