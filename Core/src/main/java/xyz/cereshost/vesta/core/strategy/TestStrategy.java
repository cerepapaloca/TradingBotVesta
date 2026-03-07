package xyz.cereshost.vesta.core.strategy;

import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.List;

import static xyz.cereshost.vesta.core.utils.StrategyUtils.isHigh;

public class TestStrategy implements TradingStrategy {

    private boolean isPeekClose = false;
    @Override
    public void executeStrategy(PredictionEngine.PredictionResult prediction, List<Candle> visibleCandles, Trading openOperations) {
        if (openOperations.hasOpenOperation()){

        }else {
            boolean b = isHigh(visibleCandles, 60);
            if (!isPeekClose) isPeekClose = b;
            if (isPeekClose && !b)
                openOperations.open(0.4, 0.2, Trading.DireccionOperation.SHORT, openOperations.getAvailableBalance()/2, 1);
        }
    }

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {
        isPeekClose = false;
    }
}
