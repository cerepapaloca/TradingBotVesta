package xyz.cereshost.vesta.core.strategys;

import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.util.List;

import static xyz.cereshost.vesta.core.utils.StrategyUtils.isHigh;

public class TestStrategy implements TradingStrategy {

    private boolean isPeekClose = false;
    @Override
    public void executeStrategy(PredictionEngine.PredictionResult prediction, List<Candle> visibleCandles, TradingManager openOperations) {
        if (openOperations.hasOpenOperation()){

        }else {
            boolean b = isHigh(visibleCandles, 60);
            if (!isPeekClose) isPeekClose = b;
            if (isPeekClose && !b)
                openOperations.open(0.4, 0.2, DireccionOperation.SHORT, openOperations.getAvailableBalance()/2, 1);
        }
    }

    @Override
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {
        isPeekClose = false;
    }
}
