package xyz.cereshost.strategy;

import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.PredictionEngine;

/**
 * Interfaz para definir estrategias de trading personalizadas
 */
public interface BacktestStrategy {

   BackTestEngine.TradeSetup preProcess(PredictionEngine.PredictionResultTP_SL prediction, Candle candle, double balance);

    void postProcess(BackTestEngine.TradeResult result);
}

