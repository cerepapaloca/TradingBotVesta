package xyz.cereshost.strategy;

import org.w3c.dom.ls.LSInput;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.ListOperations;
import xyz.cereshost.engine.PredictionEngine;

import java.util.List;

/**
 * Interfaz para definir estrategias de trading personalizadas
 */
public interface BacktestStrategy {

   BackTestEngine.OpenOperation preProcess(PredictionEngine.PredictionResultTP_SL prediction, ListOperations openOperations, Candle candle, double balance);

    void postProcess(BackTestEngine.TradeResult result);
}

