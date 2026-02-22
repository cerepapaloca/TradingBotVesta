package xyz.cereshost.strategy;

import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

import java.util.List;

/**
 * Interfaz para definir estrategias de trading personalizadas
 */
public interface TradingStrategy {

   void executeStrategy(PredictionEngine.PredictionResult prediction, List<Candle> visibleCandles, Trading openOperations);

   void closeOperation(Trading.CloseOperation closeOperation);
}

