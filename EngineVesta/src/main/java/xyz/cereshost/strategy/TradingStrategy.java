package xyz.cereshost.strategy;

import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.trading.Trading;

/**
 * Interfaz para definir estrategias de trading personalizadas
 */
public interface TradingStrategy {

   void executeStrategy(PredictionEngine.PredictionResult prediction, Trading openOperations);
}

