package xyz.cereshost.vesta.core.strategy;

import xyz.cereshost.vesta.core.engine.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;

import java.util.List;

/**
 * Interfaz para definir estrategias de trading personalizadas
 */
public interface TradingStrategy {

   void executeStrategy(PredictionEngine.PredictionResult prediction, List<Candle> visibleCandles, Trading operations);

   void closeOperation(Trading.CloseOperation closeOperation, Trading  operations);
}

