package xyz.cereshost.vesta.core.strategys;

import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.util.List;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.PredictionResult pred, List<Candle> visibleCandles, TradingManager openOperations) {
        for (TradingManager.OpenOperation op : openOperations.getOpens()) if (op.getMinutesOpen() >= 30) openOperations.close(TradingManager.ExitReason.STRATEGY, op);
        if (!openOperations.hasOpenOperation() && pred != null) {
            openOperations.open(
                    pred.getTpPercent(),
                    pred.getSlPercent(),
                    DireccionOperation.LONG,
                    openOperations.getAvailableBalance()/2,
                    4
            );
        }
    }

    @Override
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {

    }
}
