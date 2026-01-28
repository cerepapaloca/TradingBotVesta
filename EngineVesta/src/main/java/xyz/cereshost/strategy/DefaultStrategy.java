package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.engine.PredictionEngine;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, Trading openOperations) {
        for (Trading.OpenOperation op : openOperations.getOpens()) if (op.getCountCandles() >= 1) openOperations.close(Trading.ExitReason.STRATEGY, op.getUuid());
        if (openOperations.openSize() == 0) {
            openOperations.open(
                    0.06,
                    0.06,
                    Trading.DireccionOperation.LONG,
                    openOperations.getAvailableBalance()/2,
                    1
            );
        }
    }
}
