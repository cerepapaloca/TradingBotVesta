package xyz.cereshost.vesta.core.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.Trading;
import xyz.cereshost.vesta.core.engine.PredictionEngine;

import java.util.List;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, List<Candle> visibleCandles, Trading openOperations) {
        for (Trading.OpenOperation op : openOperations.getOpens()) if (op.getMinutesOpen() >= 1) openOperations.close(Trading.ExitReason.STRATEGY, op);
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

    @Override
    public void closeOperation(Trading.CloseOperation closeOperation, Trading operations) {

    }
}
