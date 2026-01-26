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

        if (openOperations.openSize() == 0) {
            openOperations.open(
                    pred.getTpPercent(),
                    pred.getSlPercent(),
                    pred.direction(),
                    openOperations.getAvailableBalance()/2,
                    1
            );
        }
    }
}
