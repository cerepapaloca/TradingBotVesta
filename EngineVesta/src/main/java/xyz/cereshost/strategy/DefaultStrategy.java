package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import org.tensorflow.op.data.MakeIterator;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.engine.PredictionEngine;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements BacktestStrategy {
    @Override
    public void preProcess(PredictionEngine.@NotNull PredictionResult pred, Trading openOperations, double balance) {

        if (openOperations.openSize() == 0) {
            openOperations.open(
                    pred.getTpPercent(),
                    pred.getSlPercent(),
                    pred.direction(),
                    balance,
                    1
            );
        }
    }

    @Override
    public void postProcess(BackTestEngine.TradeResult result) {

    }
}
