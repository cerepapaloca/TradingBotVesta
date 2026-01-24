package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.ListOperations;
import xyz.cereshost.engine.PredictionEngine;

import java.util.List;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements BacktestStrategy {
    @Override
    public BackTestEngine.OpenOperation preProcess(PredictionEngine.@NotNull PredictionResultTP_SL pred, ListOperations openOperations, Candle candle, double balance) {
        // Filtro básico: Ratio mínimo y retornos mínimos para cubrir fees
//        if (!pred.isProfitableSetup() || pred.getTpPercent() < 0.0001) {
//            return null;
//        }
        Vesta.info(pred.getRatio() + "  " + pred.direction() + " " + pred.getTpPercent() + " " + pred.getSlPercent());
        if ((pred.getRatio() > 1 && pred.getRatio() < 3) ) {
            if (openOperations.sizeOpens() == 0) {
                return new BackTestEngine.OpenOperation(
                        candle.close(),
                        pred.getTpPercent() + 0.1,
                        pred.getSlPercent() + 0.3,
                        pred.direction(),
                        balance,
                        60,
                        1
                );
            }
        }

        return null;
    }

    @Override
    public void postProcess(BackTestEngine.TradeResult result) {

    }
}
