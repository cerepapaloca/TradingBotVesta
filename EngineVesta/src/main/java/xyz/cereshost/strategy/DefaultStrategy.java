package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.PredictionEngine;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements BacktestStrategy {
    @Override
    public BackTestEngine.TradeSetup preProcess(PredictionEngine.@NotNull PredictionResultTP_SL pred, Candle candle, double balance) {
        // Filtro básico: Ratio mínimo y retornos mínimos para cubrir fees
        if (!pred.isProfitableSetup() || pred.getTpPercent() < 0.0001) {
            return null;
        }

        // Operar con el 100% del balance (Compounding) o monto fijo
        return new BackTestEngine.TradeSetup(
                candle.close(),
                pred.tpPrice(),
                pred.slPrice(),
                pred.direction(),
                balance, // Full equity
                5, // Max hold: 60 minutos (ejemplo de multi-vela)
                1
        );
    }

    @Override
    public void postProcess(BackTestEngine.TradeResult result) {

    }
}
