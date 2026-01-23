package xyz.cereshost.strategy;

import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.PredictionEngine;

/**
 * Estrategia por defecto: Usa tal cual la predicción de la IA.
 */
public class DefaultStrategy implements BacktestStrategy {
    @Override
    public BackTestEngine.TradeSetup preProcess(PredictionEngine.PredictionResultTP_SL pred, Candle candle, double balance) {
        // Filtro básico: Ratio mínimo y retornos mínimos para cubrir fees
        Vesta.info(pred.slPrice() + " " + pred.tpPrice() + " " + pred.direction());
        if (!pred.isProfitableSetup() || pred.getTpPercent() < 0.0001) {
            return null;
        }

        // Operar con el 100% del balance (Compounding) o monto fijo
        return new BackTestEngine.TradeSetup(
                candle.close(),
                pred.tpPrice(),
                pred.slPrice(),
                "LONG".equals(pred.direction()),
                balance, // Full equity
                1 // Max hold: 60 minutos (ejemplo de multi-vela)
        );
    }

    @Override
    public void postProcess(BackTestEngine.TradeResult result) {
        // Aquí podrías loguear o ajustar variables internas
    }
}
