package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.engine.PredictionEngine;

public class AlfaStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, Trading operations) {
        for (Trading.OpenOperation o : operations.getOpens()){
            if (o.getCountCandles() >= 60){
                operations.close(Trading.ExitReason.TIMEOUT, o.getUuid());
            }
//            double tpMinimo = (data.feeExit() + data.feeEntry()) * 100;
//            if (o.getCountCandles() >= 30){
//                o.setTpPercent(tpMinimo + 0.1);
//            }
        }
        if (pred.direction() == Trading.DireccionOperation.NEUTRAL) {
            operations.log("Momento no optimo para operar");
            return;
        }

        if ((pred.getRatio() > 1 && pred.getRatio() < 4) && (pred.getTpPercent() > 0.15 && pred.getTpPercent() < 0.4)) {
            if (operations.openSize() == 0) {
                operations.open(pred.getTpPercent(), pred.getSlPercent(), pred.direction(), operations.getAvailableBalance(), 1);
            }else{
                operations.log("Operación ya abierta");
            }
        }else {
            operations.log("No cumple con los mínimos para operar");
        }
    }
}
