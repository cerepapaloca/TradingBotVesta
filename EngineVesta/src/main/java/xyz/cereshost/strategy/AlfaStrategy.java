package xyz.cereshost.strategy;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.engine.PredictionEngine;

public class AlfaStrategy implements BacktestStrategy {
    @Override
    public void preProcess(PredictionEngine.@NotNull PredictionResult pred, Trading operations, double balance) {
        for (Trading.OpenOperation o : operations.getOpens()){
            if (o.getCountCandles() >= 60){
                operations.close(Trading.ExitReason.TIMEOUT, o.getUuid());
            }
//            double tpMinimo = (data.feeExit() + data.feeEntry()) * 100;
//            if (o.getCountCandles() >= 30){
//                o.setTpPercent(tpMinimo + 0.1);
//            }
        }
        //if (pred.direction() == Trading.DireccionOperation.SHORT) return;

        if ((pred.getRatio() > 1 && pred.getRatio() < 2) && (pred.getTpPercent() > 0.15 && pred.getTpPercent() < 0.4)) {
            if (operations.openSize() == 0 && pred.direction() !=  Trading.DireccionOperation.NEUTRAL) {
                operations.open(pred.getTpPercent(), pred.getSlPercent() + 0.2, pred.direction(), balance,1);
            }
        }
    }

    @Override
    public void postProcess(BackTestEngine.TradeResult result) {

    }
}
