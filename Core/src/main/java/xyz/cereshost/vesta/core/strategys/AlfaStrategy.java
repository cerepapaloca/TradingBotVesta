package xyz.cereshost.vesta.core.strategys;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.core.ia.PredictionEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.util.List;

public class AlfaStrategy implements TradingStrategy {
    @Override
    public void executeStrategy(PredictionEngine.@NotNull PredictionResult pred, List<Candle> visibleCandles, TradingManager operations) {
        for (TradingManager.OpenOperation o : operations.getOpens()){
            if (o.getMinutesOpen() >= 60){
                operations.close(TradingManager.ExitReason.TIMEOUT, o);
            }
//            double tpMinimo = (data.feeExit() + data.feeEntry()) * 100;
//            if (o.getCountCandles() >= 30){
//                o.setTpPercent(tpMinimo + 0.1);
//            }
        }
        if (pred.directionOperation() == DireccionOperation.NEUTRAL) {
            operations.log("Momento no optimo para operar");
            return;
        }

        if ((pred.getRatio() > 1 && pred.getRatio() < 4) && (pred.getTpPercent() > 0.15 && pred.getTpPercent() < 0.4)) {
            if (operations.openSize() == 0) {
                operations.open(pred.getTpPercent(), pred.getSlPercent(), pred.directionOperation(), operations.getAvailableBalance(), 1);
            }else{
                operations.log("Operación ya abierta");
            }
        }else {
            operations.log("No cumple con los mínimos para operar");
        }
    }

    @Override
    public void closeOperation(TradingManager.CloseOperation closeOperation, TradingManager operations) {

    }
}
