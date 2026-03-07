package xyz.cereshost.vesta.core.trading;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.core.engine.BackTestEngine;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.core.message.MediaNotification;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;

public class TradingBackTest implements Trading {

    private final HashMap<UUID, OpenOperationBackTest> openOperations = new HashMap<>();
    private final ArrayList<CloseOperationBackTest> closeOperations = new ArrayList<>();

    @Getter
    private final List<OpenOperationBackTest> lastOpenOperation = new ArrayList<>();
    private final BackTestEngine backTestEngine;

    public TradingBackTest(BackTestEngine backTestEngine) {
        this.backTestEngine = backTestEngine;
    }

    @Override
    public int openSize(){
        return openOperations.size();
    }

    @Override
    public int closeSize(){
        return closeOperations.size();
    }

    @Override
    public OpenOperation open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
        // Lo minimo para invertir en Binance
        if (amountUSDT*leverage < 5)return null;
        double currentPrice = backTestEngine.getCurrentPrice();
        // Simular el bin y el ask al comprar en mercado
        double realPrice = direccion == DireccionOperation.LONG ? currentPrice + 0.0001 : currentPrice - 0.0001;
        OpenOperationBackTest o = new OpenOperationBackTest(this, realPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
        o.setEntryTime(backTestEngine.getCurrentTime());
        lastOpenOperation.add(o);
        openOperations.put(o.getUuid() , o);
        return o;
    }

    @Override
    public void close(ExitReason reason, OpenOperation openOperation) {
        closeOperations.add(new CloseOperationBackTest(backTestEngine.getCurrentPrice(), backTestEngine.getCurrentTime(), openOperation.getEntryTime(), reason, openOperation));
    }

    public void closeForEngine(CloseOperationBackTest closeOperationBackTest){
        closeOperations.add(closeOperationBackTest);
    }

    @Override
    public @NotNull List<OpenOperation> getOpens() {
        return new ArrayList<>(openOperations.values());
    }

    @Override
    public @NotNull List<CloseOperation> getCloses() {
        return new ArrayList<>(closeOperations);
    }

    @Override
    public Market getMarket() {
        return backTestEngine.getMarket();
    }

    @Override
    public double getAvailableBalance() {
        double balanceAvailable = backTestEngine.getBalance();
        for (Trading.OpenOperation openOperation : getOpens()){
            balanceAvailable -= openOperation.getAmountInitUSDT();
        }
        return balanceAvailable;
    }

    @Override
    public double getCurrentPrice() {
        return backTestEngine.getCurrentPrice();
    }

    public void computeCloses() {
        lastOpenOperation.clear();
        for (CloseOperation closeOperation : closeOperations) {
            OpenOperationBackTest open = openOperations.get(closeOperation.getUuidOpenOperation());
            if (open != null) {
                backTestEngine.computeClose(closeOperation, open);
                openOperations.remove(closeOperation.getUuidOpenOperation());
            }
            backTestEngine.getStrategy().closeOperation(closeOperation, this);
        }
        closeOperations.clear();
    }

    @Override
    public @NotNull MediaNotification getMediaNotification() {
        return MediaNotification.empty();
    }

    @Override
    public void setMediaNotification(@NotNull MediaNotification mediaNotification) {

    }

    public static class OpenOperationBackTest extends OpenOperation {

        public OpenOperationBackTest(Trading trading, double currentPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            super(trading, currentPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
        }
    }

    public static class CloseOperationBackTest extends CloseOperation {

        public CloseOperationBackTest(double exitPrice, long exitTime, long entryTime, ExitReason reason, OpenOperation openOperation) {
            super(exitPrice, exitTime, entryTime, reason, openOperation);
        }
    }
}
