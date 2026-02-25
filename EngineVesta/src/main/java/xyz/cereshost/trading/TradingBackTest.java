package xyz.cereshost.trading;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.message.MediaNotification;

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
    public void open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
        // Lo minimo para invertir en Binance
        if (amountUSDT*leverage < 5)return;
        double currentPrice = backTestEngine.getCurrentPrice();
        // Simular el bin y el ask al comprar en mercado
        double realPrice = direccion == DireccionOperation.LONG ? currentPrice + 0.0001 : currentPrice - 0.0001;
        OpenOperationBackTest o = new OpenOperationBackTest(realPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
        o.setEntryTime(backTestEngine.getCurrentTime());
        lastOpenOperation.add(o);
        openOperations.put(o.getUuid() , o);
    }

    @Override
    public void close(ExitReason reason, UUID uuidOpenOperation) {
        closeOperations.add(new CloseOperationBackTest(backTestEngine.getCurrentPrice(), backTestEngine.getCurrentTime(), openOperations.get(uuidOpenOperation).getEntryTime(), reason, uuidOpenOperation));
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

    public void computeCloses() {
        lastOpenOperation.clear();
        for (CloseOperation closeOperation : closeOperations) {
            OpenOperationBackTest open = openOperations.get(closeOperation.getUuidOpenOperation());
            if (open != null) {
                backTestEngine.computeClose(closeOperation, open);
                openOperations.remove(closeOperation.getUuidOpenOperation());
            }
            backTestEngine.getStrategy().closeOperation(closeOperation);
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

        public OpenOperationBackTest(double currentPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            super(currentPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
        }
    }

    public static class CloseOperationBackTest extends CloseOperation {

        public CloseOperationBackTest(double exitPrice, long exitTime, long entryTime, ExitReason reason, UUID uuidOpenOperation) {
            super(exitPrice, exitTime, entryTime, reason, uuidOpenOperation);
        }
    }
}
