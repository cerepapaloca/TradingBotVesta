package xyz.cereshost.vesta.core.trading.backtest;

import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.core.message.MediaNotification;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TradingManager;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.UUID;

public class TradingManagerBackTest implements TradingManager {

    private final HashMap<UUID, BackTestOpenOperation> openOperations = new HashMap<>();
    private final ArrayList<BackTestCloseOperation> closeOperations = new ArrayList<>();

    @Getter
    private final List<BackTestOpenOperation> lastOpenOperation = new ArrayList<>();
    private final BackTestEngine backTestEngine;

    public TradingManagerBackTest(BackTestEngine backTestEngine) {
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
    public @Nullable OpenOperation open(double tpPercent, double slPercent, @NotNull DireccionOperation direccion, double amountUSD, int leverage) {
        // Lo minimo para invertir en Binance
        if (amountUSD *leverage < 5)return null;
        double currentPrice = backTestEngine.getCurrentPrice();
        // Simular el bin y el ask al comprar en mercado
        double realPrice = direccion == DireccionOperation.LONG ? currentPrice + 0.0001 : currentPrice - 0.0001;
        BackTestOpenOperation o = new BackTestOpenOperation(this, realPrice, tpPercent, slPercent, direccion, amountUSD, leverage);
        lastOpenOperation.add(o);
        openOperations.put(o.getUuid() , o);
        return o;
    }

    @Override
    public @Nullable CloseOperation close(ExitReason reason, OpenOperation openOperation) {
        BackTestCloseOperation closeOperation = new BackTestCloseOperation(backTestEngine.getCurrentPrice(), backTestEngine.getCurrentTime(), reason, openOperation);
        closeOperations.add(closeOperation);
        return closeOperation;
    }

    @Override
    public @Nullable LimiteOperation limit(double entryPrice, double tpPercent, double slPercent, @NotNull DireccionOperation direccion, double amountUSD, int leverage) {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void closeForEngine(BackTestCloseOperation backTestCloseOperation){
        closeOperations.add(backTestCloseOperation);
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
        for (TradingManager.OpenOperation openOperation : getOpens()){
            balanceAvailable -= openOperation.getAmountInitUSDT();
        }
        return balanceAvailable;
    }

    @Override
    public double getCurrentPrice() {
        return backTestEngine.getCurrentPrice();
    }

    @Override
    public long getCurrentTime() {
        return backTestEngine.getCurrentTime();
    }

    public void computeCloses() {
        lastOpenOperation.clear();
        for (CloseOperation closeOperation : closeOperations) {
            BackTestOpenOperation open = openOperations.get(closeOperation.getUuid());
            if (open != null) {
                backTestEngine.computeClose(closeOperation, open);
                openOperations.remove(closeOperation.getUuid());
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

    public static class BackTestOpenOperation extends OpenOperation {

        @Getter @Setter
        private double lastExitPrices;
        public BackTestOpenOperation(TradingManager tradingManager, double currentPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            super(tradingManager, currentPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
        }
    }

    public static class BackTestCloseOperation extends CloseOperation {

        public BackTestCloseOperation(double exitPrice, long exitTime, ExitReason reason, OpenOperation openOperation) {
            super(exitPrice, exitTime, reason, openOperation);
        }
    }
}
