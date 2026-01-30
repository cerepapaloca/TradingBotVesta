package xyz.cereshost.engine;

import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.strategy.AlfaStrategy;
import xyz.cereshost.strategy.TradingStrategy;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.trading.TradingBackTest;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

@Getter
public class BackTestEngine {

    private final BackTestStats stats;
    private final List<CompleteTrade> extraStats = new ArrayList<>();
    private final TradingBackTest operations = new TradingBackTest(this);
    private final Market market;
    private final PredictionEngine engine;
    private final TradingStrategy strategy;
    private double balance = 1000.0;
    private double currentPrice;
    private long currentTime;

    public BackTestEngine(Market market, PredictionEngine engine, TradingStrategy strategy) {
        this.market = market;
        this.engine = engine;
        this.strategy = strategy;
        this.stats = new BackTestStats(market);

    }

    public BackTestEngine(Market market, PredictionEngine engine) {
        this(market, engine, new AlfaStrategy());
    }

    public BackTestResult run(){
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        allCandles.sort(Comparator.comparingLong(Candle::openTime));

        market.buildTradeCache(); // Crucial para velocidad

        int totalSamples = allCandles.size();
        int lookBack = engine.getLookBack();

        // Empezamos donde tenemos datos suficientes
        int startIndex = lookBack + 1;

        // Variables de estado
        double initialBalance = balance;


        // Loop principal
        for (int i = startIndex; i < totalSamples - 1; i++) {

            // Obtener predicción
            List<Candle> window = allCandles.subList(i - lookBack, i + 1);
            PredictionEngine.PredictionResult prediction = engine.predictNextPriceDetail(window, market.getSymbol());



            // Consultar estrategia
            strategy.executeStrategy(prediction, operations);

            if (operations.getLastOpenOperation().isEmpty()){
                stats.nothing++;
            }

            for (TradingBackTest.OpenOperationBackTest setup : operations.getLastOpenOperation()){
                if (setup != null && setup.getDireccion() != Trading.DireccionOperation.NEUTRAL) {
                    switch (setup.getDireccion()) {
                        case LONG -> stats.longs++;
                        case SHORT -> stats.shorts++;
                    }
                } else {
                    stats.nothing++;
                }
            }

            // Inicia la simulaciónn de una vela de duración
            simulateOneTick(
                    market,
                    allCandles,
                    i + 1, // Empezamos a buscar en la siguiente vela
                    // Debe ser una lista mutable
                    operations
            );
            operations.getOpens().forEach(Trading.OpenOperation::next);
            operations.computeCloses();
        }
        stats.getTradesComplete().addAll(extraStats);
        return new BackTestResult(
                initialBalance, balance, balance - initialBalance, stats.getRoi(),
                stats.totalTrades, stats.wins, stats.losses, stats.maxDrawdownPercent,
                stats
        );
    }

    public void computeClose(Trading.@NotNull CloseOperation closeOperation, Trading.OpenOperation openOperation) {
        if (closeOperation.getReason() == Trading.ExitReason.NO_DATA_ERROR) {
            return;
        }

        // D. Calcular PnL Real (con fees)
        // Fee se cobra al entrar (sobre entry) y al salir (sobre exit)
        double netPnL = getNetPnL(closeOperation, openOperation);

        double pnlPercent = netPnL / openOperation.getAmountInitUSDT(); // sobre margen

        balance += netPnL;
        if (balance < 0) balance = 0; // Quiebra
//        Vesta.info(openOperation.getDireccion().name() + " " + closeOperation.getReason().name());
//        LockSupport.parkNanos((long) 3e+9);

        // F. Registrar estadísticas
        TradeResult resultObj = new TradeResult(netPnL, pnlPercent, closeOperation.getExitPrice(), closeOperation.getReason(), closeOperation.getEntryTime(), closeOperation.getExitTime());
        stats.addTrade(resultObj, balance);
        extraStats.add(new CompleteTrade(
                openOperation.getEntryPrice(),
                openOperation.getTpPrice(),
                openOperation.getSlPrice(),
                (float) openOperation.getTpPercent(),
                (float) openOperation.getSlPercent(),
                openOperation.getDireccion(),
                closeOperation.getExitPrice(),
                closeOperation.getReason(),
                closeOperation.getEntryTime(),
                closeOperation.getExitTime(),
                netPnL,
                (float) balance,
                (float) (openOperation.getTpPercent() / openOperation.getSlPercent()),
                (float) pnlPercent
        ));
    }

    private double getNetPnL(Trading.@NotNull CloseOperation closeOperation, Trading.OpenOperation openOperation) {
        double positionSize = openOperation.getAmountInitUSDT() * openOperation.getLeverage(); // notional
        double qty = positionSize / openOperation.getEntryPrice();

        double entryFee = positionSize * market.getFeedMaker();
        double exitNotional = qty * closeOperation.getExitPrice();
        double exitFee = exitNotional * market.getFeedTaker();

        double grossPnL = getGrossPnL(closeOperation, openOperation, qty);

        return grossPnL - entryFee - exitFee;
    }

    private static double getGrossPnL(Trading.@NotNull CloseOperation closeOperation, Trading.OpenOperation openOperation, double qty) {
        double grossPnL;
        if (openOperation.getDireccion() == Trading.DireccionOperation.LONG) {
            // LONG
            grossPnL = (closeOperation.getExitPrice() - openOperation.getEntryPrice()) * qty;
        } else if (openOperation.getDireccion() == Trading.DireccionOperation.SHORT) {
            // SHORT
            grossPnL = (openOperation.getEntryPrice() - closeOperation.getExitPrice()) * qty;
        } else {
            // NEUTRAL
            grossPnL = 0;
        }
        return grossPnL;
    }

    /**
     * Simula la vida de un trade a través del tiempo (velas) y trades (ticks).
     */
    private void simulateOneTick(
            Market market,
            List<Candle> allCandles,
            int startIndex,
            TradingBackTest operations
    ) {
        Candle candle = allCandles.get(startIndex);
        long endTime = candle.openTime() + 60_000;

        // Obtener trades reales de este minuto


        List<Trade> trades = market.getTradesInWindow(candle.openTime(), endTime);
        if (trades.isEmpty()) {
            return;
        }

        if (operations.getOpens().isEmpty()) {
            currentPrice = candle.close();
            currentTime = trades.get(trades.size() -1).time();
        }else {

            for (Trading.OpenOperation openOperation : operations.getOpens()) {
                // Analiza cada operacion
                for (Trade t : trades) {
                    currentPrice = t.price();
                    currentTime = t.time();
                    double price = t.price();
                    openOperation.setLastExitPrices(price);
                    boolean computeLimit = false;
                    switch (openOperation.getDireccion()) {
                        case LONG -> {
                            if (price >= openOperation.getTpPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.LONG_TAKE_PROFIT, openOperation.getUuid()));
                                computeLimit = true;
                                break;

                            }
                            if (price <= openOperation.getSlPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.LONG_STOP_LOSS, openOperation.getUuid()));
                                computeLimit = true;
                            }
                        }
                        case SHORT -> {
                            if (price <= openOperation.getTpPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.SHORT_TAKE_PROFIT, openOperation.getUuid()));
                                computeLimit = true;
                                break;
                            }
                            if (price >= openOperation.getSlPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.SHORT_STOP_LOSS, openOperation.getUuid()));
                                computeLimit = true;
                            }

                        }
                    }
                    if (computeLimit) break;
                }
            }
        }
    }


    @Getter
    @Setter
    public static class BackTestStats {
        private final Market market;
        private int totalTrades = 0;
        private int wins = 0;
        private int losses = 0;
        private int timeouts = 0;
        private int longs = 0;
        private int shorts = 0;
        private int nothing = 0;

        double maxDrawdownPercent = 0.0;
        double peakBalance = 0.0;
        double currentDrawdown = 0.0;
        double totalPnL = 0.0;
        double initialBalance = 0.0;

        private List<TradeResult> trades = new ArrayList<>();
        private List<CompleteTrade> TradesComplete = new ArrayList<>();

        public BackTestStats(Market market) {
            this.market = market;
        }

        public int getTrades(Trading.ExitReason reason) {
            int i = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(reason)) i++;
            return i;
        }

        public double getRoiTPMinLong(){
            double roi = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(Trading.ExitReason.LONG_TAKE_PROFIT)) {
                if (roi > tr.pnlPercent()){
                    roi = tr.pnlPercent();
                }
            }
            return roi*100;
        }

        public double getRoiTPMinShort(){
            double roi = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(Trading.ExitReason.SHORT_TAKE_PROFIT)) {
                if (roi > tr.pnlPercent()){
                    roi = tr.pnlPercent();
                }
            }
            return roi*100;
        }


        // Para calcular Hold Time promedio
        long totalHoldTimeMillis = 0;

        public void addTrade(TradeResult result, double currentBalance) {
            trades.add(result);
            if (totalTrades == 0) {
                initialBalance = currentBalance - result.pnl; // Reconstruir inicial
                peakBalance = initialBalance;
            }

            totalTrades++;
            totalPnL += result.pnl;

            if (result.pnl > 0) wins++;
            else losses++;

            if (result.reason == Trading.ExitReason.TIMEOUT) timeouts++;

            if (result.exitTime > 0 && result.entryTime > 0) {
                totalHoldTimeMillis += (result.exitTime - result.entryTime);
            }

            // Drawdown calculation
            if (currentBalance > peakBalance) {
                peakBalance = currentBalance;
                currentDrawdown = 0;
            } else {
                double dd = (peakBalance - currentBalance) / peakBalance; // 0.10 = 10%
                if (dd > maxDrawdownPercent) {
                    maxDrawdownPercent = dd;
                }
            }
        }

        public double getRoi() {
            return initialBalance > 0 ? (totalPnL / initialBalance) * 100 : 0;
        }

        public double getRoiTimeOut() {
            double roi = 0;
            for (TradeResult tr : trades) {
                if (tr.reason().equals(Trading.ExitReason.TIMEOUT)) roi += tr.pnlPercent;
            }
            return roi*100;
        }

        public double getRoiLong() {
            double roi = 0;
            for (TradeResult tr : trades) {
                if (tr.reason().equals(Trading.ExitReason.LONG_STOP_LOSS) || tr.reason().equals(Trading.ExitReason.LONG_TAKE_PROFIT)) roi += tr.pnlPercent;
            }
            return roi*100;
        }

        public double getRoiShort() {
            double roi = 0;
            for (TradeResult tr : trades) {
                if (tr.reason().equals(Trading.ExitReason.SHORT_STOP_LOSS) || tr.reason().equals(Trading.ExitReason.SHORT_TAKE_PROFIT)) roi += tr.pnlPercent;
            }
            return roi*100;
        }

        public double getRatioAvg() {
            return TradesComplete.stream().mapToDouble(CompleteTrade::ratio).average().orElse(0);
        }

        public double getRatioMin(){
            return TradesComplete.stream().mapToDouble(CompleteTrade::ratio).min().orElse(0);
        }

        public double getRatioMax(){
            return TradesComplete.stream().mapToDouble(CompleteTrade::ratio).max().orElse(0);
        }
    }

    public record TradeResult(double pnl, double pnlPercent, double exitPrice, Trading.ExitReason reason, long entryTime, long exitTime) {}

    public record CompleteTrade(
            double entryPrice,
            double tpPrice,
            double slPrice,
            float tpPercent,
            float slPercent,
            Trading.DireccionOperation direction,
            double exitPrice,
            Trading.ExitReason exitReason,
            long entryTime,
            long exitTime,
            double pnl,
            float balance,
            float ratio,
            float pnlPercent
    ) {}

    // Mantener compatibilidad con tu código existente
    public record BackTestResult(
            double initialBalance,
            double finalBalance,
            double netPnL,
            double roiPercent,
            int totalTrades,
            int winTrades,
            int lossTrades,
            double maxDrawdown,
            BackTestStats stats
    ) {}

}