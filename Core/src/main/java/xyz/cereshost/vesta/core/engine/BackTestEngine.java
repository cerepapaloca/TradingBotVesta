package xyz.cereshost.vesta.core.engine;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.common.market.Trade;
import xyz.cereshost.vesta.core.utils.BuilderData;
import xyz.cereshost.vesta.core.strategy.AlfaStrategy;
import xyz.cereshost.vesta.core.strategy.TradingStrategy;
import xyz.cereshost.vesta.core.trading.Trading;
import xyz.cereshost.vesta.core.trading.TradingBackTest;
import xyz.cereshost.vesta.core.utils.ChartUtils;

import java.util.ArrayList;
import java.util.List;
import java.math.BigDecimal;
import java.math.RoundingMode;

@Getter
public class BackTestEngine {

    private final BackTestStats stats;
    private final List<CompleteTrade> extraStats = new ArrayList<>();
    @NotNull private final TradingBackTest operations = new TradingBackTest(this);
    @NotNull private final Market market;
    @NotNull private final TradingStrategy strategy;
    @Nullable private final PredictionEngine engine;
    private double balance = 5;
    private double currentPrice;
    private long currentTime;

    public BackTestEngine(@NotNull Market market, @Nullable PredictionEngine engine, @NotNull TradingStrategy strategy) {
        this.market = market;
        this.engine = engine;
        this.strategy = strategy;
        this.stats = new BackTestStats(market);
        currentPrice = market.getCandleSimples().getFirst().open();

    }

    public BackTestEngine(Market market, PredictionEngine engine) {
        this(market, engine, new AlfaStrategy());
    }

    public BackTestResult run() {
        market.sortd();
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        ChartUtils.showCandleChart("Mercado", allCandles, market.getSymbol());
        return run(allCandles);
    }

    public BackTestResult run(List<Candle> allCandles){

        market.buildTradeCache();

        int totalSamples = allCandles.size();
        int lookBack = engine == null ? 150 : engine.getLookBack();

        // Empezamos donde tenemos datos suficientes
        int startIndex = lookBack + 1;

        // Variables de estado
        double initialBalance = balance;

        // Loop principal
        for (int i = startIndex; i < totalSamples - 1; i++) {
            // Obtener predicción
            List<Candle> window = allCandles.subList(i - lookBack, i + 1);
            PredictionEngine.PredictionResult prediction ;
            if (engine != null) {
                prediction = engine.predictNextPriceDetail(window, market.getSymbol());
                stats.getAllTrades().add(new InCompleteTrade(currentPrice, prediction.getTpPrice(), prediction.getSlPrice(), currentTime));
            }else {
                prediction = null;
            }

            // Consultar estrategia
            strategy.executeStrategy(prediction, window, operations);

            if (operations.getLastOpenOperation().isEmpty())stats.nothing++;
            for (TradingBackTest.OpenOperationBackTest setup : operations.getLastOpenOperation()){
                if (setup != null) {
                    if (setup.getDireccion() != Trading.DireccionOperation.NEUTRAL) {
                        switch (setup.getDireccion()) {
                            case LONG -> stats.longs++;
                            case SHORT -> stats.shorts++;
                        }
                    } else stats.nothing++;
                } else stats.nothing++;
            }

            // Inicia la simulaciónn de una vela de duración

            simulateOneTick(
                    allCandles.get(i + 1),
                    operations
            );
            operations.getOpens().forEach(Trading.OpenOperation::next);
            operations.computeCloses();
        }
        stats.getTradesComplete().addAll(extraStats);
        return new BackTestResult(
                initialBalance, balance, balance - initialBalance, stats.getRoi(),
                stats.totalTrades, stats.getWins(), stats.getLosses(), stats.maxDrawdownPercent,
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
        stats.addComplenteTrade(resultObj, balance);
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

        double netPnL = grossPnL - entryFee - exitFee;
        return netPnL;
    }

    /**
     * Redondeo conservador a 2 decimales:
     * positivo -> hacia abajo, negativo -> mas negativo.
     */
    private static double roundPnlAgainst(double pnl) {
        return BigDecimal.valueOf(pnl)
                .setScale(2, RoundingMode.FLOOR)
                .doubleValue();
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
            Candle candle,
            TradingBackTest operations
    ) {
        long endTime = candle.openTime() + 60_000;

        // Obtener trades reales de este minuto
        List<Trade> trades = market.getTradesInWindow(candle.openTime(), endTime);
        if (trades.isEmpty()) {
            currentPrice = candle.close();
            currentTime = candle.openTime();
            return;
        }

        if (operations.getOpens().isEmpty()) {
            currentPrice = candle.close();
            currentTime = trades.getLast().time();
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
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.LONG_TAKE_PROFIT, openOperation));
                                computeLimit = true;
                                break;

                            }
                            if (price <= openOperation.getSlPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.LONG_STOP_LOSS, openOperation));
                                computeLimit = true;
                            }
                        }
                        case SHORT -> {
                            if (price <= openOperation.getTpPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.SHORT_TAKE_PROFIT, openOperation));
                                computeLimit = true;
                                break;
                            }
                            if (price >= openOperation.getSlPrice()) {
                                operations.closeForEngine(new TradingBackTest.CloseOperationBackTest(price, t.time(), openOperation.getEntryTime(), Trading.ExitReason.SHORT_STOP_LOSS, openOperation));
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
        private List<InCompleteTrade> allTrades = new ArrayList<>();

        public BackTestStats(Market market) {
            this.market = market;
        }

        public int getTrades(Trading.ExitReason reason) {
            int i = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(reason)) i++;
            return i;
        }

        public double getRoiTPMinLong(){
            double roi = 1;
            for (TradeResult tr : trades) if (tr.reason().equals(Trading.ExitReason.LONG_TAKE_PROFIT)) {
                if (roi > tr.pnlPercent()){
                    roi = tr.pnlPercent();
                }
            }
            return roi*100;
        }

        public double getRoiTPMinShort(){
            double roi = 1;
            for (TradeResult tr : trades) if (tr.reason().equals(Trading.ExitReason.SHORT_TAKE_PROFIT)) {
                if (roi > tr.pnlPercent()){
                    roi = tr.pnlPercent();
                }
            }
            return roi*100;
        }

        public double getRoiAvg(){
            double roi = 0;
            for (TradeResult tr : trades){
                roi += tr.pnlPercent()*100;
            }
            return roi/trades.size();
        }

        public double getRoiAvg(Trading.ExitReason reason) {
            double roi = 0;
            int count = 0;
            for (TradeResult tr : trades){
                if (reason.equals(tr.reason())) {
                    roi += tr.pnlPercent()*100;
                    count++;
                }
            }
            return roi/count;
        }


        // Para calcular Hold Time promedio

        public double getHoldAvg(){
            long totalHoldTimeMillis = 0;
            for (TradeResult tr : trades) totalHoldTimeMillis += (tr.exitTime - tr.entryTime);
            return (double) totalHoldTimeMillis / (double) trades.size();
        }

        public double getHoldAvgWins(){
            long totalHoldTimeMillis = 0;
            for (TradeResult tr : trades) if (tr.pnlPercent > 0) totalHoldTimeMillis += (tr.exitTime - tr.entryTime);
            return (double) totalHoldTimeMillis / (double) getWins();
        }

        public double getHoldAvgLoss(){
            long totalHoldTimeMillis = 0;
            for (TradeResult tr : trades) if (tr.pnlPercent < 0) totalHoldTimeMillis += (tr.exitTime - tr.entryTime);
            return (double) totalHoldTimeMillis / (double) getLosses();
        }

        public int getLosses(){
            int i = 0;
            for (TradeResult tr : trades) if (tr.pnlPercent < 0 ) i++;
            return i;
        }

        public int getWins(){
            int i = 0;
            for (TradeResult tr : trades) if (tr.pnlPercent > 0 ) i++;
            return i;
        }

        public void addComplenteTrade(TradeResult result, double currentBalance) {
            trades.add(result);
            if (totalTrades == 0) {
                initialBalance = currentBalance - result.pnl; // Reconstruir inicial
                peakBalance = initialBalance;
            }

            totalTrades++;
            totalPnL += result.pnl;

            if (result.reason == Trading.ExitReason.TIMEOUT) timeouts++;

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

        public double getRoiWins() {
            double roi = 0;
            for (TradeResult tr : trades) if (tr.pnlPercent > 0) roi += tr.pnlPercent*100;
            return roi / getWins();
        }

        public double getRoiLosses() {
            double roi = 0;
            for (TradeResult tr : trades) if (tr.pnlPercent < 0) roi += tr.pnlPercent*100;
            return roi / getLosses();
        }

        public double getRoiTimeOut() {
            double roi = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(Trading.ExitReason.TIMEOUT)) roi += tr.pnlPercent;
            return roi*100;
        }

        public double getRoiLong() {
            double roi = 0;
            for (TradeResult tr : trades)
                if (tr.reason().equals(Trading.ExitReason.LONG_STOP_LOSS) || tr.reason().equals(Trading.ExitReason.LONG_TAKE_PROFIT))
                    roi += tr.pnlPercent;
            return roi*100;
        }

        public double getRoiShort() {
            double roi = 0;
            for (TradeResult tr : trades)
                if (tr.reason().equals(Trading.ExitReason.SHORT_STOP_LOSS) || tr.reason().equals(Trading.ExitReason.SHORT_TAKE_PROFIT))
                    roi += tr.pnlPercent;
            return roi*100;
        }

        public double getRoiStrategy() {
            double roi = 0;
            for (TradeResult tr : trades) {
                if (tr.reason().equals(Trading.ExitReason.STRATEGY)) roi += tr.pnlPercent;
            }
            return roi*100;
        }

        public double getRatioAvg() {
            return TradesComplete.stream().mapToDouble(CompleteTrade::getRatio).average().orElse(0);
        }

        public double getRatioMin(){
            return TradesComplete.stream().mapToDouble(CompleteTrade::getRatio).min().orElse(0);
        }

        public double getRatioMax(){
            return TradesComplete.stream().mapToDouble(CompleteTrade::getRatio).max().orElse(0);
        }
    }

    public record TradeResult(double pnl, double pnlPercent, double exitPrice, Trading.ExitReason reason, long entryTime, long exitTime) {}

    @EqualsAndHashCode(callSuper = true)
    @Data
    public static final class CompleteTrade extends InCompleteTrade {

        private final float tpPercent;
        private final float slPercent;
        private final Trading.DireccionOperation direction;
        private final double exitPrice;
        private final Trading.ExitReason exitReason;
        private final long exitTime;
        private final double pnl;
        private final float balance;
        private final float ratio;
        private final float pnlPercent;

        public CompleteTrade(
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
        ) {
            super(entryPrice, tpPrice, slPrice, entryTime);
            this.tpPercent = tpPercent;
            this.slPercent = slPercent;
            this.direction = direction;
            this.exitPrice = exitPrice;
            this.exitReason = exitReason;
            this.exitTime = exitTime;
            this.pnl = pnl;
            this.balance = balance;
            this.ratio = ratio;
            this.pnlPercent = pnlPercent;
        }
    }

    @Data
    public static class InCompleteTrade {
        private final double entryPrice;
        private final double tpPrice;
        private final double slPrice;
        private final long entryTime;
    }

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
