package xyz.cereshost.engine;

import lombok.Data;
import lombok.Getter;
import lombok.Setter;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.strategy.BacktestStrategy;
import xyz.cereshost.strategy.DefaultStrategy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static xyz.cereshost.engine.BackTestEngine.DireccionOperation.SHORT;

@Getter
public class BackTestEngine {

    private final BackTestStats stats = new BackTestStats();
    private final List<ExtraDataPlot> extraStats = new ArrayList<>();
    private final ListOperations operations = new ListOperations(this);
    private final Market market;
    private final PredictionEngine engine;
    private final BacktestStrategy strategy;
    private double balance = 1000.0;
    private final double feeMaker;
    private final double feeTaker;

    public BackTestEngine(Market market, PredictionEngine engine, BacktestStrategy strategy) {
        this.market = market;
        this.engine = engine;
        this.strategy = strategy;
        if (market.getSymbol().endsWith("USDT")) {
            feeMaker = 0.0002;
            feeTaker = 0.0005;
        } else {
            feeMaker = 0;
            feeTaker = 0.0004;
        }
    }

    public BackTestEngine(Market market, PredictionEngine engine) {
        this(market, engine, new DefaultStrategy());
    }

    public BackTestResult run(){
        // 1. Preparar datos
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        allCandles.sort(Comparator.comparingLong(Candle::openTime));
        market.buildTradeCache(); // Crucial para velocidad

        int totalSamples = allCandles.size();
        int lookBack = engine.getLookBack();

        // Empezamos donde tenemos datos suficientes
        int startIndex = lookBack + 1;

        // Variables de estado
        double initialBalance = balance;


        // 2. Loop principal (Walk-Forward)
        // Usamos un loop 'i' que podemos avanzar manualmente si una operación dura varias velas
        for (int i = startIndex; i < totalSamples - 1; i++) {

            Candle currentCandle = allCandles.get(i);

            // A. Obtener predicción
            List<Candle> window = allCandles.subList(i - lookBack, i + 1);
            PredictionEngine.PredictionResultTP_SL prediction = engine.predictNextPriceDetail(window, market.getSymbol());

            // B. Consultar estrategia
            OpenOperation setup = strategy.preProcess(prediction, operations,currentCandle, balance);

            if (setup != null && setup.getDireccion() != DireccionOperation.NEUTRAL) {
                setup.setOpenTime(currentCandle.openTime());
                operations.add(setup);
                switch (setup.getDireccion()) {
                    case LONG -> stats.longs++;
                    case SHORT -> stats.shorts++;
                }
            } else {
                stats.nothing++;
            }

            // Inicia la simulaciónn de una vela de duración
            simulateOneTick(
                    market,
                    allCandles,
                    i + 1, // Empezamos a buscar en la siguiente vela
                    // Debe ser una lista mutable
                    operations
            );
            operations.computeCloses();
        }
        stats.getExtraDataPlot().addAll(extraStats);
        return new BackTestResult(
                initialBalance, balance, balance - initialBalance, stats.getRoi(),
                stats.totalTrades, stats.wins, stats.losses, stats.maxDrawdownPercent,
                stats
        );
    }

    public void computeFinal(CloseOperation closeOperation) {
        if (closeOperation.reason() == ExitReason.NO_DATA_ERROR) {
            return;
        }

        // D. Calcular PnL Real (con fees)
        // Fee se cobra al entrar (sobre entry) y al salir (sobre exit)
        OpenOperation operation = closeOperation.openOperationLastEstate();
        double positionSize = operation.amountInitUSDT * operation.leverage; // notional
        double qty = positionSize / operation.entryPrice;

        double entryFee = positionSize * feeMaker;
        double exitNotional = qty * closeOperation.exitPrice;
        double exitFee = exitNotional * feeTaker;

        double grossPnL = (closeOperation.exitPrice - operation.entryPrice) * qty;
        double netPnL = grossPnL - entryFee - exitFee;

        double pnlPercent = netPnL / operation.amountInitUSDT; // sobre margen

        // E. Actualizar Balance
        balance += netPnL;
        if (balance < 0) balance = 0; // Quiebra

        // F. Registrar estadísticas
        TradeResult resultObj = new TradeResult(netPnL, pnlPercent, closeOperation.exitPrice, closeOperation.reason, closeOperation.openOperationLastEstate().getOpenTime(), closeOperation.exitTime);
        strategy.postProcess(resultObj);
        stats.addTrade(resultObj, balance);
        extraStats.add(new ExtraDataPlot((float) pnlPercent,
                (float) balance,
                (float) (operation.getTpPercent() / operation.getSlPercent()),
                (float) operation.getTpPercent(),
                (float) operation.getSlPercent()
        ));
    }

    /**
     * Simula la vida de un trade a través del tiempo (velas) y trades (ticks).
     */
    private static void simulateOneTick(
            Market market,
            List<Candle> allCandles,
            int startIndex,
            ListOperations operations
    ) {

        int candlesChecked = 0;

        Candle candle = allCandles.get(startIndex);
        long endTime = candle.openTime() + 60_000;

        // Obtener trades reales de este minuto
        List<Trade> trades = market.getTradesInWindow(candle.openTime(), endTime);

        if (trades.isEmpty()) {
            // Si no hay trades no hay resultados ya que no se cerro nigun ninguna operación
            return;
        }
        double lastPrice = 0;

        for (OpenOperation openOperation : operations.iteratorOpens()) {
            // Analiza cada operacion
            for (Trade t : trades) {
                lastPrice = t.price();
                double price = t.price();
                openOperation.setLastExitPrices(price);
                boolean computeLimit = false;
                switch (openOperation.getDireccion()) {
                    case LONG -> {
                        if (price >= openOperation.getTpPrice()) {
                            operations.close(new CloseOperation(price, ExitReason.LONG_TAKE_PROFIT, t.time(), candlesChecked, openOperation));
                            computeLimit = true;
                            break;

                        }
                        if (price <= openOperation.getSlPrice()) {
                            operations.close(new CloseOperation(price, ExitReason.LONG_STOP_LOSS, t.time(), candlesChecked, openOperation));
                            computeLimit = true;
                        }
                    }
                    case SHORT -> {
                        if (price <= openOperation.getTpPrice()) {
                            operations.close(new CloseOperation(price, ExitReason.SHORT_TAKE_PROFIT, t.time(), candlesChecked, openOperation));
                            computeLimit = true;
                            break;
                        }
                        if (price >= openOperation.getSlPrice()) {
                            operations.close(new CloseOperation(price, ExitReason.SHORT_STOP_LOSS, t.time(), candlesChecked, openOperation));
                            computeLimit = true;
                        }

                    }
                }
                if (computeLimit) break;
            }
        }
        for (OpenOperation openOperation : operations.iteratorOpens()) {
            // Se superó el límite?
            if (openOperation.next())  {
                operations.close(new CloseOperation(lastPrice, ExitReason.TIMEOUT, 0, candlesChecked, openOperation));
            }
        }
        return;
    }


    @Getter
    @Setter
    public static class BackTestStats {
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
        private List<ExtraDataPlot> extraDataPlot = new ArrayList<>();

        public int getTakeProfit() {
            int i = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(ExitReason.LONG_TAKE_PROFIT)) i++;
            return i;
        }

        public int getStopLoss() {
            int i = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(ExitReason.LONG_STOP_LOSS)) i++;
            return i;
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

            if (result.reason == ExitReason.TIMEOUT) timeouts++;

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
                if (tr.reason().equals(ExitReason.TIMEOUT)) roi += tr.pnlPercent;
            }
            return roi*100;
        }

        public double getRoiLong() {
            double roi = 0;
            for (TradeResult tr : trades) {
                if (tr.reason().equals(ExitReason.LONG_STOP_LOSS) || tr.reason().equals(ExitReason.LONG_TAKE_PROFIT)) roi += tr.pnlPercent;
            }
            return roi*100;
        }

        public double getRoiShort() {
            double roi = 0;
            for (TradeResult tr : trades) {
                if (tr.reason().equals(ExitReason.SHORT_STOP_LOSS) || tr.reason().equals(ExitReason.SHORT_TAKE_PROFIT)) roi += tr.pnlPercent;
            }
            return roi*100;
        }
    }

    public enum ExitReason {
        LONG_TAKE_PROFIT,
        LONG_STOP_LOSS,
        SHORT_TAKE_PROFIT,
        SHORT_STOP_LOSS,
        TIMEOUT,
        NO_DATA_ERROR
    }

    public record CloseOperation(double exitPrice, ExitReason reason, long exitTime, int candlesConsumed, OpenOperation openOperationLastEstate) {}

    @Data
    public static final class OpenOperation {
        private double tpPercent;
        private double slPercent;
        private int maxCandles;
        private int countCandles = 0;
        private double lastExitPrices;

        private long openTime;

        private final double entryPrice;
        private final DireccionOperation direccion;
        private final double amountInitUSDT;
        private final int leverage;

        // Ojo el TP y SL en Porcentajes ABS sin apalancar
        public OpenOperation(double entryPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int maxCandles, int leverage) {
            this.tpPercent = tpPercent;
            this.slPercent = slPercent;
            this.maxCandles = maxCandles;

            this.entryPrice = entryPrice;
            this.direccion = direccion;
            this.amountInitUSDT = amountUSDT;
            this.leverage = leverage;
        }

        public double getSlPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? slPercent : -slPercent)*0.01));
        }

        public double getTpPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? -tpPercent : tpPercent)*0.01));
        }

        public boolean next() {
            if (maxCandles <= 0) return false;
            countCandles++;
            return countCandles >= maxCandles;
        }
    }

    public enum DireccionOperation {
        SHORT,
        LONG,
        // OJO no se puede operar con neutral solo es una forma de identificar operacion sin movimiento ósea nada
        NEUTRAL
    }

    public record TradeResult(double pnl, double pnlPercent, double exitPrice, ExitReason reason, long entryTime, long exitTime) {}

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

    public record ExtraDataPlot(float pnlPercent, float balance, float ratio, float tpPercent, float slPercent) {}
}