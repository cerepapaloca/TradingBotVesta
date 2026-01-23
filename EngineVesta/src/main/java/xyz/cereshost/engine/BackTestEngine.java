package xyz.cereshost.engine;

import lombok.Getter;
import lombok.Setter;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.strategy.BacktestStrategy;
import xyz.cereshost.strategy.DefaultStrategy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class BackTestEngine {

    // --- Clases Internas para Estrategia (definidas arriba, pero incluidas aquí para portabilidad) ---
    // (Puedes moverlas a archivos separados si prefieres)

    public enum ExitReason { TAKE_PROFIT, STOP_LOSS, TIMEOUT, NO_DATA_ERROR }

    public record TradeSetup(double entryPrice, double tpPrice, double slPrice, PredictionEngine.DireccionOperation direccion, double amountUsdt, int maxCandles, int leverage) {}

    public record TradeResult(double pnl, double pnlPercent, double exitPrice, ExitReason reason, long entryTime, long exitTime) {}

    // --- Motor Principal ---

    public static BackTestResult runBacktest(Market market, PredictionEngine engine) {
        return runBacktest(market, engine, new DefaultStrategy());
    }

    public static BackTestResult runBacktest(Market market, PredictionEngine engine, BacktestStrategy strategy) {
        Vesta.info("⚙️ Iniciando Backtest Avanzado (Tick-Replay + Multi-Candle) para " + market.getSymbol());

        // 1. Preparar datos
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        allCandles.sort(Comparator.comparingLong(Candle::openTime));
        market.buildTradeCache(); // Crucial para velocidad

        int totalSamples = allCandles.size();
        int lookBack = engine.getLookBack();

        // Empezamos donde tenemos datos suficientes
        int startIndex = lookBack + 1;

        // Variables de estado
        double balance = 1000.0;
        double initialBalance = balance;
        double feeMaker;
        double feeTaker;
        if (market.getSymbol().endsWith("USDT")) {
            feeMaker = 0.0002;
            feeTaker = 0.0005;
        }else {
            feeMaker = 0;
            feeTaker = 0.0004;
        }


        BackTestStats stats = new BackTestStats();
        List<ExtraDataPlot> extraStats = new ArrayList<>();

        // 2. Loop principal (Walk-Forward)
        // Usamos un loop 'i' que podemos avanzar manualmente si una operación dura varias velas
        for (int i = startIndex; i < totalSamples - 1; i++) {

            Candle currentCandle = allCandles.get(i);

            // A. Obtener predicción
            List<Candle> window = allCandles.subList(i - lookBack, i + 1);
            PredictionEngine.PredictionResultTP_SL prediction = engine.predictNextPriceDetail(window, market.getSymbol());

            // B. Consultar estrategia
            TradeSetup setup = strategy.preProcess(prediction, currentCandle, balance);

            if (setup == null) {
                // La estrategia decidió no operar
                continue;
            }

            switch (setup.direccion){
                case LONG -> stats.longs++;
                case SHORT -> stats.shorts++;
                case NEUTRAL -> stats.nothing++;
            }

            if (setup.direccion == PredictionEngine.DireccionOperation.NEUTRAL) continue;

            // C. Ejecutar Simulación de la Operación (Puede durar múltiples velas)
            SimulatedTradeResult simResult = simulateTradeExecution(
                    market,
                    allCandles,
                    i, // Empezamos a buscar en la siguiente vela
                    setup
            );

            if (simResult.reason == ExitReason.NO_DATA_ERROR) {
                continue; // Ignorar operación por falta de datos
            }

            // D. Calcular PnL Real (con fees)
            // Fee se cobra al entrar (sobre entry) y al salir (sobre exit)
            double positionSize = setup.amountUsdt * setup.leverage; // notional
            double qty = positionSize / setup.entryPrice;

            double entryFee = positionSize * feeMaker;
            double exitNotional = qty * simResult.exitPrice;
            double exitFee = exitNotional * feeTaker;

            double grossPnL = (simResult.exitPrice - setup.entryPrice) * qty;
            double netPnL = grossPnL - entryFee - exitFee;

            double pnlPercent = netPnL / setup.amountUsdt; // sobre margen

            // E. Actualizar Balance
            balance += netPnL;
            if (balance < 0) balance = 0; // Quiebra

            // F. Registrar estadísticas
            TradeResult resultObj = new TradeResult(netPnL, pnlPercent, simResult.exitPrice, simResult.reason, currentCandle.openTime(), simResult.exitTime);
            strategy.postProcess(resultObj);
            stats.addTrade(resultObj, balance);
            extraStats.add(new ExtraDataPlot((float) pnlPercent, (float) balance));

            // G. Avanzar el índice 'i'
            // Si la operación duró 5 velas, saltamos esas 5 para no abrir operaciones superpuestas
            // (A menos que tu estrategia soporte grid/hedging, aquí asumimos 1 operación a la vez)
            int candlesConsumed = simResult.candlesConsumed;
            if (candlesConsumed > 0) {
                i += (candlesConsumed - 1);
            }
        }

        stats.getExtraDataPlot().addAll(extraStats);
        return new BackTestResult(
                initialBalance, balance, balance - initialBalance, stats.getRoi(),
                stats.totalTrades, stats.wins, stats.losses, stats.maxDrawdownPercent,
                stats
        );
    }

    private static int findCandleIndexByTime(List<Candle> candles, long exitTime, int currentIndex) {
        for (int j = currentIndex; j < candles.size(); j++) {
            if (candles.get(j).openTime() >= exitTime) {
                return j;
            }
        }
        return candles.size(); // Fin de los datos
    }

    /**
     * Simula la vida de un trade a través del tiempo (velas) y trades (ticks).
     */
    private static SimulatedTradeResult simulateTradeExecution(
            Market market,
            List<Candle> allCandles,
            int startIndex,
            TradeSetup setup) {

        int currentIndex = startIndex;
        int candlesChecked = 0;

        while (currentIndex < allCandles.size() && candlesChecked < setup.maxCandles) {
            Candle candle = allCandles.get(currentIndex);
            long endTime = candle.openTime() + 60_000;

            // Obtener trades reales de este minuto
            List<Trade> trades = market.getTradesInWindow(candle.openTime(), endTime);

            if (trades.isEmpty()) {
                // ERROR DE DATOS: Si no hay trades, es arriesgado asumir OHLC.
                // Opción A: Saltar vela (asumir precio estable).
                // Opción B: Abortar trade.
                // Aquí elegimos abortar si es justo en la entrada, o continuar si ya estamos dentro.
                if (candlesChecked == 0) return new SimulatedTradeResult(setup.entryPrice, ExitReason.NO_DATA_ERROR, 0, 0);

                currentIndex++;
                candlesChecked++;
                continue;
            }

            // REPLAY TICK A TICK
            for (Trade t : trades) {
                double price = t.price();

                final SimulatedTradeResult sl = new SimulatedTradeResult(price, ExitReason.STOP_LOSS, t.time(), candlesChecked + 1);
                final SimulatedTradeResult tp = new SimulatedTradeResult(price, ExitReason.TAKE_PROFIT, t.time(), candlesChecked + 1);
                switch (setup.direccion()) {
                    case LONG -> {
                        // LONG: SL abajo, TP arriba
                        if (price <= setup.slPrice) {
                            return sl;
                        }
                        if (price >= setup.tpPrice) {
                            return tp;
                        }
                    }
                    case SHORT -> {
                        // SHORT: SL arriba, TP abajo
                        if (price >= setup.slPrice) {
                            return sl;
                        }
                        if (price <= setup.tpPrice) {
                            return tp;
                        }
                    }
                }
            }

            currentIndex++;
            candlesChecked++;
        }

        // Si salimos del loop, se acabó el tiempo (TIMEOUT) o los datos
        double closePrice = allCandles.get(Math.min(currentIndex - 1, allCandles.size() - 1)).close();
        return new SimulatedTradeResult(closePrice, ExitReason.TIMEOUT, 0, candlesChecked);
    }

    // Record interno para comunicar resultado de la simulación
    private record SimulatedTradeResult(double exitPrice, ExitReason reason, long exitTime, int candlesConsumed) {}

    // --- Clase de Estadísticas (Telemetría) ---

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
            for (TradeResult tr : trades) if (tr.reason().equals(ExitReason.TAKE_PROFIT)) i++;
            return i;
        }

        public int getStopLoss() {
            int i = 0;
            for (TradeResult tr : trades) if (tr.reason().equals(ExitReason.STOP_LOSS)) i++;
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

    public record ExtraDataPlot(float pnlPercent, float balance) {}
}