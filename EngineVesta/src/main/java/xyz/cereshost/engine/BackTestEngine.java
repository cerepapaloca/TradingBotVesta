package xyz.cereshost.engine;

import com.google.common.collect.MultimapBuilder;
import lombok.Getter;
import xyz.cereshost.ChartUtils;
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

    public record TradeSetup(double entryPrice, double tpPrice, double slPrice, boolean isLong, double amountUsdt, int maxCandles) {}

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
        double feeRate = 0.0004; // 0.04% por trade (entrada + salida)

        BackTestStats stats = new BackTestStats();
        List<AuxiliaryBackTestResult> equityCurve = new ArrayList<>();
        List<EngineUtils.ResultPrediccion> predictionsLog = new ArrayList<>();

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

            // C. Ejecutar Simulación de la Operación (Puede durar múltiples velas)
            SimulatedTradeResult simResult = simulateTradeExecution(
                    market,
                    allCandles,
                    i + 1, // Empezamos a buscar en la siguiente vela
                    setup
            );

            if (simResult.reason == ExitReason.NO_DATA_ERROR) {
                continue; // Ignorar operación por falta de datos
            }

            // D. Calcular PnL Real (con fees)
            // Fee se cobra al entrar (sobre entry) y al salir (sobre exit)
            double entryFee = setup.amountUsdt * feeRate;
            double exitVal = (setup.amountUsdt / setup.entryPrice) * simResult.exitPrice;
            double exitFee = exitVal * feeRate;

            double grossPnL = exitVal - setup.amountUsdt;
            double netPnL = grossPnL - entryFee - exitFee;
            double pnlPercent = netPnL / setup.amountUsdt;

            // E. Actualizar Balance
            balance += netPnL;
            if (balance < 0) balance = 0; // Quiebra

            // F. Registrar estadísticas
            TradeResult resultObj = new TradeResult(netPnL, pnlPercent, simResult.exitPrice, simResult.reason, currentCandle.openTime(), simResult.exitTime);
            strategy.postProcess(resultObj);
            stats.addTrade(resultObj, balance);
            equityCurve.add(new AuxiliaryBackTestResult((float) pnlPercent, (float) balance));

            // Guardar log de predicción (para gráficos)
            predictionsLog.add(new EngineUtils.ResultPrediccion(
                    prediction.tpLogReturn(), prediction.slLogReturn(),
                    // Nota: Los valores reales TP/SL aquí son aproximados para el gráfico,
                    // ya que en multi-vela es más complejo definir el "real" de una sola vela.
                    (float)Math.abs(Math.log(simResult.exitPrice/setup.entryPrice)), 0,
                    currentCandle.openTime()
            ));

            // G. Avanzar el índice 'i'
            // Si la operación duró 5 velas, saltamos esas 5 para no abrir operaciones superpuestas
            // (A menos que tu estrategia soporte grid/hedging, aquí asumimos 1 operación a la vez)
            int candlesConsumed = simResult.candlesConsumed;
            if (candlesConsumed > 0) {
                i += (candlesConsumed - 1);
            }
        }

        // 3. Generar Reporte Final
        stats.printSummary(market.getSymbol());

        if (!predictionsLog.isEmpty()) {
            ChartUtils.CandleChartUtils.showPredictionComparison("Backtest " + market.getSymbol(), predictionsLog);
            ChartUtils.plotRatioDistribution("Ratios " + market.getSymbol(), predictionsLog);
        }

        return new BackTestResult(
                initialBalance, balance, balance - initialBalance, stats.getRoi(),
                stats.totalTrades, stats.wins, stats.losses, stats.maxDrawdownPercent,
                equityCurve
        );
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

                if (setup.isLong) {
                    // LONG: SL abajo, TP arriba
                    if (price <= setup.slPrice) {
                        return new SimulatedTradeResult(setup.slPrice, ExitReason.STOP_LOSS, t.time(), candlesChecked + 1);
                    }
                    if (price >= setup.tpPrice) {
                        return new SimulatedTradeResult(setup.tpPrice, ExitReason.TAKE_PROFIT, t.time(), candlesChecked + 1);
                    }
                } else {
                    // SHORT: SL arriba, TP abajo
                    if (price >= setup.slPrice) {
                        return new SimulatedTradeResult(setup.slPrice, ExitReason.STOP_LOSS, t.time(), candlesChecked + 1);
                    }
                    if (price <= setup.tpPrice) {
                        return new SimulatedTradeResult(setup.tpPrice, ExitReason.TAKE_PROFIT, t.time(), candlesChecked + 1);
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

    public static class BackTestStats {
        int totalTrades = 0;
        int wins = 0;
        int losses = 0;
        int timeouts = 0;
        double maxDrawdownPercent = 0.0;
        double peakBalance = 0.0;
        double currentDrawdown = 0.0;
        double totalPnL = 0.0;
        double initialBalance = 0.0;

        // Para calcular Hold Time promedio
        long totalHoldTimeMillis = 0;

        public void addTrade(TradeResult result, double currentBalance) {
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

        public void printSummary(String symbol) {
            double winRate = totalTrades > 0 ? (double) wins / totalTrades * 100 : 0;
            double avgHoldMinutes = totalTrades > 0 ? (totalHoldTimeMillis / 1000.0 / 60.0) / totalTrades : 0;

            Vesta.info("====== REPORTE BACKTEST: " + symbol + " ======");
            Vesta.info(" Trades Totales: " + totalTrades);
            Vesta.info(" Win Rate:       %.2f%% (%d W / %d L)", winRate, wins, losses);
            Vesta.info(" Timeouts:       %d (Salida por tiempo)", timeouts);
            Vesta.info(" ROI Total:      %.2f%%", getRoi());
            Vesta.info(" Max Drawdown:   %.2f%%", maxDrawdownPercent * 100);
            Vesta.info(" Avg Hold Time:  %.1f min", avgHoldMinutes);
            Vesta.info("==========================================");
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
            List<AuxiliaryBackTestResult> auxiliaryResults
    ) {}

    public record AuxiliaryBackTestResult(float pnlPercent, float balance) {}
}