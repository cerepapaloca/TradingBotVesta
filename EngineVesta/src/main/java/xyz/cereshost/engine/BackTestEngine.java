package xyz.cereshost.engine;

import xyz.cereshost.ChartUtils;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.market.Trade;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class BackTestEngine {

    public static double checkPrecisionExit(long candleTime, double entryPrice, double tpPrice, double slPrice, boolean isLong, Market market) {
        // 1. Obtenemos los trades exactos de ese minuto
        long endTime = candleTime + 60_000;
        List<Trade> trades = market.getTradesInWindow(candleTime, endTime);

        // Si no hay trades (raro), usamos la lógica estándar de vela OHLC
        if (trades.isEmpty()) {
            return 0; // O lógica fallback
        }

        // 2. Recorremos los trades en orden cronológico (Tick Replay)
        for (Trade t : trades) {
            double currentPrice = t.price();

            if (isLong) {
                // REGLA LONG:
                // ¿Tocó Stop Loss primero?
                if (currentPrice <= slPrice) return -Math.abs(entryPrice - slPrice) / entryPrice; // Retorna % pérdida real

                // ¿Tocó Take Profit primero?
                if (currentPrice >= tpPrice) return Math.abs(tpPrice - entryPrice) / entryPrice; // Retorna % ganancia fija

            } else {
                // REGLA SHORT:
                // ¿Tocó Stop Loss primero? (Precio subió)
                if (currentPrice >= slPrice) return -Math.abs(slPrice - entryPrice) / entryPrice;

                // ¿Tocó Take Profit primero? (Precio bajó)
                if (currentPrice <= tpPrice) return Math.abs(entryPrice - tpPrice) / entryPrice;
            }
        }

        // 3. Si terminó el minuto y no tocó nada, cerramos al precio del último trade (Market Close)
        double closePrice = trades.get(trades.size() - 1).price();

        if (isLong) {
            return (closePrice - entryPrice) / entryPrice;
        } else {
            return (entryPrice - closePrice) / entryPrice;
        }
    }

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

    /**
     * Ejecuta un Backtest realista (Walk-Forward) sobre el último 15% de los datos.
     * Usa timestamps reales y verifica TP/SL tick-by-tick.
     */
    public static BackTestResult runRealisticBacktest(Market market, PredictionEngine engine) {
        Vesta.info("⚙️ Iniciando Backtest (Tick-Level) sobre datos de prueba...");

        // 1. Obtener todas las velas (con indicadores ya calculados)
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        allCandles.sort(Comparator.comparingLong(Candle::openTime));

        int totalSamples = allCandles.size();
        int lookBack = engine.getLookBack();

        // Definir zona de Test (último 15%)
        int splitIndex = (int) (totalSamples * 0.85);

        // Ajustar split si es muy pequeño para el lookback
        if (splitIndex < lookBack + 1) splitIndex = lookBack + 1;

        Vesta.info("  Total Velas: %d | Inicio Test (Index): %d | Velas a testear: %d",
                totalSamples, splitIndex, totalSamples - splitIndex);

        // Variables de Simulación
        double balance = 1000.0; // Capital inicial
        double initialBalance = balance;
        double fee = 0.0004; // 0.04% comisión
        double minThreshold = 0.0006; // Umbral mínimo de predicción para entrar (0.06%)

        int wins = 0, losses = 0, totalTrades = 0;
        double maxBalance = balance;
        double maxDrawdown = 0.0;

        List<AuxiliaryBackTestResult> auxiliaryBackTestResults = new ArrayList<>();
        List<EngineUtils.ResultPrediccion> predictionLog = new ArrayList<>();

        // 2. Bucle Walk-Forward (Avanzamos una vela cada vez)
        // Iteramos hasta size - 1 porque necesitamos la vela 'futura' para validar
        for (int i = splitIndex; i < totalSamples - 1; i++) {

            // A. Preparar Ventana: [i - lookback, i]
            // Necesitamos 'lookback + 1' velas: la actual (i) hacia atrás
            int startWindow = i - lookBack;
            if (startWindow < 0) continue; // Seguridad

            List<Candle> window = allCandles.subList(startWindow, i + 1);

            // B. Predecir (Sin ver el futuro)
            PredictionEngine.PredictionResultSimple predictionResultSimple = engine.predictNextPriceDetail(window);
            float predictedLogReturn = predictionResultSimple.predictedPrice();

            // C. Decisión de Trading
            System.out.println(predictedLogReturn + " | " + predictionResultSimple.currentPrice() + " | " + predictionResultSimple.getAbsChange());
            if (Math.abs(predictedLogReturn) < minThreshold) {
                continue; // Filtro de ruido
            }

            boolean isLong = predictedLogReturn > 0;
            double entryPrice = allCandles.get(i).close(); // Entramos al cierre de la vela actual

            // Definir TP / SL (Ratio 2:1)
            double targetAbs = Math.abs(predictedLogReturn);
            double tpPercent = targetAbs * 2.0;
            double slPercent = targetAbs * 1.0;

            double tpPrice = isLong ? entryPrice * (1 + tpPercent) : entryPrice * (1 - tpPercent);
            double slPrice = isLong ? entryPrice * (1 - slPercent) : entryPrice * (1 + slPercent);

            // D. Validación con el Futuro (Vela i+1)
            Candle futureCandle = allCandles.get(i + 1);

            // Usamos la función precisa que busca en los TRADES de ese minuto
            // Nota: checkPrecisionExit debe estar en ChartUtils o EngineUtils (donde lo hayas puesto)
            double pnlPercent = checkPrecisionExit(
                    futureCandle.openTime(),
                    entryPrice,
                    tpPrice,
                    slPrice,
                    isLong,
                    market
            );

            // E. Actualizar Balance
            double tradePnL = balance * (pnlPercent - (fee * 2)); // Restar comisiones in/out
            balance += tradePnL;

            // Estadísticas
            if (pnlPercent > 0) wins++; else losses++;
            totalTrades++;

            // Drawdown
            if (balance > maxBalance) maxBalance = balance;
            double dd = (maxBalance - balance) / maxBalance;
            if (dd > maxDrawdown) maxDrawdown = dd;

            // Guardar datos para gráficas
            auxiliaryBackTestResults.add(new AuxiliaryBackTestResult((float)pnlPercent, (float)balance));

            // Log de predicción para evaluación MAE (usamos log return de cierre a cierre para comparar precisión IA)
            float realLogReturn = (float) Math.log(futureCandle.close() / entryPrice);
            predictionLog.add(new EngineUtils.ResultPrediccion(predictedLogReturn, realLogReturn, futureCandle.openTime()));
        }

        // 3. Generar Resultados Finales
        double netPnL = balance - initialBalance;
        double roiPercent = (netPnL / initialBalance) * 100.0;

        // Recalcular métricas de IA sobre este set realista
        double totalMae = predictionLog.stream().mapToDouble(r -> Math.abs(r.pred() - r.real())).average().orElse(0);
        long hitCount = predictionLog.stream().filter(r -> Math.signum(r.pred()) == Math.signum(r.real())).count();
        double hitRate = predictionLog.isEmpty() ? 0 : (double) hitCount / predictionLog.size() * 100.0;

        Vesta.info("=== Fin Backtest ===");
        Vesta.info("Trades: %d | Wins: %d | ROI: %.2f%%", totalTrades, wins, roiPercent);

        return new BackTestResult(
                initialBalance, balance, netPnL, roiPercent,
                totalTrades, wins, losses, maxDrawdown * 100.0, auxiliaryBackTestResults
        );
    }

    public record AuxiliaryBackTestResult(float pnlPercent, float balance) {}

}
