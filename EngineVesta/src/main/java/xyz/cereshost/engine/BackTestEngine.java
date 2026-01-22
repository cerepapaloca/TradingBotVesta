package xyz.cereshost.engine;

import lombok.Getter;
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

    /**
     * Verifica si se alcanzó TP o SL durante un minuto usando trades reales
     * Ahora usa TP y SL individuales en lugar de un ratio fijo
     */
    public static double checkPrecisionExit(double entryPrice,
                                            double tpPrice, double slPrice,
                                            boolean isLong, List<Trade> trades) {
        // 1. Obtenemos los trades exactos de ese minuto


        // Si no hay trades, usar lógica de vela OHLC

        if (trades.isEmpty()) {
            return 0.0;
            // Fallback: usar high/low de la vela
//            Candle nextCandle = getCandleAtTime(market, candleTime);
//            if (nextCandle == null) {
//                return 0;
//            }
//            if (isLong) {
//                // Verificar si se alcanzó SL (low <= slPrice)
//                if (nextCandle.low() <= slPrice) {
//                    return -Math.abs(entryPrice - slPrice) / entryPrice;
//                }
//                // Verificar si se alcanzó TP (high >= tpPrice)
//                if (nextCandle.high() >= tpPrice) {
//                    return Math.abs(tpPrice - entryPrice) / entryPrice;
//                }
//                // Cierre de mercado
//                return (nextCandle.close() - entryPrice) / entryPrice;
//            } else {
//                // SHORT: SL es por encima, TP es por debajo
//                if (nextCandle.high() >= slPrice) {
//                    return -Math.abs(slPrice - entryPrice) / entryPrice;
//                }
//                if (nextCandle.low() <= tpPrice) {
//                    return Math.abs(entryPrice - tpPrice) / entryPrice;
//                }
//                return (entryPrice - nextCandle.close()) / entryPrice;
//            }
        }

        // 2. Recorrer trades en orden cronológico (Tick Replay)
        for (Trade t : trades) {
            double currentPrice = t.price();

            if (isLong) {
                // REGLA LONG:
                if (currentPrice <= slPrice) {
                    return -Math.abs(entryPrice - slPrice) / entryPrice;
                }
                if (currentPrice >= tpPrice) {
                    return Math.abs(tpPrice - entryPrice) / entryPrice;
                }
            } else {
                // REGLA SHORT:
                if (currentPrice >= slPrice) {
                    return -Math.abs(slPrice - entryPrice) / entryPrice;
                }
                if (currentPrice <= tpPrice) {
                    return Math.abs(entryPrice - tpPrice) / entryPrice;
                }
            }
        }

        // 3. Si terminó el minuto y no tocó nada, cerrar al último trade
        double closePrice = trades.get(trades.size() - 1).price();
        if (isLong) {
            return (closePrice - entryPrice) / entryPrice;
        } else {
            return (entryPrice - closePrice) / entryPrice;
        }
    }

    private static Candle getCandleAtTime(Market market, long time) {
        List<Candle> candles = BuilderData.to1mCandles(market);
        for (Candle candle : candles) {
            if (candle.openTime() == time) {
                return candle;
            }
        }
        return null;
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
     * Ejecuta un Backtest realista (Walk-Forward) usando TP y SL predichos por el modelo
     */
    public static BackTestResult runBacktest(Market market, PredictionEngine engine) {
        Vesta.info("⚙️ Iniciando Backtest (Tick-Level) con TP/SL predichos...");

        // 1. Obtener todas las velas
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        allCandles.sort(Comparator.comparingLong(Candle::openTime));

        int totalSamples = allCandles.size();
        int lookBack = engine.getLookBack();

        // Definir zona de Test (último 15%)
        int splitIndex = (int) (totalSamples * 0);
        if (splitIndex < lookBack + 2) splitIndex = lookBack + 2;

        Vesta.info("  Total Velas: %d | Inicio Test (Index): %d | Velas a testear: %d",
                totalSamples, splitIndex, totalSamples - splitIndex);

        // Variables de Simulación
        double balance = 1000.0;
        double initialBalance = balance;
        double fee = 0.0004; // 0.04% comisión
        double minThreshold = 0.000000; // Umbral mínimo para operar

        int wins = 0, losses = 0, totalTrades = 0;
        double maxBalance = balance;
        double maxDrawdown = 0.0;

        List<AuxiliaryBackTestResult> auxiliaryBackTestResults = new ArrayList<>();
        List<EngineUtils.ResultPrediccion> predictionLog = new ArrayList<>();

        // 2. Bucle Walk-Forward
        for (int i = splitIndex; i < totalSamples - 1; i++) {
            // A. Preparar Ventana: [i - lookback, i]
            int startWindow = i - lookBack;
            if (startWindow < 0) continue;

            List<Candle> window = allCandles.subList(startWindow, i + 1);

            // B. Predecir TP y SL
            PredictionEngine.PredictionResultTP_SL prediction = engine.predictNextPriceDetail(window, market.getSymbol());

            float tpLogReturn = prediction.tpLogReturn();
            float slLogReturn = prediction.slLogReturn();
            String direction = prediction.direction();
            boolean isLong = "LONG".equals(direction);

            // Filtrar predicciones muy pequeñas
            if (tpLogReturn < minThreshold || slLogReturn < minThreshold) {
                continue;
            }
            Candle futureCandle = allCandles.get(i + 1);
            long endTime = futureCandle.openTime() + 60_000;
            market.buildTradeCache();
            List<Trade> trades = market.getTradesInWindow(futureCandle.openTime(), endTime);
            if (trades.isEmpty()) {
                continue;
            }

            // C. Configurar entrada y salidas
            double entryPrice = allCandles.get(i).close();

            // Calcular precios de TP y SL según dirección

            double tpPrice, slPrice;
            if (isLong) {
                tpPrice = entryPrice * Math.exp(tpLogReturn);
                slPrice = entryPrice * Math.exp(-slLogReturn);
            } else {
                tpPrice = entryPrice * Math.exp(-tpLogReturn);
                slPrice = entryPrice * Math.exp(slLogReturn);
            }

            // Verificar TP/SL usando trades reales
            double pnlPercent = checkPrecisionExit(
                    entryPrice,
                    tpPrice,
                    slPrice,
                    isLong,
                    trades
            );

            // E. Actualizar Balance
            double tradePnL = balance * (pnlPercent - (fee * 2));
            balance += tradePnL;

            // Estadísticas
            if (pnlPercent > 0) {
                wins++;
            } else if (pnlPercent < 0) {
                losses++;
            }
            totalTrades++;

            // Drawdown
            if (balance > maxBalance) {
                maxBalance = balance;
            }
            double dd = (maxBalance - balance) / maxBalance;
            if (dd > maxDrawdown) {
                maxDrawdown = dd;
            }

            // Guardar datos para gráficas
            auxiliaryBackTestResults.add(new AuxiliaryBackTestResult((float) pnlPercent, (float) balance));

            // Calcular TP y SL reales para evaluación
            float realTP, realSL;
            if (isLong) {
                realTP = (float) Math.log(futureCandle.high() / entryPrice);
                realSL = (float) -Math.log(futureCandle.low() / entryPrice);
            } else {
                realTP = (float) -Math.log(futureCandle.low() / entryPrice);
                realSL = (float) Math.log(futureCandle.high() / entryPrice);
            }

            // Guardar predicciones para evaluación
            predictionLog.add(new EngineUtils.ResultPrediccion(
                    tpLogReturn, slLogReturn,
                    Math.max(0, realTP), Math.max(0, realSL),
                    futureCandle.openTime()
            ));

            // Log detallado (opcional, para debugging)
            if (totalTrades <= 10) {
                Vesta.info("Trade %d: Dir=%s, TP=%.4f%%, SL=%.4f%%, PnL=%.4f%%, Balance=%.2f",
                        totalTrades, direction,
                        (Math.exp(tpLogReturn) - 1) * 100,
                        (Math.exp(slLogReturn) - 1) * 100,
                        pnlPercent * 100, balance);
            }
        }

        // 3. Generar Resultados Finales
        double netPnL = balance - initialBalance;
        double roiPercent = (netPnL / initialBalance) * 100.0;

        // Mostrar gráficas
        if (!predictionLog.isEmpty()) {
            ChartUtils.CandleChartUtils.showPredictionComparison(
                    "Backtest - " + market.getSymbol(),
                    predictionLog
            );
            ChartUtils.plotPrecisionByMagnitude(
                    "Precisión por Magnitud - " + market.getSymbol(),
                    predictionLog
            );
            ChartUtils.plotRatioDistribution(
                    "Distribución de Ratios - " + market.getSymbol(),
                    predictionLog
            );
        }

        return new BackTestResult(
                initialBalance, balance, netPnL, roiPercent,
                totalTrades, wins, losses, maxDrawdown * 100.0,
                auxiliaryBackTestResults
        );
    }

    /**
     * Ejecuta un Backtest simplificado para múltiples símbolos
     */
    public static List<BackTestResult> runMultiSymbolBacktest(
            List<Market> markets,
            PredictionEngine engine,
            double initialBalancePerSymbol
    ) {
        List<BackTestResult> results = new ArrayList<>();

        Vesta.info("Iniciando Backtest Multi-Símbolo...");
        Vesta.info("Número de símbolos: %d", markets.size());

        for (Market market : markets) {
            try {
                Vesta.info("\n=== Procesando %s ===", market.getSymbol());

                // Crear un engine específico para este símbolo (pero usar mismo modelo)
                PredictionEngine symbolEngine = new PredictionEngine(
                        engine.getXNormalizer(),
                        engine.getYNormalizer(),
                        engine.getModel(),
                        engine.getLookBack(),
                        engine.getFeatures()
                );

                // Ejecutar backtest
                BackTestResult result = runBacktest(market, symbolEngine);
                results.add(result);

            } catch (Exception e) {
                Vesta.error("Error en backtest para %s: %s",  market.getSymbol(), e.getMessage());
            }
        }

        // Calcular estadísticas globales
        if (!results.isEmpty()) {
            calculateGlobalStatistics(results);
        }

        return results;
    }

    private static void calculateGlobalStatistics(List<BackTestResult> results) {
        double totalInitial = 0;
        double totalFinal = 0;
        int totalTrades = 0;
        int totalWins = 0;
        int totalLosses = 0;
        double avgRoi = 0;

        for (BackTestResult result : results) {
            totalInitial += result.initialBalance();
            totalFinal += result.finalBalance();
            totalTrades += result.totalTrades();
            totalWins += result.winTrades();
            totalLosses += result.lossTrades();
            avgRoi += result.roiPercent();
        }

        avgRoi /= results.size();

        Vesta.info("\n=== ESTADÍSTICAS GLOBALES ===");
        Vesta.info("Símbolos procesados: %d", results.size());
        Vesta.info("Capital total inicial: $%.2f", totalInitial);
        Vesta.info("Capital total final: $%.2f", totalFinal);
        Vesta.info("Trades totales: %d", totalTrades);
        Vesta.info("Trades ganadores: %d (%.1f%%)", totalWins,
                totalTrades > 0 ? ((double) totalWins / totalTrades) * 100.0 : 0);
        Vesta.info("ROI promedio: %.2f%%", avgRoi);
    }

    public record AuxiliaryBackTestResult(float pnlPercent, float balance) {}

    @Getter
    public static class PerformanceBucket {
        private double minTP;
        private double maxTP;
        private String name;
        private List<EngineUtils.ResultPrediccion> predictions;
        private int count;
        private int profitable;
        private double avgRatio;

        PerformanceBucket(double minTP, double maxTP, String name) {
            this.minTP = minTP;
            this.maxTP = maxTP;
            this.name = name;
            this.predictions = new ArrayList<>();
            this.count = 0;
            this.profitable = 0;
            this.avgRatio = 0;
        }

        void analyze(List<EngineUtils.ResultPrediccion> allPredictions) {
            for (EngineUtils.ResultPrediccion p : allPredictions) {
                double tp = p.predTP();
                if (tp >= minTP && tp < maxTP) {
                    predictions.add(p);
                    count++;

                    // Verificar si el trade habría sido rentable
                    if (p.realTP() > p.realSL()) {
                        profitable++;
                    }

                    // Calcular ratio
                    if (p.predSL() > 0) {
                        avgRatio += p.predTP() / p.predSL();
                    }
                }
            }

            if (count > 0) {
                avgRatio /= count;
            }
        }

        void printResults() {
            double winRate = count > 0 ? (double) profitable / count * 100.0 : 0;
            Vesta.info("Bucket %s: %d trades, Win Rate: %.1f%%, Ratio avg: %.2f:1",
                    name, count, winRate, avgRatio);
        }
    }
}