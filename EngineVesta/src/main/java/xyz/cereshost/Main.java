package xyz.cereshost;

import ai.djl.translate.TranslateException;
import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.EngineUtils;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.packet.PacketHandler;
import xyz.cereshost.strategy.DefaultStrategy;
import xyz.cereshost.trading.Trading;

import java.io.IOException;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    public static final List<String> SYMBOLS_TRAINING = List.of(/*"SOLUSDT/*/ "ETHUSDC"/* "BNBUSDT"*/);
    @NotNull
    public static final DataSource DATA_SOURCE_FOR_TRAINING_MODEL = DataSource.CSV_CHUNK;
    @NotNull
    public static final DataSource DATA_SOURCE_FOR_BACK_TEST = DataSource.LOCAL_NETWORK;

    public static final int FINAL_MONTH = 12;
    public static final int INIT_MONTH = 1;


    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, TranslateException, InterruptedException {
        instance = new Main();

        System.setProperty("org.slf4j.simpleLogger.showThreadName", "false");
        System.setProperty("org.slf4j.simpleLogger.showLogName", "false");
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.pytorch", "WARN");
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.mxnet", "WARN");
        System.setProperty("org.slf4j.simpleLogger.log.ai.djl.tensorflow", "WARN");
        new PacketHandler();
        switch (args[0]) {
            case "training" -> {
                List<String> symbols = SYMBOLS_TRAINING;// Vesta.MARKETS_NAMES;
                //checkEngines();


                VestaEngine.TrainingTestsResults result = VestaEngine.trainingModel(symbols);
                EngineUtils.ResultsEvaluate evaluateResult = result.evaluate();
                BackTestEngine.BackTestResult backtestResult = result.backtest();

                Vesta.info("--------------------------------------------------");
                String fullNameSymbol = String.join(" ", symbols);

                Vesta.info("RESULTADOS FINALES DE " + fullNameSymbol.toUpperCase(Locale.ROOT) + ":");
                Vesta.info("  MAE Promedio TP:           %.8f", evaluateResult.avgMaeTP());
                Vesta.info("  MAE Promedio SL:           %.8f", evaluateResult.avgMaeSL());
                Vesta.info("  Acierto de Tendencia:      %.2f%% %.2f%% %.2f%%", evaluateResult.hitRateSimple(), evaluateResult.hitRateAdvanced(), evaluateResult.hitRateSafe());
                int[] longHits = evaluateResult.hitRateLong();
                Vesta.info("  Real Long                  %d L %d S %d N", longHits[0], longHits[1], longHits[2]);
                int[] shortHits = evaluateResult.hitRateShort();
                Vesta.info("  Real Short                 %d L %d S %d N", shortHits[0],  shortHits[1], shortHits[2]);
                int[] NeutralHits = evaluateResult.hitRateNeutral();
                Vesta.info("  Real Neutral               %d L %d S %d N", NeutralHits[0], NeutralHits[1], NeutralHits[2]);
                ChartUtils.CandleChartUtils.showPredictionComparison("Backtest " + String.join(" ", symbols), evaluateResult.resultPrediction());
                ChartUtils.plotRatioDistribution("Ratios " + String.join(" ", symbols), evaluateResult.resultPrediction());
                showDataBackTest(backtestResult);
                // Gráfica de distribución de errores porcentuales
                List<EngineUtils.ResultPrediction> resultPrediction = evaluateResult.resultPrediction();

                resultPrediction.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::tpDiff));
                ChartUtils.plot("Desviación del SL de " + fullNameSymbol, "Resultados De la evaluación",
                        List.of(new ChartUtils.DataPlot("Diferencia", resultPrediction.stream().map(EngineUtils.ResultPrediction::tpDiff).toList()),
                                new ChartUtils.DataPlot("Predicción", resultPrediction.stream().map(EngineUtils.ResultPrediction::predSL).toList()),
                                new ChartUtils.DataPlot("Real", resultPrediction.stream().map(EngineUtils.ResultPrediction::realSL).toList())
                        ));
                resultPrediction.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::lsDiff));
                ChartUtils.plot("Desviación del TP de " + fullNameSymbol, "Resultados De la evaluación",
                        List.of(new ChartUtils.DataPlot("Diferencia", resultPrediction.stream().map(EngineUtils.ResultPrediction::lsDiff).toList()),
                                new ChartUtils.DataPlot("Predicción", resultPrediction.stream().map(EngineUtils.ResultPrediction::predTP).toList()),
                                new ChartUtils.DataPlot("Real", resultPrediction.stream().map(EngineUtils.ResultPrediction::realTP).toList())
                        ));
                resultPrediction.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::dirDiff));
                ChartUtils.plot("Desviación dela Dirección de " + fullNameSymbol, "Resultados De la evaluación",
                        List.of(new ChartUtils.DataPlot("Diferencia", resultPrediction.stream().map(EngineUtils.ResultPrediction::dirDiff).toList()),
                                new ChartUtils.DataPlot("Predicción", resultPrediction.stream().map(EngineUtils.ResultPrediction::predDir).toList()),
                                new ChartUtils.DataPlot("Real", resultPrediction.stream().map(EngineUtils.ResultPrediction::realDir).toList())
                        ));
            }
            case "predict" -> {
                String symbol = "ETHUSDC";
                PredictionEngine.makePrediction(symbol);
            }
            case "backtest" -> {
                String symbol = "ETHUSDC";

                showDataBackTest(new BackTestEngine(IOdata.loadMarkets(DATA_SOURCE_FOR_BACK_TEST, symbol), PredictionEngine.loadPredictionEngine("VestaIA")).run());
            }
            case "trading" -> {
                String symbol = "ETHUSDC";
                new TradingLoopBinance(symbol, PredictionEngine.loadPredictionEngine("VestaIA"), new DefaultStrategy()).startCandleLoop();
            }
        }
    }



    private static void showDataBackTest(BackTestEngine.BackTestResult backtestResult) {

        BackTestEngine.BackTestStats stats = backtestResult.stats();
        ChartUtils.plotRatioVsROI("BackTest Ratio/ROI (Walk-Forward)", stats.getExtraDataPlot());
        ChartUtils.plotTPSLMagnitudeVsROI("BackTest Magnitud/ROI (Walk-Forward)", stats.getExtraDataPlot());
        ChartUtils.plot("BackTest Ratio (Walk-Forward)", "Trades", List.of(
                new ChartUtils.DataPlot("Ratio", stats.getExtraDataPlot().stream().map(BackTestEngine.ExtraDataPlot::ratio).toList())
        ));
        ChartUtils.plot("BackTest Balance (Walk-Forward)", "Trades", List.of(
                new ChartUtils.DataPlot("Balance", stats.getExtraDataPlot().stream().map(BackTestEngine.ExtraDataPlot::balance).toList())
        ));
        ChartUtils.plot("BackTest ROI (Walk-Forward)", "Trades", List.of(
                new ChartUtils.DataPlot("ROI%", stats.getExtraDataPlot().stream().map(BackTestEngine.ExtraDataPlot::pnlPercent).toList())
        ));

        double winRate = stats.getTotalTrades() > 0 ? (double) stats.getWins() / stats.getTotalTrades() * 100 : 0;
        double avgHoldMinutes = stats.getTotalTrades() > 0 ? (stats.getTotalHoldTimeMillis() / 1000.0 / 60.0) / stats.getTotalTrades() : 0;

        Vesta.info("--------------------------------------------------");
        Vesta.info("💰 SIMULACIÓN DE TRADING (Capital: $%.0f)", backtestResult.initialBalance());
        Vesta.info(" Trades Totales:          %d",  stats.getTotalTrades());
        Vesta.info("  Win Rate:               %.2f%% (%d W / %d L)", winRate, stats.getWins(), stats.getLosses());
        Vesta.info("  Timeouts:               %d (Salida por tiempo) ROI %.2f%% ", stats.getTimeouts(), stats.getRoiTimeOut());
        Vesta.info("  Total TP/SL:            %d TP / %s SL", stats.getTrades(Trading.ExitReason.LONG_TAKE_PROFIT) + stats.getTrades(Trading.ExitReason.SHORT_TAKE_PROFIT), stats.getTrades(Trading.ExitReason.LONG_STOP_LOSS) + stats.getTrades(Trading.ExitReason.SHORT_STOP_LOSS));
        Vesta.info("  L TP/SL:                %d TP / %s SL", stats.getTrades(Trading.ExitReason.LONG_TAKE_PROFIT), stats.getTrades(Trading.ExitReason.LONG_STOP_LOSS));
        Vesta.info("  S TP/SL:                %d TP / %s SL", stats.getTrades(Trading.ExitReason.SHORT_TAKE_PROFIT), stats.getTrades(Trading.ExitReason.SHORT_STOP_LOSS));
        Vesta.info("  ROI TP                  %.2f%% L %.2f%% S", stats.getRoiTPAvgLong(), stats.getRoiTPAvgShort());
        Vesta.info("  Max Drawdown:           %.2f%%", stats.getCurrentDrawdown() * 100);
        Vesta.info("  DireRate:               (%d %.2f%% L, %d %.2f%% S, %d N)", stats.getLongs(), stats.getRoiLong(), stats.getShorts(), stats.getRoiShort(), stats.getNothing());
        Vesta.info("  Avg Hold Time:          %.3f min", avgHoldMinutes);
        Vesta.info("  PNL Neto:               %s$%.2f%s", backtestResult.netPnL() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.netPnL(), "\u001B[0m");
        Vesta.info("  ROI Total:              %s%.2f%%%s", backtestResult.roiPercent() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.roiPercent(), "\u001B[0m");
        Vesta.info("  Max Drawdown:           %.2f%%", backtestResult.maxDrawdown()*100);
        Vesta.info("--------------------------------------------------");

        System.gc();
    }
}