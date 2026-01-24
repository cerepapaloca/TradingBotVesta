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

import java.io.IOException;
import java.util.*;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    public static final List<String> SYMBOLS_TRAINING = List.of(/*"SOLUSDT/*/ "SOLUSDT"/* "BNBUSDT"*/);
    @NotNull
    public static final DataSource DATA_SOURCE_FOR_TRAINING_MODEL = DataSource.CSV;
    @NotNull
    public static final DataSource DATA_SOURCE_FOR_BACK_TEST = DataSource.LOCAL_NETWORK;


    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, TranslateException, InterruptedException {
        instance = new Main();
        String datasetLimit = System.getenv("DATASET_LIMIT");
        if (datasetLimit != null) {
            System.setProperty("DATASET_LIMIT", datasetLimit);
        }

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
                Vesta.info("  Acierto de Tendencia:   %.2f%%", evaluateResult.hitRate());
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
                String symbol = "SOLUSDT";
                PredictionEngine.makePrediction(symbol);
            }
            case "backtest" -> {
                String symbol = "SOLUSDT";
                IOdata.loadMarkets(DATA_SOURCE_FOR_BACK_TEST, symbol);
                showDataBackTest(new BackTestEngine(Vesta.MARKETS.get(symbol), PredictionEngine.loadPredictionEngine("VestaIA")).run());
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
        Vesta.info("  TP/SL:                  %d TP / %s SL", stats.getTakeProfit(), stats.getStopLoss());
        Vesta.info("  Max Drawdown:           %.2f%%", stats.getCurrentDrawdown() * 100);
        Vesta.info("  DireRate:               (%d %.2f%% L, %d %.2f%% S, %d N)", stats.getLongs(), stats.getRoiLong(), stats.getShorts(), stats.getRoiShort(), stats.getNothing());
        Vesta.info("  Avg Hold Time:          %.3f min", avgHoldMinutes);
        Vesta.info("  PNL Neto:               %s$%.2f%s", backtestResult.netPnL() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.netPnL(), "\u001B[0m");
        Vesta.info("  ROI Total:              %s%.2f%%%s", backtestResult.roiPercent() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.roiPercent(), "\u001B[0m");
        Vesta.info("  Max Drawdown:           %.2f%%", backtestResult.maxDrawdown()*100);
        Vesta.info("--------------------------------------------------");
    }
}