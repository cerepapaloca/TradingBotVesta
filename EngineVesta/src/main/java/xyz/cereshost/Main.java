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

    public static final List<String> SYMBOLS_TRAINING = List.of(/*"SOLUSDT/, "XRPUSDT",*/ "BNBUSDT");
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
                //List.of("BTCUSDT");// Vesta.MARKETS_NAMES;
                //checkEngines();
                HashMap<List<String>, VestaEngine.TrainingTestsResults> results = new HashMap<>();
                for (String symbol : List.of("BNBUSDT")) {
                    results.put(Arrays.stream(symbol.split(",")).toList(), VestaEngine.trainingModel(Arrays.stream(symbol.split(",")).toList()));
                }

                for (Map.Entry<List<String>, VestaEngine.TrainingTestsResults> entry : results.entrySet()) {
                    VestaEngine.TrainingTestsResults result = entry.getValue();
                    EngineUtils.ResultsEvaluate evaluateResult = result.evaluate();
                    BackTestEngine.BackTestResult backtestResult = result.backtest();

                    Vesta.info("--------------------------------------------------");
                    String fullNameSymbol = String.join(" ", entry.getKey());
                    Vesta.info("RESULTADOS FINALES DE " + fullNameSymbol.toUpperCase(Locale.ROOT) + ":");
                    Vesta.info("  MAE Promedio TP:           %.8f", evaluateResult.avgMaeTP());
                    Vesta.info("  MAE Promedio SL:           %.8f", evaluateResult.avgMaeSL());
                    Vesta.info("  Acierto de Tendencia:   %.2f%%", evaluateResult.hitRate());
                    showDataBackTest(backtestResult);
                    // Gráfica de distribución de errores porcentuales
                    evaluateResult.resultPrediccions().sort(Comparator.comparingDouble(EngineUtils.ResultPrediccion::tpDiff));
                    ChartUtils.plot("Desviación del SL de " + fullNameSymbol, "Resultados De la evaluación",
                            List.of(new ChartUtils.DataPlot("Diferencia", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::tpDiff).toList()),
                                    new ChartUtils.DataPlot("Predicción", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::predSL).toList()),
                                    new ChartUtils.DataPlot("Real", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::realSL).toList())
                            ));
                    evaluateResult.resultPrediccions().sort(Comparator.comparingDouble(EngineUtils.ResultPrediccion::lsDiff));
                    ChartUtils.plot("Desviación del TP de " + fullNameSymbol, "Resultados De la evaluación",
                            List.of(new ChartUtils.DataPlot("Diferencia", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::lsDiff).toList()),
                                    new ChartUtils.DataPlot("Predicción", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::predTP).toList()),
                                    new ChartUtils.DataPlot("Real", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::realTP).toList())
                            ));
//                    EngineUtils.analyzePerformanceBySize(evaluateResult.resultPrediccions());
                    if (!backtestResult.auxiliaryResults().isEmpty()) {
                        ChartUtils.plot("BackTest Balance (Walk-Forward)", "Trades", List.of(
                                new ChartUtils.DataPlot("Balance", backtestResult.auxiliaryResults().stream().map(BackTestEngine.AuxiliaryBackTestResult::balance).toList())
                        ));
                        ChartUtils.plot("BackTest ROI (Walk-Forward)", "Trades", List.of(
                                new ChartUtils.DataPlot("ROI%", backtestResult.auxiliaryResults().stream().map(BackTestEngine.AuxiliaryBackTestResult::pnlPercent).toList())
                        ));
                    }
                }
            }
            case "predict" -> {
                String symbol = "BNBUSDT";
                PredictionEngine.makePrediction(symbol);
            }
            case "backtest" -> {
                String symbol = "BNBUSDT";
                IOdata.loadMarkets(DATA_SOURCE_FOR_BACK_TEST, symbol);
                showDataBackTest(BackTestEngine.runBacktest(Vesta.MARKETS.get(symbol), PredictionEngine.loadPredictionEngine("VestaIA")));
            }
        }
    }

    private static void showDataBackTest(BackTestEngine.BackTestResult backtestResult) {
        Vesta.info("--------------------------------------------------");
        Vesta.info("💰 SIMULACIÓN DE TRADING (Capital: $%.0f)", backtestResult.initialBalance());
        Vesta.info("  PNL Neto:               %s$%.2f%s", backtestResult.netPnL() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.netPnL(), "\u001B[0m");
        Vesta.info("  ROI Total:              %s%.2f%%%s", backtestResult.roiPercent() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.roiPercent(), "\u001B[0m");
        Vesta.info("  Trades (Win/Loss):      %d (%d/%d)", backtestResult.totalTrades(), backtestResult.winTrades(), backtestResult.lossTrades());
        Vesta.info("  Max Drawdown:           %.2f%%", backtestResult.maxDrawdown());
        Vesta.info("--------------------------------------------------");
    }
}