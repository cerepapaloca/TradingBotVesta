package xyz.cereshost;

import ai.djl.translate.TranslateException;
import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.EngineUtils;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.packet.PacketHandler;

import java.io.IOException;
import java.util.*;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    public static final List<String> SYMBOLS_TRAINING = List.of(/*"SOLUSDT/, "XRPUSDT",*/ "BNBUSDT");
    @NotNull
    public static final TypeData TYPE_DATA = TypeData.ALL;

    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, TranslateException, InterruptedException {
        instance = new Main();
        new PacketHandler();
        if (args.length > 0 && args[0].equals("tr")) {
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
                Vesta.info("  MAE Promedio:           %.8f", evaluateResult.avgMae());
                Vesta.info("  Acierto de Tendencia:   %.2f%%", evaluateResult.hitRate());
                Vesta.info("--------------------------------------------------");
                Vesta.info("💰 SIMULACIÓN DE TRADING (Capital: $%.0f)", backtestResult.initialBalance());
                Vesta.info("  PNL Neto:               %s$%.2f%s", backtestResult.netPnL() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.netPnL(), "\u001B[0m");
                Vesta.info("  ROI Total:              %s%.2f%%%s", backtestResult.roiPercent() >= 0 ? "\u001B[32m" : "\u001B[31m", backtestResult.roiPercent(), "\u001B[0m");
                Vesta.info("  Trades (Win/Loss):      %d (%d/%d)", backtestResult.totalTrades(), backtestResult.winTrades(), backtestResult.lossTrades());
                Vesta.info("  Max Drawdown:           %.2f%%", backtestResult.maxDrawdown());
                Vesta.info("--------------------------------------------------");
                ChartUtils.plotPrecisionByMagnitude("Precision en la magnitud" + fullNameSymbol , evaluateResult.resultPrediccions());
                // Gráfica de distribución de errores porcentuales
                ChartUtils.plot("Desviación del precio de " + fullNameSymbol, "Resultados De la evaluación",
                        List.of(new ChartUtils.DataPlot("Diferencia", evaluateResult.resultPrediccions().stream().map(r -> r.pred() - r.real()).toList()),
                                new ChartUtils.DataPlot("Predicción", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::pred).toList()),
                                new ChartUtils.DataPlot("Real", evaluateResult.resultPrediccions().stream().map(EngineUtils.ResultPrediccion::real).toList())
                        ));
                if (!backtestResult.auxiliaryResults().isEmpty()) {
                    ChartUtils.plot("BackTest Balance (Walk-Forward)", "Trades", List.of(
                            new ChartUtils.DataPlot("Balance", backtestResult.auxiliaryResults().stream().map(BackTestEngine.AuxiliaryBackTestResult::balance).toList())
                    ));
                    ChartUtils.plot("BackTest ROI (Walk-Forward)", "Trades", List.of(
                            new ChartUtils.DataPlot("ROI%", backtestResult.auxiliaryResults().stream().map(BackTestEngine.AuxiliaryBackTestResult::pnlPercent).toList())
                    ));
                }
            }

        }else {
            String symbol = "SOLUSDT";
            PredictionEngine.makePrediction(symbol);
        }
    }
}