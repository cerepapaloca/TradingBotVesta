package xyz.cereshost;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.io.IOMarket;
import xyz.cereshost.message.DiscordNotification;
import xyz.cereshost.packet.PacketHandler;
import xyz.cereshost.strategy.GammaStrategy;
import xyz.cereshost.strategy.TestStrategy;
import xyz.cereshost.trading.BinanceApiRest;
import xyz.cereshost.trading.Trading;
import xyz.cereshost.trading.TradingLoopBinance;
import xyz.cereshost.utils.BuilderData;
import xyz.cereshost.utils.ChartUtils;
import xyz.cereshost.utils.EngineUtils;
import xyz.cereshost.utils.ModelDiagnostics;

import java.io.IOException;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    @NotNull public static final List<String> SYMBOLS_TRAINING = List.of("ETHUSDT");
    @NotNull public static final String SYMBOL = "ETHUSDT";
    @NotNull public static final DataSource DATA_SOURCE_FOR_TRAINING_MODEL = DataSource.LOCAL_ZIP;
    @NotNull public static final DataSource DATA_SOURCE_FOR_BACK_TEST = DataSource.LOCAL_ZIP;
    public static final int MAX_MONTH_TRAINING = 3;


    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, InterruptedException, ExecutionException {
        instance = new Main();
//        System.setProperty("java.awt.headless","true");
        new PacketHandler();
        switch (args[0]) {
            case "training" -> {
                List<String> symbols = SYMBOLS_TRAINING;
                //checkEngines();

                VestaEngine.TrainingTestsResults result = VestaEngine.trainingModel(symbols);
                EngineUtils.ResultsEvaluate evaluateResult = result.evaluate();
                BackTestEngine.BackTestResult backtestResult = result.backtest();

                Vesta.info("--------------------------------------------------");
                String fullNameSymbol = String.join(" ", symbols);

                Vesta.info("RESULTADOS FINALES DE " + fullNameSymbol.toUpperCase(Locale.ROOT) + ":");
                Vesta.info("  MAE Promedio TP:           %.8f", evaluateResult.avgMaeTP());
                Vesta.info("  MAE Promedio SL:           %.8f", evaluateResult.avgMaeSL());
                Vesta.info("  Acierto de Tendencia:      %.2f%%S %.2f%%A %.2f%%F %.2f%%C", evaluateResult.hitRateSimple(), evaluateResult.hitRateAdvanced(), evaluateResult.hitRateSafe(), evaluateResult.hitRateConfident(0.7f));
                int[] longHits = evaluateResult.hitRate(Trading.DireccionOperation.LONG);
                Vesta.info("  Real Long                  %d L %d S %d N", longHits[0], longHits[1], longHits[2]);
                int[] shortHits = evaluateResult.hitRate(Trading.DireccionOperation.SHORT);
                Vesta.info("  Real Short                 %d L %d S %d N", shortHits[0],  shortHits[1], shortHits[2]);
                int[] NeutralHits = evaluateResult.hitRate(Trading.DireccionOperation.NEUTRAL);
                Vesta.info("  Real Neutral               %d L %d S %d N", NeutralHits[0], NeutralHits[1], NeutralHits[2]);
                ChartUtils.showPredictionComparison("Backtest " + String.join(" ", symbols), evaluateResult.resultEvaluate());
                List<EngineUtils.ResultEvaluate> resultEvaluate = evaluateResult.resultEvaluate();

                // Mostrar dispersión de predicción vs real
                ChartUtils.plotPredictionVsRealScatter(resultEvaluate, "Predición VS Real");

                // Mostrar distribución de errores por dirección
                ChartUtils.plotErrorDistributionByDirection(resultEvaluate, "Error de distrución por distacia");
                ChartUtils.plotRatioDistribution("Ratios " + String.join(" ", symbols), evaluateResult.resultEvaluate());
                showDataBackTest(backtestResult);
                // Gráfica de distribución de errores porcentuales

//                resultPrediction.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::tpDiff));
//                ChartUtils.plot("Desviación del SL de " + fullNameSymbol, "Resultados De la evaluación",
//                        List.of(new ChartUtils.DataPlot("Diferencia", resultPrediction.stream().map(EngineUtils.ResultPrediction::tpDiff).toList()),
//                                new ChartUtils.DataPlot("Predicción", resultPrediction.stream().map(EngineUtils.ResultPrediction::predSL).toList()),
//                                new ChartUtils.DataPlot("Real", resultPrediction.stream().map(EngineUtils.ResultPrediction::realSL).toList())
//                        ));
//                resultPrediction.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::lsDiff));
//                ChartUtils.plot("Desviación del TP de " + fullNameSymbol, "Resultados De la evaluación",
//                        List.of(new ChartUtils.DataPlot("Diferencia", resultPrediction.stream().map(EngineUtils.ResultPrediction::lsDiff).toList()),
//                                new ChartUtils.DataPlot("Predicción", resultPrediction.stream().map(EngineUtils.ResultPrediction::predTP).toList()),
//                                new ChartUtils.DataPlot("Real", resultPrediction.stream().map(EngineUtils.ResultPrediction::realTP).toList())
//                        ));
//                resultPrediction.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::dirDiff));
//                ChartUtils.plot("Desviación dela Dirección de " + fullNameSymbol, "Resultados De la evaluación",
//                        List.of(new ChartUtils.DataPlot("Diferencia", resultPrediction.stream().map(EngineUtils.ResultPrediction::dirDiff).toList()),
//                                new ChartUtils.DataPlot("Predicción", resultPrediction.stream().map(EngineUtils.ResultPrediction::predDir).toList()),
//                                new ChartUtils.DataPlot("Real", resultPrediction.stream().map(EngineUtils.ResultPrediction::realDir).toList())
//                        ));
            }
//            case "backtest" -> showDataBackTest(new BackTestEngine(IOMarket.loadMarkets(DATA_SOURCE_FOR_BACK_TEST, SYMBOL).limit(3), PredictionEngine.loadPredictionEngine("VestaIA"), new AlfaStrategy()).run());
            case "backtest" -> {
                Market market = new Market("XRPUSDC");
                List<CompletableFuture<Market>> task = new ArrayList<>();
                for (int day = 600; day >= 365; day--) {
                    int finalDay = day;
                    task.add(CompletableFuture.supplyAsync(() -> {
                        try {
                            return Objects.requireNonNull(IOMarket.loadMarkets(Main.DATA_SOURCE_FOR_BACK_TEST, "XRPUSDC", finalDay), "Dia: " + finalDay);                        } catch (InterruptedException | IOException e) {
                            throw new RuntimeException(e);
                        }
                    }, VestaEngine.EXECUTOR_AUXILIAR_BUILD));
                }
                for (CompletableFuture<Market> future : task) {
                    Market m = future.get();
                    market.concat(m);
                }
                Vesta.info("🔙 Ejecutando backtest...");
                showDataBackTest(new BackTestEngine(market, null, new GammaStrategy()).run());
//                HashMap<String, Double> roiMap = new HashMap<>(); IOMarket.loadMarkets(DATA_SOURCE_FOR_BACK_TEST, "XRPUSDC", 10)
//                for (String symbol : Vesta.MARKETS_NAMES) {
//                    roiMap.put(symbol, new BackTestEngine(IOMarket.loadMarkets(DATA_SOURCE_FOR_BACK_TEST, symbol, 34), null, new GammaStrategy()).run().roiPercent());
//                }
//                Vesta.info(roiMap.toString());
            }
            case "trading" -> new TradingLoopBinance("XRPUSDC", null, new GammaStrategy(), new BinanceApiRest(true), new DiscordNotification()).startCandleLoop();
            case "extract" -> IOMarket.extractFirstBin(Path.of(IOMarket.STORAGE_DIR + "\\" + SYMBOL +"\\trades"));
            case "diagnose" -> ModelDiagnostics.run();
        }
    }



    private static void showDataBackTest(BackTestEngine.BackTestResult backtestResult) {
        BackTestEngine.BackTestStats stats = backtestResult.stats();
        ChartUtils.plotRatioVsROI("BackTest Ratio/ROI (Walk-Forward)", stats.getTradesComplete());
        ChartUtils.plotTPSLMagnitudeVsROI("BackTest Magnitud/ROI (Walk-Forward)", stats.getTradesComplete());
        ChartUtils.plot("BackTest Ratio (Walk-Forward)", "Trades", List.of(
                new ChartUtils.DataPlot("Ratio", stats.getTradesComplete().stream().map(BackTestEngine.CompleteTrade::getRatio).toList())
        ));
        ChartUtils.plot("BackTest Balance (Walk-Forward)", "Trades", List.of(
                new ChartUtils.DataPlot("Balance", stats.getTradesComplete().stream().map(BackTestEngine.CompleteTrade::getBalance).toList())
        ));
        ChartUtils.plot("BackTest ROI (Walk-Forward)", "Trades", List.of(
                new ChartUtils.DataPlot("ROI%", stats.getTradesComplete().stream().map(BackTestEngine.CompleteTrade::getPnlPercent).toList())
        ));
        ChartUtils.animateCandlePredictions(stats.getMarket().getSymbol(), stats.getMarket().getCandleSimples().stream().toList(), stats.getAllTrades(), BuilderData.DEFAULT_FUTURE_WINDOW, 200);
        ChartUtils.showCandleChartWithTrades("Trades", stats.getMarket().getCandleSimples().stream().toList(), stats.getMarket().getSymbol(), stats.getTradesComplete());
        double winRate = stats.getTotalTrades() > 0 ? (double) stats.getWins() / stats.getTotalTrades() * 100 : 0;
        double avgHoldMinutes = stats.getTotalTrades() > 0 ? (stats.getTotalHoldTimeMillis() / 1000.0 / 60.0) / stats.getTotalTrades() : 0;

        DecimalFormat decimalFormat = new DecimalFormat("###,###,###,###,##0.00");

        Vesta.info("--------------------------------------------------");
        Vesta.info("💰 SIMULACIÓN DE TRADING (Capital: $%.0f)", backtestResult.initialBalance());
        Vesta.info(" Trades Totales:          %d",  stats.getTotalTrades());
        Vesta.info("  Win Rate:               %.2f%% (%d W / %d L)", winRate, stats.getWins(), stats.getLosses());
        Vesta.info("  Timeouts:               %d (Salida por tiempo) ROI %.2f%% ", stats.getTimeouts(), stats.getRoiTimeOut());
        Vesta.info("  Total TP/SL:            %d TP / %s SL", stats.getTrades(Trading.ExitReason.LONG_TAKE_PROFIT) + stats.getTrades(Trading.ExitReason.SHORT_TAKE_PROFIT), stats.getTrades(Trading.ExitReason.LONG_STOP_LOSS) + stats.getTrades(Trading.ExitReason.SHORT_STOP_LOSS));
        Vesta.info("  L TP/SL:                %d TP / %s SL", stats.getTrades(Trading.ExitReason.LONG_TAKE_PROFIT), stats.getTrades(Trading.ExitReason.LONG_STOP_LOSS));
        Vesta.info("  S TP/SL:                %d TP / %s SL", stats.getTrades(Trading.ExitReason.SHORT_TAKE_PROFIT), stats.getTrades(Trading.ExitReason.SHORT_STOP_LOSS));
        Vesta.info("  Ratio (P/M/N)           %.3f %.3f %.3f", stats.getRatioAvg(), stats.getRatioMax(), stats.getRatioMin());
        Vesta.info("  ROI TP (Min)            %.2f%% L %.2f%% S", stats.getRoiTPMinLong(), stats.getRoiTPMinShort());
        Vesta.info("  DireRate:               (%d %.2f%% L, %d %.2f%% S, %d N)", stats.getLongs(), stats.getRoiLong(), stats.getShorts(), stats.getRoiShort(), stats.getNothing());
        Vesta.info("  Avg Hold Time:          %.3f min", avgHoldMinutes);
        Vesta.info("  PNL Neto:               %s$%s%s", backtestResult.netPnL() >= 0 ? "\u001B[32m" : "\u001B[31m", decimalFormat.format(backtestResult.netPnL()), "\u001B[0m");
        Vesta.info("  ROI Total:              %s%s%%%s", backtestResult.roiPercent() >= 0 ? "\u001B[32m" : "\u001B[31m", decimalFormat.format(backtestResult.roiPercent()), "\u001B[0m");
        Vesta.info("  Max Drawdown:           %.2f%%", backtestResult.maxDrawdown()*100);
        Vesta.info("--------------------------------------------------");

        System.gc();
    }
}
