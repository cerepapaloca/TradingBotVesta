package xyz.cereshost.vesta.core.utils;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.annotations.XYBoxAnnotation;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.*;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.chart.renderer.xy.XYAreaRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.DefaultHighLowDataset;
import org.jfree.data.xy.OHLCDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.ia.utils.EngineUtils;
import xyz.cereshost.vesta.core.trading.backtest.BackTestEngine;
import xyz.cereshost.vesta.common.market.Candle;
import xyz.cereshost.vesta.common.market.CandleSimple;
import xyz.cereshost.vesta.core.trading.TradingManager;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

import static org.jfree.chart.axis.NumberAxis.createIntegerTickUnits;
import static xyz.cereshost.vesta.core.ia.PredictionEngine.THRESHOLD_RELATIVE;

public class ChartUtils {

    private static boolean chartsDisabledLogged = false;

    private static boolean chartsEnabled() {
        boolean enabled = !GraphicsEnvironment.isHeadless() && !Boolean.getBoolean("vesta.charts.disabled");
        if (!enabled && !chartsDisabledLogged) {
            Vesta.info("Charts deshabilitados (headless o vesta.charts.disabled=true).");
            chartsDisabledLogged = true;
        }
        return enabled;
    }


    public static XYSeriesCollection plot(
            String title,
            String xLabel,
            List<DataPlot> plots
    ) {
        XYSeriesCollection dataset = new XYSeriesCollection();
        for (DataPlot plot : plots) {
            List<Float> values = plot.getValues();
            String seriesName = plot.getYLabel();
            XYSeries series = new XYSeries(seriesName);
            for (int i = 0; i < values.size(); i++) {
                series.add(i + 1, (float) values.get(i));

            }
            dataset.addSeries(series);
        }


        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                xLabel,
                "Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        NumberAxis xAxis = getNumberAxisAndApplyStyle(plots, chart);
        xAxis.setNumberFormatOverride(new DecimalFormat("0"));
        xAxis.setAutoTickUnitSelection(true);
        xAxis.setStandardTickUnits(createIntegerTickUnits());

        if (!chartsEnabled()) {
            return dataset;
        }
        
        ChartFrame frame = new ChartFrame(title, chart);
        darkMode(chart);
        frame.pack();
        frame.setVisible(true);

        return dataset;
    }

    private static NumberAxis getNumberAxisAndApplyStyle(List<DataPlot> plots, JFreeChart chart) {
        XYPlot chartSeries = chart.getXYPlot();
        int idx = 0;
        for (DataPlot plot : plots) {
            XYItemRenderer render = chartSeries.getRenderer();
            if (plot.getColor() != null) render.setSeriesPaint(idx, plot.getColor());
            switch (plot.getStyleLine()){
                case DISCONTINUA -> render.setSeriesStroke(idx,
                        new BasicStroke(2.5f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_BEVEL, 0, new float[]{8.0f, 6.0f}, 0));
                default -> render.setSeriesStroke(idx,
                        new BasicStroke(2f));
            }
            idx++;
        }
        return (NumberAxis) chartSeries.getDomainAxis();
    }

    @Getter
    public static final class DataPlot {
        @NotNull
        private final String yLabel;
        @NotNull
        private final List<Float> values;
        private final Color color;
        @NotNull
        private final StyleLine styleLine;

        public DataPlot(
                @NotNull String yLabel,
                @NotNull List<Float> values
        ) {
            this.yLabel = yLabel;
            this.values = values;
            this.color = null;
            this.styleLine = StyleLine.NORMAL;
        }

        public DataPlot(
                @NotNull String yLabel,
                @NotNull List<Float> values,
                Color color,
                @NotNull StyleLine styleLine
        ) {
            this.yLabel = yLabel;
            this.values = values;
            this.color = color;
            this.styleLine = styleLine;
        }

        public enum StyleLine{
            NORMAL,
            DISCONTINUA
        }

    }

    public static void darkMode(JFreeChart chart){
        Plot plot = chart.getPlot();
        chart.getTitle().setPaint(Color.WHITE);
        switch (plot){
            case XYPlot xyplot -> {
                ValueAxis xAxis = xyplot.getDomainAxis();
                ValueAxis yAxis = xyplot.getRangeAxis();
                xAxis.setLabelPaint(Color.WHITE);
                xAxis.setTickLabelPaint(Color.WHITE);
                yAxis.setLabelPaint(Color.WHITE);
                yAxis.setTickLabelPaint(Color.WHITE);
                xyplot.setDomainGridlinePaint(Color.GRAY);
                xyplot.setRangeGridlinePaint(Color.GRAY);
            }
            default -> {}
        }
        plot.setBackgroundPaint(Color.BLACK);
        chart.getLegend().setBackgroundPaint(Color.BLACK);
        chart.getLegend().setItemPaint(Color.WHITE);
        chart.setBackgroundPaint(Color.BLACK);
    };



    public static void showCandleChart(String title, List<Candle> candles, String symbol) {
        if (candles == null || candles.isEmpty()) {
            Vesta.error("No hay velas para mostrar");
            return;
        }
        if (!chartsEnabled()) {
            return;
        }

        try {
            // Preparar datos para JFreeChart
            int itemCount = candles.size();
            Date[] dates = new Date[itemCount];
            double[] highs = new double[itemCount];
            double[] lows = new double[itemCount];
            double[] opens = new double[itemCount];
            double[] closes = new double[itemCount];
            double[] volumes = new double[itemCount];
            XYSeries superTrendSeries = new XYSeries("SuperTrend");

            for (int i = 0; i < itemCount; i++) {
                Candle candle = candles.get(i);
                dates[i] = new Date(candle.openTime());
                opens[i] = candle.open();
                highs[i] = candle.high();
                lows[i] = candle.low();
                closes[i] = candle.close();
                volumes[i] = candle.volumeBase();

                // Candle.superTrendSlow() almacena (lineaSuperTrend - close), por eso sumamos close para obtener el precio real de la linea.
                double superTrendPrice = candle.emaSlow();// candle.close() + candle.superTrendSlow();
                if (Double.isFinite(superTrendPrice) && superTrendPrice > 0) {
                    superTrendSeries.add(dates[i].getTime(), superTrendPrice);
                }
            }

            // Crear dataset de velas
            OHLCDataset dataset = new DefaultHighLowDataset(
                    symbol, dates, highs, lows, opens, closes, volumes
            );

            // Crear gráfico de velas
            JFreeChart chart = ChartFactory.createCandlestickChart(
                    title + " - " + symbol,
                    "Fecha",
                    "Precio",
                    dataset,
                    true
            );

            // Configurar eje de fechas
            XYPlot plot = (XYPlot) chart.getPlot();
            DateAxis axis = (DateAxis) plot.getDomainAxis();
            axis.setDateFormatOverride(new SimpleDateFormat("dd/MM HH:mm"));

            // Overlay: linea de SuperTrend
            if (!superTrendSeries.isEmpty()) {
                XYSeriesCollection superTrendDataset = new XYSeriesCollection();
                superTrendDataset.addSeries(superTrendSeries);

                XYLineAndShapeRenderer superTrendRenderer = new XYLineAndShapeRenderer(true, false);
                superTrendRenderer.setSeriesPaint(0, new Color(255, 215, 0));
                superTrendRenderer.setSeriesStroke(0, new BasicStroke(1.7f));

                plot.setDataset(1, superTrendDataset);
                plot.setRenderer(1, superTrendRenderer);
            }

            // Ajusta el zoon
            double minPrice = Arrays.stream(lows).min().orElse(0);
            double maxPrice = Arrays.stream(highs).max().orElse(0);
            if (!superTrendSeries.isEmpty()) {
                minPrice = Math.min(minPrice, superTrendSeries.getMinY());
                maxPrice = Math.max(maxPrice, superTrendSeries.getMaxY());
            }

            double padding = (maxPrice - minPrice) * 0.02; // pading

            NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
            rangeAxis.setRange(
                    minPrice - padding,
                    maxPrice + padding
            );

            // Mostrar en ventana
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(1200, 600));

            JFrame frame = new JFrame("Visualización de Velas - " + symbol);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            Vesta.info("Mostrando gráfico de " + candles.size() + " velas para " + symbol);

        } catch (Exception e) {
            Vesta.error("Error mostrando gráfico: " + e.getMessage());
            e.printStackTrace();
        }
    }

    public record PredictionPoint(long time, double tpPrice, double slPrice) {}

    public static void animateCandlePredictions(
            String title,
            List<CandleSimple> candles,
            List<BackTestEngine.InCompleteTrade> trades,
            int windowSize,
            int delayMs
    ) {
        animateCandlePredictions(title, candles, trades, windowSize, delayMs, 0, 15);
    }

    public static void animateCandlePredictions(
            String title,
            List<CandleSimple> candles,
            List<BackTestEngine.InCompleteTrade> trades,
            int windowSize,
            int delayMs,
            double zoomOutFactor,
            double horizonZoomOutFactor
    ) {
        if (candles == null || candles.isEmpty()) {
            Vesta.error("No hay velas para animar");
            return;
        }
        if (!chartsEnabled()) {
            return;
        }

        List<PredictionPoint> preds = new ArrayList<>();
        for (BackTestEngine.InCompleteTrade trade : trades) {
            preds.add(new PredictionPoint(trade.getEntryTime(), trade.getTpPrice(), trade.getSlPrice()));
        }

        List<CandleSimple> sorted = new ArrayList<>(candles);
        sorted.sort(Comparator.comparingLong(CandleSimple::openTime));

        preds.sort(Comparator.comparingLong(PredictionPoint::time));

        int safeDelay = Math.max(10, delayMs);
        double safeZoomOut = Math.max(0.0, zoomOutFactor);
        double safeHorizonZoomOut = Math.max(0.0, horizonZoomOutFactor);

        DefaultHighLowDataset dataset = buildHighLowDataset(sorted, 0, 0, title);
        JFreeChart chart = ChartFactory.createCandlestickChart(
                title,
                "Fecha",
                "Precio",
                dataset,
                true
        );

        XYPlot plot = (XYPlot) chart.getPlot();
        DateAxis axis = (DateAxis) plot.getDomainAxis();
        axis.setDateFormatOverride(new SimpleDateFormat("dd/MM HH:mm"));

        XYSeries maxSeries = new XYSeries("Max");
        XYSeries minSeries = new XYSeries("Min");
        XYSeries priceSeries = new XYSeries("Precio");
        XYSeriesCollection predDataset = new XYSeriesCollection();
        predDataset.addSeries(maxSeries);
        predDataset.addSeries(minSeries);
        predDataset.addSeries(priceSeries);

        XYLineAndShapeRenderer predRenderer = new XYLineAndShapeRenderer(true, false);
        predRenderer.setSeriesPaint(0, new Color(0, 255, 0, 200));
        predRenderer.setSeriesPaint(1, new Color(255, 0, 0, 200));
        predRenderer.setSeriesPaint(2, new Color(0, 120, 255, 220));
        predRenderer.setSeriesStroke(0, new BasicStroke(1.5f));
        predRenderer.setSeriesStroke(1, new BasicStroke(1.5f));
        predRenderer.setSeriesStroke(2, new BasicStroke(1.2f));
        plot.setDataset(1, predDataset);
        plot.setRenderer(1, predRenderer);

        darkMode(chart);

        ChartFrame frame = new ChartFrame(title, chart);
        frame.pack();
        frame.setVisible(true);

        Thread animator = new Thread(() -> {
            int predIndex = 0;
            PredictionPoint activePrediction = null;
            XYBoxAnnotation[] shadeHolder = new XYBoxAnnotation[1];
            for (int i = 0; i < sorted.size(); i++) {
                int endIndex = i;
                int horizonExtra = (int) Math.round(windowSize * safeHorizonZoomOut);
                int startIndex = Math.max(0, endIndex - windowSize + 1 - horizonExtra);
                int endIndexWindow = Math.min(sorted.size() - 1, endIndex + horizonExtra);
                long currentTime = sorted.get(endIndex).openTime();
                double currentPrice = sorted.get(endIndex).close();

                while (predIndex < preds.size() && preds.get(predIndex).time() <= currentTime) {
                    activePrediction = preds.get(predIndex);
                    predIndex++;
                }

                PredictionPoint predictionFinal = activePrediction;
                int startFinal = startIndex;
                int endFinal = endIndexWindow;
                long currentTimeFinal = currentTime;
                long windowEndTimeFinal = sorted.get(Math.min(sorted.size() - 1, i + windowSize)).openTime();
                double currentPriceFinal = currentPrice;

                SwingUtilities.invokeLater(() -> {
                    DefaultHighLowDataset windowDataset = buildHighLowDataset(sorted, startFinal, endFinal, title);
                    plot.setDataset(0, windowDataset);
                    updateAxisRange(plot, sorted, startFinal, endFinal, predictionFinal, safeZoomOut);

                    maxSeries.clear();
                    minSeries.clear();
                    priceSeries.clear();
                    if (predictionFinal != null) {
                        long startTime = sorted.get(startFinal).openTime();
                        long lineStart = Math.max(startTime, predictionFinal.time());
                        long lineEnd = Math.max(currentTimeFinal, windowEndTimeFinal);
                        double low = Math.min(predictionFinal.tpPrice(), predictionFinal.slPrice());
                        double high = Math.max(predictionFinal.tpPrice(), predictionFinal.slPrice());
                        maxSeries.add(lineStart, high);
                        maxSeries.add(lineEnd, high);
                        minSeries.add(lineStart, low);
                        minSeries.add(lineEnd, low);

                        if (shadeHolder[0] != null) {
                            plot.removeAnnotation(shadeHolder[0]);
                        }

                        XYBoxAnnotation shade = new XYBoxAnnotation(
                                lineStart,
                                low,
                                lineEnd,
                                high,
                                new BasicStroke(0.5f),
                                new Color(0, 120, 255, 40),
                                new Color(0, 120, 255, 20)
                        );
                        plot.addAnnotation(shade);
                        shadeHolder[0] = shade;
                    } else if (shadeHolder[0] != null) {
                        plot.removeAnnotation(shadeHolder[0]);
                        shadeHolder[0] = null;
                    }

                    long priceStart = sorted.get(startFinal).openTime();
                    long priceLineStart = Math.max(priceStart, currentTimeFinal);
                    long priceLineEnd = Math.max(currentTimeFinal, windowEndTimeFinal);
                    priceSeries.add(priceLineStart, currentPriceFinal);
                    priceSeries.add(priceLineEnd, currentPriceFinal);
                });

                LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(safeDelay));
            }
        });
        animator.setDaemon(true);
        animator.start();
    }

    /**
     * Muestra la distribución de datos con dos salidas (TP y SL)
     */
    public static void showTPSLDistribution(String title, float[][] y, String symbol) {
        if (!chartsEnabled()) {
            return;
        }
        try {
            // Extraer TP y SL de los datos de salida
            List<Float> maxValues = new ArrayList<>();
            List<Float> minValues = new ArrayList<>();
            int j = 0;
            for (float[] floats : y) {
                if (floats.length >= 2) {
                    float max = floats[0];
                    float min = floats[1];
                    maxValues.add(max);
                    minValues.add(-min);
                }
            }

            // Crear dataset para TP y SL
            XYSeries tpSeries = new XYSeries("Distancia");
            XYSeries slSeries = new XYSeries("Relativo");

            for (int i = 0; i < maxValues.size(); i++) {
                tpSeries.add(i, maxValues.get(i));
                slSeries.add(i, minValues.get(i));
            }

            XYSeriesCollection dataset = new XYSeriesCollection();
            dataset.addSeries(tpSeries);
            dataset.addSeries(slSeries);

            // Crear gráfico
            JFreeChart chart = ChartFactory.createScatterPlot(
                    title + " - " + symbol + " (Distribución de Distancia y Relativo)",
                    "Índice de Muestra",
                    "Valor (Log Return)",
                    dataset
            );

            // Configurar colores
            XYPlot plot = (XYPlot) chart.getPlot();
            plot.getRenderer().setSeriesPaint(0, Color.green);  // TP en verde
            plot.getRenderer().setSeriesPaint(1, Color.red);    // SL en rojo

            // Mostrar en ventana
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(1200, 600));

            darkMode(chart);
            JFrame frame = new JFrame("Distribución de Datos - " + symbol);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

//                Vesta.info("Mostrando distribución de " + longValues.size() + " muestras para " + symbol);
//                Vesta.info("TP promedio: " + longValues.stream().mapToDouble(f -> f).average().orElse(0));
//                Vesta.info("SL promedio: " + neutralValues.stream().mapToDouble(f -> f).average().orElse(0));
//                Vesta.info("Ratio TP/SL promedio: " + shortValues.stream().mapToDouble(f -> f).average().orElse(0));

        } catch (Exception e) {
            Vesta.error("Error mostrando distribución: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Muestra la comparación de predicciones vs reales para TP y SL
     */
    public static void showPredictionComparison(
            String title,
            List<EngineUtils.ResultEvaluate> results
    ) {
        if (results == null || results.isEmpty()) {
            Vesta.error("No hay resultados para mostrar");
            return;
        }
        if (!chartsEnabled()) {
            return;
        }

        try {
            // Separar datos para TP y SL
            List<Float> actualTP = new ArrayList<>();
            List<Float> predictedTP = new ArrayList<>();
            List<Float> actualSL = new ArrayList<>();
            List<Float> predictedSL = new ArrayList<>();
            List<Float> tpError = new ArrayList<>();
            List<Float> slError = new ArrayList<>();

            JTabbedPane tabbedPane = new JTabbedPane();
            results.sort(Comparator.comparingDouble(EngineUtils.ResultEvaluate::tpDiff));
            for (EngineUtils.ResultEvaluate r : results) {
                actualTP.add(r.getRealTP());
                predictedTP.add((float) r.getTpPrice());
                tpError.add(Math.abs(r.getRealTP() - (float) r.getTpPrice()));
            }
            results.sort(Comparator.comparingDouble(EngineUtils.ResultEvaluate::lsDiff));
            for (EngineUtils.ResultEvaluate r : results) {
                actualSL.add(r.getRealSL());
                predictedSL.add((float) r.getSlPrice());
                slError.add(Math.abs(r.getRealSL() - (float) r.getSlPrice()));
            }

            // Gráfico 1: TP predicho vs real
            XYSeriesCollection datasetTP = new XYSeriesCollection();
            XYSeries actualTPSeries = new XYSeries("TP Real");
            XYSeries predictedTPSeries = new XYSeries("TP Predicho");
            for (int i = 0; i < actualTP.size(); i++) {
                actualTPSeries.add(i, actualTP.get(i));
                predictedTPSeries.add(i, predictedTP.get(i));
            }
            datasetTP.addSeries(actualTPSeries);
            datasetTP.addSeries(predictedTPSeries);

            JFreeChart chartTP = ChartFactory.createXYLineChart(
                    "Take Profit (TP)",
                    "Muestra",
                    "Log Return",
                    datasetTP
            );
            ((XYPlot) chartTP.getPlot()).getRenderer().setSeriesPaint(0, Color.GREEN);
            ((XYPlot) chartTP.getPlot()).getRenderer().setSeriesPaint(1, Color.BLUE);

            tabbedPane.addTab("TP Predicho/Real", new ChartPanel(chartTP));


            // Gráfico 2: SL predicho vs real
            XYSeriesCollection datasetSL = new XYSeriesCollection();
            XYSeries actualSLSeries = new XYSeries("SL Real");
            XYSeries predictedSLSeries = new XYSeries("SL Predicho");
            for (int i = 0; i < actualSL.size(); i++) {
                actualSLSeries.add(i, actualSL.get(i));
                predictedSLSeries.add(i, predictedSL.get(i));
            }
            datasetSL.addSeries(actualSLSeries);
            datasetSL.addSeries(predictedSLSeries);

            JFreeChart chartSL = ChartFactory.createXYLineChart(
                    "Stop Loss (SL)",
                    "Muestra",
                    "Log Return",
                    datasetSL
            );
            ((XYPlot) chartSL.getPlot()).getRenderer().setSeriesPaint(0, Color.RED);
            ((XYPlot) chartSL.getPlot()).getRenderer().setSeriesPaint(1, Color.ORANGE);
            tabbedPane.addTab("SL Predicho/Real", new ChartPanel(chartSL));

            // Gráfico 3: Error de predicción
            XYSeriesCollection datasetError = new XYSeriesCollection();
            XYSeries tpErrorSeries = new XYSeries("Error TP");
            XYSeries slErrorSeries = new XYSeries("Error SL");
            for (int i = 0; i < tpError.size(); i++) {
                tpErrorSeries.add(i, tpError.get(i));
                slErrorSeries.add(i, slError.get(i));
            }
            datasetError.addSeries(tpErrorSeries);
            datasetError.addSeries(slErrorSeries);

            JFreeChart chartError = ChartFactory.createXYLineChart(
                    "Error de Predicción",
                    "Muestra",
                    "Error Absoluto",
                    datasetError
            );
            ((XYPlot) chartError.getPlot()).getRenderer().setSeriesPaint(0, Color.magenta);
            ((XYPlot) chartError.getPlot()).getRenderer().setSeriesPaint(1, Color.CYAN);
            tabbedPane.addTab("Error TP/SL", new ChartPanel(chartError));

            // Mostrar ventana
            JFrame frame = new JFrame(title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(tabbedPane);
            frame.setSize(1600, 1200);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            Vesta.info("Mostrando comparación de " + results.size() + " predicciones");

        } catch (Exception e) {
            Vesta.error("Error mostrando comparación: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Muestra la distribución de ratios TP/SL
     */
    public static void plotRatioDistribution(String title, List<EngineUtils.ResultEvaluate> results) {
        if (results == null || results.isEmpty()) return;
        if (!chartsEnabled()) {
            return;
        }

        // Calcular ratios
        List<Float> realRatios = new ArrayList<>();
        List<Float> predRatios = new ArrayList<>();

        for (EngineUtils.ResultEvaluate r : results) {
            if (r.getRealSL() != 0) {
                realRatios.add(r.getRealTP() / r.getRealSL());
            }else {
                realRatios.add(0f);
            }
            if (r.getSlPrice() != 0) {
                predRatios.add((float) (r.getTpPrice() / r.getSlPrice()));
            }else {
                predRatios.add(0f);
            }
        }

        // Crear dataset
        XYSeriesCollection dataset = new XYSeriesCollection();

        XYSeries realSeries = new XYSeries("Ratio Real TP/SL");
        XYSeries predSeries = new XYSeries("Ratio Predicho TP/SL");

        for (int i = 0; i < realRatios.size(); i++) {
            realSeries.add(i, realRatios.get(i));
        }
        for (int i = 0; i < predRatios.size(); i++) {
            predSeries.add(i, predRatios.get(i));
        }

        dataset.addSeries(realSeries);
        dataset.addSeries(predSeries);

        // Crear gráfico
        JFreeChart chart = ChartFactory.createScatterPlot(
                title,
                "Muestra",
                "Ratio TP/SL",
                dataset
        );

        XYPlot plot = chart.getXYPlot();
        plot.getRenderer().setSeriesPaint(0, Color.GREEN);
        plot.getRenderer().setSeriesPaint(1, Color.BLUE);

        // Línea horizontal en ratio 2:1 (umbral de rentabilidad común)
        plot.addRangeMarker(new ValueMarker(2.0, Color.RED, new BasicStroke(2.0f)));

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(1200, 600));

        JFrame frame = new JFrame("Distribución de Ratios");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    public static void plotRatioVsROI(String title, List<BackTestEngine.CompleteTrade> extraStats) {
        if (extraStats == null || extraStats.isEmpty()) return;
        if (!chartsEnabled()) {
            return;
        }

        try {
            // Ordenar por ratio (de menor a mayor)
            List<BackTestEngine.CompleteTrade> sortedStats = new ArrayList<>(extraStats);
            sortedStats.sort(Comparator.comparingDouble(BackTestEngine.CompleteTrade::getRatio));

            // Crear dataset para ROI vs Ratio
            XYSeriesCollection dataset = new XYSeriesCollection();
            XYSeries roiSeries = new XYSeries("ROI por Ratio");
            XYSeries cumulativeROISeries = new XYSeries("ROI Acumulado");

            double cumulativeROI = 0;
            for (BackTestEngine.CompleteTrade stat : sortedStats) {
                double roi = stat.getPnlPercent() * 100; // Convertir a porcentaje
                float ratio = stat.getRatio();

                roiSeries.add(ratio, roi);

                cumulativeROI += roi;
                cumulativeROISeries.add(ratio, cumulativeROI);
            }

            dataset.addSeries(roiSeries);
            dataset.addSeries(cumulativeROISeries);

            // Crear gráfico
            JFreeChart chart = ChartFactory.createXYLineChart(
                    title + " - ROI vs Ratio TP/SL",
                    "Ratio TP/SL (Ordenado)",
                    "ROI (%)",
                    dataset
            );

            // Configurar colores y estilo
            XYPlot plot = (XYPlot) chart.getPlot();
            plot.getRenderer().setSeriesPaint(0, Color.BLUE);
            plot.getRenderer().setSeriesPaint(1, Color.RED);

            // Añadir línea en ROI=0 para referencia
            plot.addRangeMarker(new ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));

            // Añadir línea en Ratio=1 (TP=SL) para referencia
            plot.addDomainMarker(new ValueMarker(1, Color.GREEN, new BasicStroke(1.0f)));

            // Mostrar en ventana
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(1200, 600));

            JFrame frame = new JFrame("Ratio vs ROI");
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

        } catch (Exception e) {
            Vesta.error("Error en plotRatioVsROI: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Muestra un gráfico que ordena las operaciones por magnitud del TP (de menor a mayor)
     * y visualiza el ROI correspondiente
     */
    public static void plotTPSLMagnitudeVsROI(
            String title,
            List<BackTestEngine.CompleteTrade> ExtraData
    ) {
        if ( ExtraData == null || ExtraData.isEmpty()) return;
        if (!chartsEnabled()) {
            return;
        }

        try {
            // Crear lista combinada de datos
            List<TPSLROIData> dataList = new ArrayList<>();
            for (int i = 0; i < ExtraData.size(); i++) {
                BackTestEngine.CompleteTrade stat = ExtraData.get(i);
                // Calcular magnitud del TP (en porcentaje de log return)
                float tpMagnitude = Math.abs(stat.getTpPercent()); // Convertir a puntos base
                float slMagnitude = Math.abs(stat.getSlPercent()); // Convertir a puntos base
                float roi = (float) (stat.getPnlPercent() * 100); // Convertir a porcentaje

                dataList.add(new TPSLROIData(tpMagnitude, slMagnitude, roi));
            }

            // Ordenar por magnitud del TP (menor a mayor)
            dataList.sort(Comparator.comparingDouble(TPSLROIData::tpMagnitude));

            // Crear datasets
            XYSeriesCollection datasetTP = new XYSeriesCollection();
            XYSeriesCollection datasetSL = new XYSeriesCollection();
            XYSeriesCollection datasetRatio = new XYSeriesCollection();

            XYSeries tpSeries = new XYSeries("ROI vs Magnitud TP");
            XYSeries slSeries = new XYSeries("ROI vs Magnitud SL");
            XYSeries ratioSeries = new XYSeries("ROI vs Ratio TP/SL");

            XYSeries cumTPROI = new XYSeries("ROI Acumulado TP");
            XYSeries cumSLROI = new XYSeries("ROI Acumulado SL");

            double cumTP = 0;
            double cumSL = 0;

            for (TPSLROIData data : dataList) {
                tpSeries.add(data.tpMagnitude(), data.roi());
                slSeries.add(data.slMagnitude(), data.roi());

                float ratio = data.tpMagnitude() > 0 && data.slMagnitude() > 0 ?
                        data.tpMagnitude() / data.slMagnitude() : 0;
                ratioSeries.add(ratio, data.roi());

                cumTP += data.roi();
                cumSL += data.roi();

                cumTPROI.add(data.tpMagnitude(), cumTP);
                cumSLROI.add(data.slMagnitude(), cumSL);
            }

            datasetTP.addSeries(tpSeries);
            datasetTP.addSeries(cumTPROI);

            datasetSL.addSeries(slSeries);
            datasetSL.addSeries(cumSLROI);

            datasetRatio.addSeries(ratioSeries);

            // Crear panel con múltiples gráficos
            JPanel panel = new JPanel(new GridLayout(2, 2));

            // Gráfico 1: ROI vs Magnitud TP
            JFreeChart chartTP = ChartFactory.createScatterPlot(
                    "ROI vs Magnitud TP",
                    "Magnitud TP (puntos base)",
                    "ROI (%)",
                    datasetTP
            );
            XYPlot plotTP = (XYPlot) chartTP.getPlot();
            plotTP.getRenderer().setSeriesPaint(0, Color.BLUE);
            plotTP.getRenderer().setSeriesPaint(1, Color.RED);
            plotTP.addRangeMarker(new ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
            panel.add(new ChartPanel(chartTP));

            // Gráfico 2: ROI vs Magnitud SL
            JFreeChart chartSL = ChartFactory.createScatterPlot(
                    "ROI vs Magnitud SL",
                    "Magnitud SL (puntos base)",
                    "ROI (%)",
                    datasetSL
            );
            XYPlot plotSL = (XYPlot) chartSL.getPlot();
            plotSL.getRenderer().setSeriesPaint(0, Color.GREEN);
            plotSL.getRenderer().setSeriesPaint(1, Color.ORANGE);
            plotSL.addRangeMarker(new ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
            panel.add(new ChartPanel(chartSL));

            // Gráfico 3: ROI vs Ratio TP/SL
            JFreeChart chartRatio = ChartFactory.createScatterPlot(
                    "ROI vs Ratio TP/SL",
                    "Ratio TP/SL",
                    "ROI (%)",
                    datasetRatio
            );
            XYPlot plotRatio = (XYPlot) chartRatio.getPlot();
            plotRatio.getRenderer().setSeriesPaint(0, Color.MAGENTA);
            plotRatio.addRangeMarker(new ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
            plotRatio.addDomainMarker(new ValueMarker(1, Color.GREEN, new BasicStroke(1.0f)));
            panel.add(new ChartPanel(chartRatio));

            // Gráfico 4: Heatmap de densidad
            JFreeChart heatmap = createROIDensityHeatmap(dataList);
            panel.add(new ChartPanel(heatmap));

            // Mostrar ventana
            JFrame frame = new JFrame(title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(new JScrollPane(panel));
            frame.setSize(1600, 1200);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            // Calcular estadísticas
            double avgTP = dataList.stream()
                    .mapToDouble(TPSLROIData::tpMagnitude)
                    .average()
                    .orElse(0);

            double avgSL = dataList.stream()
                    .mapToDouble(TPSLROIData::slMagnitude)
                    .average()
                    .orElse(0);

            double avgROI = dataList.stream()
                    .mapToDouble(TPSLROIData::roi)
                    .average()
                    .orElse(0);

            Vesta.info("Estadísticas Magnitud TP/SL vs ROI:");
            Vesta.info("  Magnitud TP promedio: " + String.format("%.2f", avgTP) + " pbs");
            Vesta.info("  Magnitud SL promedio: " + String.format("%.2f", avgSL) + " pbs");
            Vesta.info("  ROI promedio: " + String.format("%.2f", avgROI) + "%");
            Vesta.info("  ROI total acumulado: " + String.format("%.2f", cumTP) + "%");

        } catch (Exception e) {
            Vesta.error("Error en plotTPSLMagnitudeVsROI: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
         * Clase auxiliar para almacenar datos de TP/SL y ROI
         */
        private record TPSLROIData(float tpMagnitude, float slMagnitude, float roi) {
    }

    /**
     * Crea un heatmap de densidad para visualizar la relación entre TP, SL y ROI
     */
    private static JFreeChart createROIDensityHeatmap(List<TPSLROIData> dataList) {
        try {
            // Crear dataset para gráfico de dispersión 3D usando colores para representar el ROI
            XYSeriesCollection dataset = new XYSeriesCollection();

            // Crear series para diferentes rangos de ROI
            XYSeries highROI = new XYSeries("ROI Alto (>5%)");
            XYSeries mediumROI = new XYSeries("ROI Medio (0-5%)");
            XYSeries lowROI = new XYSeries("ROI Bajo (-5-0%)");
            XYSeries negativeROI = new XYSeries("ROI Negativo (<-5%)");

            for (TPSLROIData data : dataList) {
                if (data.roi() > 5) {
                    highROI.add(data.slMagnitude(), data.tpMagnitude());
                } else if (data.roi() > 0) {
                    mediumROI.add(data.slMagnitude(), data.tpMagnitude());
                } else if (data.roi() > -5) {
                    lowROI.add(data.slMagnitude(), data.tpMagnitude());
                } else {
                    negativeROI.add(data.slMagnitude(), data.tpMagnitude());
                }
            }

            if (highROI.getItemCount() > 0) dataset.addSeries(highROI);
            if (mediumROI.getItemCount() > 0) dataset.addSeries(mediumROI);
            if (lowROI.getItemCount() > 0) dataset.addSeries(lowROI);
            if (negativeROI.getItemCount() > 0) dataset.addSeries(negativeROI);

            // Crear gráfico de dispersión
            JFreeChart chart = ChartFactory.createScatterPlot(
                    "Heatmap: TP vs SL vs ROI",
                    "Stop Loss (puntos base)",
                    "Take Profit (puntos base)",
                    dataset
            );

            XYPlot plot = (XYPlot) chart.getPlot();

            // Configurar colores para cada serie
            plot.getRenderer().setSeriesPaint(0, Color.GREEN);     // Alto ROI
            plot.getRenderer().setSeriesPaint(1, Color.YELLOW);    // Medio ROI
            plot.getRenderer().setSeriesPaint(2, Color.ORANGE);    // Bajo ROI
            plot.getRenderer().setSeriesPaint(3, Color.RED);       // Negativo ROI

            // Añadir líneas de referencia
            plot.addDomainMarker(new ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
            plot.addRangeMarker(new ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));

            return chart;

        } catch (Exception e) {
            Vesta.error("Error creando heatmap: " + e.getMessage());
            e.printStackTrace();
            // Crear un gráfico vacío en caso de error
            return ChartFactory.createScatterPlot(
                    "Heatmap: TP vs SL vs ROI (Error)",
                    "Stop Loss",
                    "Take Profit",
                    new XYSeriesCollection()
            );
        }
    }
    /**
     * Gráfico de dispersión predicción vs realidad
     */
    public static void plotPredictionVsRealScatter(List<EngineUtils.ResultEvaluate> predictions, String title) {
        if (!chartsEnabled()) {
            return;
        }
        try {
            XYSeries longSeries = new XYSeries("LONG");
            XYSeries neutralSeries = new XYSeries("NEUTRAL");
            XYSeries shortSeries = new XYSeries("SHORT");

              for (EngineUtils.ResultEvaluate pred : predictions) {
                  float real = pred.getRealDir();
                  float predVal = pred.getPredDir();
                  switch (pred.getPredDirection()){
                      case LONG -> longSeries.add(real, predVal);
                      case SHORT -> shortSeries.add(real, predVal);
                      case NEUTRAL -> neutralSeries.add(real, predVal);
                  }
              }

            XYSeriesCollection dataset = new XYSeriesCollection();
            dataset.addSeries(longSeries);
            dataset.addSeries(neutralSeries);
            dataset.addSeries(shortSeries);

            JFreeChart chart = ChartFactory.createScatterPlot(
                    title + " - Predicción vs Realidad",
                    "Dirección Real",
                    "Dirección Predicha",
                    dataset
            );

            XYPlot plot = chart.getXYPlot();
            plot.getRenderer().setSeriesPaint(0, Color.GREEN);   // Long
            plot.getRenderer().setSeriesPaint(1, Color.DARK_GRAY);  // Neutral
            plot.getRenderer().setSeriesPaint(2, Color.RED);     // Short

            // Agregar línea de referencia (ideal)
            XYSeries idealSeries = new XYSeries("Ideal");
            idealSeries.add(-1, -1);
            idealSeries.add(1, 1);
            XYSeriesCollection idealDataset = new XYSeriesCollection(idealSeries);
            plot.setDataset(1, idealDataset);

            XYLineAndShapeRenderer idealRenderer =
                    new XYLineAndShapeRenderer();
            idealRenderer.setSeriesLinesVisible(0, true);
            idealRenderer.setSeriesShapesVisible(0, false);
            idealRenderer.setSeriesPaint(0, Color.BLUE);
            plot.setRenderer(1, idealRenderer);

            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(800, 600));

            darkMode(chart);
            JFrame frame = new JFrame("Dispersión - " + title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

        } catch (Exception e) {
            Vesta.error("Error en gráfico de dispersión: " + e.getMessage());
        }
    }

    /**
     * Distribución de errores por dirección
     */
    public static void plotErrorDistributionByDirection(List<EngineUtils.ResultEvaluate> predictions, String title) {
        if (!chartsEnabled()) {
            return;
        }
        try {
            List<Float> longErrors = new ArrayList<>();
            List<Float> neutralErrors = new ArrayList<>();
            List<Float> shortErrors = new ArrayList<>();

            for (EngineUtils.ResultEvaluate pred : predictions) {
                float error = pred.dirDiff();

                if (pred.getRealDir() > THRESHOLD_RELATIVE) {
                    longErrors.add(error);
                } else if (pred.getRealDir() < -THRESHOLD_RELATIVE) {
                    shortErrors.add(error);
                } else {
                    neutralErrors.add(error);
                }
            }

            DefaultCategoryDataset dataset = new DefaultCategoryDataset();

            // Agregar estadísticas de error
            dataset.addValue(calculateMean(longErrors), "Error Promedio", "LONG");
            dataset.addValue(calculateMean(neutralErrors), "Error Promedio", "NEUTRAL");
            dataset.addValue(calculateMean(shortErrors), "Error Promedio", "SHORT");

            dataset.addValue(calculateStdDev(longErrors), "Desviación", "LONG");
            dataset.addValue(calculateStdDev(neutralErrors), "Desviación", "NEUTRAL");
            dataset.addValue(calculateStdDev(shortErrors), "Desviación", "SHORT");

            dataset.addValue(longErrors.size(), "Muestras", "LONG");
            dataset.addValue(neutralErrors.size(), "Muestras", "NEUTRAL");
            dataset.addValue(shortErrors.size(), "Muestras", "SHORT");

            JFreeChart chart = ChartFactory.createBarChart(
                    title + " - Distribución de Errores",
                    "Categoría",
                    "Valor",
                    dataset,
                    PlotOrientation.VERTICAL,
                    true, true, false
            );

            CategoryPlot plot = chart.getCategoryPlot();
            plot.getRenderer().setSeriesPaint(0, new Color(65, 105, 225));  // Azul real
            plot.getRenderer().setSeriesPaint(1, new Color(255, 140, 0));   // Naranja
            plot.getRenderer().setSeriesPaint(2, new Color(50, 205, 50));   // Verde lima

            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(700, 500));
            darkMode(chart);
            JFrame frame = new JFrame("Distribución de Errores - " + title);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(chartPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

        } catch (Exception e) {
            Vesta.error("Error en distribución de errores: " + e.getMessage());
        }
    }

    /**
     * Calcular media de una lista
     */
    private static double calculateMean(List<Float> values) {
        if (values.isEmpty()) return 0;
        double sum = 0;
        for (float val : values) {
            sum += val;
        }
        return sum / values.size();
    }

    /**
     * Calcular desviación estándar
     */
    private static double calculateStdDev(List<Float> values) {
        if (values.size() <= 1) return 0;
        double mean = calculateMean(values);
        double sumSq = 0;
        for (float val : values) {
            double diff = val - mean;
            sumSq += diff * diff;
        }
        return Math.sqrt(sumSq / (values.size() - 1));
    }

    public static void showCandleChartWithTrades(String title, List<CandleSimple> candles, String symbol, List<BackTestEngine.CompleteTrade> trades) {
        if (candles == null || candles.isEmpty()) {
            Vesta.error("No hay velas para mostrar");
            return;
        }
        if (!chartsEnabled()) {
            return;
        }

        try {
            // Preparar datos para JFreeChart
            int itemCount = candles.size();
            Date[] dates = new Date[itemCount];
            double[] highs = new double[itemCount];
            double[] lows = new double[itemCount];
            double[] opens = new double[itemCount];
            double[] closes = new double[itemCount];
            double[] volumes = new double[itemCount];

            // Mapa para buscar velas por tiempo rápidamente
            Map<Long, Integer> timeToIndex = new HashMap<>();

            for (int i = 0; i < itemCount; i++) {
                CandleSimple candle = candles.get(i);
                dates[i] = new Date(candle.openTime());
                opens[i] = candle.open();
                highs[i] = candle.high();
                lows[i] = candle.low();
                closes[i] = candle.close();
                volumes[i] = candle.volumen().baseVolume();
                timeToIndex.put(candle.openTime(), i);
            }

            // Crear dataset de velas
            OHLCDataset dataset = new DefaultHighLowDataset(
                    symbol, dates, highs, lows, opens, closes, volumes
            );

            // Crear gráfico de velas
            JFreeChart chart = ChartFactory.createCandlestickChart(
                    title + " - " + symbol + " (Trades: " + trades.size() + ")",
                    "Fecha",
                    "Precio",
                    dataset,
                    true
            );

            // Configurar eje de fechas
            XYPlot plot = (XYPlot) chart.getPlot();
            DateAxis axis = (DateAxis) plot.getDomainAxis();
            axis.setDateFormatOverride(new SimpleDateFormat("dd/MM HH:mm"));

            // Ajustar zoom
            double minPrice = Arrays.stream(lows).min().orElse(0);
            double maxPrice = Arrays.stream(highs).max().orElse(0);

            double padding = (maxPrice - minPrice) * 0.05; // Más padding para mostrar SL/TP
            NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
            rangeAxis.setRange(minPrice - padding, maxPrice + padding);

            // Preparar renderer para agregar anotaciones
            CandlestickRenderer renderer = new CandlestickRenderer();

            XYSeriesCollection balanceDataset = buildBalanceDataset(candles, trades);
            int candleDatasetIndex = 0;
            int markerDatasetIndex = 1;
            if (balanceDataset != null) {
                candleDatasetIndex = 1;
                markerDatasetIndex = 2;

                NumberAxis balanceAxis = new NumberAxis("Balance");
                balanceAxis.setAutoRangeIncludesZero(false);
                balanceAxis.setLabelPaint(Color.WHITE);
                balanceAxis.setTickLabelPaint(Color.WHITE);
                plot.setRangeAxis(1, balanceAxis);

                XYAreaRenderer balanceRenderer = new XYAreaRenderer();
                balanceRenderer.setSeriesPaint(0, new Color(0, 140, 255, 50));
                balanceRenderer.setSeriesOutlinePaint(0, new Color(0, 140, 255, 140));
                balanceRenderer.setSeriesOutlineStroke(0, new BasicStroke(1.2f));
                balanceRenderer.setOutline(true);

                plot.setDataset(0, balanceDataset);
                plot.setRenderer(0, balanceRenderer);
                plot.mapDatasetToRangeAxis(0, 1);
            }

            plot.setDataset(candleDatasetIndex, dataset);
            plot.setRenderer(candleDatasetIndex, renderer);
            plot.mapDatasetToRangeAxis(candleDatasetIndex, 0);

            // Agregar anotaciones para cada trade
            if (trades != null && !trades.isEmpty()) {
                addTradeAnnotations(plot, trades, timeToIndex, candles, markerDatasetIndex);
            }

            // Agregar leyenda para colores
            addLegend(chart, trades);

            // Mostrar en ventana
            ChartPanel chartPanel = new ChartPanel(chart);
            chartPanel.setPreferredSize(new Dimension(1400, 700));

            // Agregar panel de información de trades
            JPanel infoPanel = createTradeInfoPanel(trades);

            JFrame frame = new JFrame("Visualización de Velas con Trades - " + symbol);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);

            // Usar BorderLayout para dividir la ventana
            JPanel mainPanel = new JPanel(new BorderLayout());
            mainPanel.add(chartPanel, BorderLayout.CENTER);
            mainPanel.add(infoPanel, BorderLayout.EAST);
            darkMode(chart);
            frame.getContentPane().add(mainPanel);
            frame.pack();
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

            Vesta.info("Mostrando gráfico de " + candles.size() + " velas con " +
                    (trades != null ? trades.size() : 0) + " trades para " + symbol);

        } catch (Exception e) {
            Vesta.error("Error mostrando gráfico: " + e.getMessage());
            e.printStackTrace();
        }
    }

    private static XYSeriesCollection buildBalanceDataset(List<CandleSimple> candles, List<BackTestEngine.CompleteTrade> trades) {
        if (candles == null || candles.isEmpty() || trades == null || trades.isEmpty()) {
            return null;
        }

        List<BackTestEngine.CompleteTrade> sorted = new ArrayList<>(trades);
        sorted.sort(Comparator.comparingLong(BackTestEngine.CompleteTrade::getExitTime));

        BackTestEngine.CompleteTrade first = sorted.getFirst();
        double initialBalance = first.getBalance() - first.getPnl();
        if (!Double.isFinite(initialBalance)) {
            return null;
        }

        XYSeries balanceSeries = new XYSeries("Balance", true, true);
        long startTime = candles.getFirst().openTime();
        balanceSeries.add(startTime, initialBalance);

        long lastTime = startTime;
        double lastBalance = initialBalance;
        for (BackTestEngine.CompleteTrade trade : sorted) {
            double balance = trade.getBalance();
            if (!Double.isFinite(balance)) {
                continue;
            }
            long time = trade.getExitTime();
            balanceSeries.add(time, balance);
            lastTime = time;
            lastBalance = balance;
        }

        long endTime = candles.getLast().openTime();
        if (endTime > lastTime && Double.isFinite(lastBalance)) {
            balanceSeries.add(endTime, lastBalance);
        }

        XYSeriesCollection dataset = new XYSeriesCollection();
        dataset.addSeries(balanceSeries);
        return dataset;
    }

    private static DefaultHighLowDataset buildHighLowDataset(
            List<CandleSimple> candles,
            int startIndex,
            int endIndex,
            String seriesKey
    ) {
        int itemCount = endIndex - startIndex + 1;
        Date[] dates = new Date[itemCount];
        double[] highs = new double[itemCount];
        double[] lows = new double[itemCount];
        double[] opens = new double[itemCount];
        double[] closes = new double[itemCount];
        double[] volumes = new double[itemCount];

        for (int i = 0; i < itemCount; i++) {
            CandleSimple candle = candles.get(startIndex + i);
            dates[i] = new Date(candle.openTime());
            opens[i] = candle.open();
            highs[i] = candle.high();
            lows[i] = candle.low();
            closes[i] = candle.close();
            volumes[i] = candle.volumen().baseVolume();
        }
        return new DefaultHighLowDataset(seriesKey, dates, highs, lows, opens, closes, volumes);
    }

    private static void updateAxisRange(
            XYPlot plot,
            List<CandleSimple> candles,
            int startIndex,
            int endIndex,
            PredictionPoint prediction,
            double zoomOutFactor
    ) {
        double min = Double.MAX_VALUE;
        double max = -Double.MAX_VALUE;
        for (int i = startIndex; i <= endIndex; i++) {
            CandleSimple candle = candles.get(i);
            min = Math.min(min, candle.low());
            max = Math.max(max, candle.high());
        }
        if (prediction != null) {
            min = Math.min(min, Math.min(prediction.tpPrice(), prediction.slPrice()));
            max = Math.max(max, Math.max(prediction.tpPrice(), prediction.slPrice()));
        }
        if (!Double.isFinite(min) || !Double.isFinite(max)) {
            return;
        }
        double range = max - min;
        double basePadding = range > 0 ? range * 0.05 : Math.max(Math.abs(max) * 0.01, 0.0001);
        double padding = basePadding + (range > 0 ? range * zoomOutFactor : Math.abs(max) * zoomOutFactor);
        NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
        rangeAxis.setRange(min - padding, max + padding);

        DateAxis domainAxis = (DateAxis) plot.getDomainAxis();
        long startTime = candles.get(startIndex).openTime();
        long endTime = candles.get(endIndex).openTime();
        domainAxis.setRange(new Date(startTime), new Date(endTime));
    }

    private static void addTradeAnnotations(XYPlot plot, List<BackTestEngine.CompleteTrade> trades,
                                            Map<Long, Integer> timeToIndex, List<CandleSimple> candles,
                                            int datasetIndex) {

        // Crear un dataset para markers
        XYSeriesCollection datasetMarkers = new XYSeriesCollection();

        // Series para diferentes tipos de markers
        XYSeries entryLongSeries = new XYSeries("Entrada LONG");
        XYSeries entryShortSeries = new XYSeries("Entrada SHORT");
        XYSeries exitProfitSeries = new XYSeries("Salida TP");
        XYSeries exitLossSeries = new XYSeries("Salida SL");
        XYSeries exitOtherSeries = new XYSeries("Otras Salidas");

        // Series para líneas de TP/SL
        XYSeries tpLinesSeries = new XYSeries("TP Line");
        XYSeries slLinesSeries = new XYSeries("SL Line");

        for (BackTestEngine.CompleteTrade trade : trades) {
            // Encontrar el índice de la vela más cercana al tiempo de entrada
            int entryIndex = findClosestCandleIndex(candles, trade.getEntryTime());
            int exitIndex = findClosestCandleIndex(candles, trade.getExitTime());

            if (entryIndex >= 0) {
                CandleSimple entryCandle = candles.get(entryIndex);
                double entryX = entryCandle.openTime();

                // Agregar marker de entrada
                if (trade.getDirection() == DireccionOperation.LONG) {
                    entryLongSeries.add(entryX, trade.getEntryPrice());
                } else if (trade.getDirection() == DireccionOperation.SHORT) {
                    entryShortSeries.add(entryX, trade.getEntryPrice());
                }

                // Agregar líneas de TP y SL
                tpLinesSeries.add(entryX, trade.getTpPrice());
                tpLinesSeries.add(exitIndex >= 0 ? candles.get(exitIndex).openTime() : entryX + 3600000, trade.getTpPrice());

                slLinesSeries.add(entryX, trade.getSlPrice());
                slLinesSeries.add(exitIndex >= 0 ? candles.get(exitIndex).openTime() : entryX + 3600000, trade.getSlPrice());
            }

            // Agregar marker de salida
            if (exitIndex >= 0) {
                CandleSimple exitCandle = candles.get(exitIndex);
                double exitX = exitCandle.openTime();

                switch (trade.getExitReason()) {
                    case LONG_TAKE_PROFIT:
                    case SHORT_TAKE_PROFIT:
                        exitProfitSeries.add(exitX, trade.getExitPrice());
                        break;
                    case LONG_STOP_LOSS:
                    case SHORT_STOP_LOSS:
                        exitLossSeries.add(exitX, trade.getExitPrice());
                        break;
                    default:
                        exitOtherSeries.add(exitX, trade.getExitPrice());
                        break;
                }
            }
        }

        // Agregar series al dataset
        if (entryLongSeries.getItemCount() > 0) datasetMarkers.addSeries(entryLongSeries);
        if (entryShortSeries.getItemCount() > 0) datasetMarkers.addSeries(entryShortSeries);
        if (exitProfitSeries.getItemCount() > 0) datasetMarkers.addSeries(exitProfitSeries);
        if (exitLossSeries.getItemCount() > 0) datasetMarkers.addSeries(exitLossSeries);
        if (exitOtherSeries.getItemCount() > 0) datasetMarkers.addSeries(exitOtherSeries);
        if (tpLinesSeries.getItemCount() > 0) datasetMarkers.addSeries(tpLinesSeries);
        if (slLinesSeries.getItemCount() > 0) datasetMarkers.addSeries(slLinesSeries);

        // Crear renderer para markers
        XYLineAndShapeRenderer markerRenderer = new XYLineAndShapeRenderer(false, true);

        // Configurar colores y formas
        int seriesIndex = 0;

        // Entrada LONG - triángulo verde hacia arriba
        if (entryLongSeries.getItemCount() > 0) {
            markerRenderer.setSeriesShape(seriesIndex,
                    new Polygon(new int[]{0, 5, 10}, new int[]{10, 0, 10}, 3));
            markerRenderer.setSeriesPaint(seriesIndex, Color.GREEN);
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(2));
            seriesIndex++;
        }

        // Entrada SHORT - triángulo rojo hacia abajo
        if (entryShortSeries.getItemCount() > 0) {
            markerRenderer.setSeriesShape(seriesIndex,
                    new Polygon(new int[]{0, 10, 5}, new int[]{0, 0, 10}, 3));
            markerRenderer.setSeriesPaint(seriesIndex, Color.RED);
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(2));
            seriesIndex++;
        }

        // Salida TP - círculo verde
        if (exitProfitSeries.getItemCount() > 0) {
            markerRenderer.setSeriesShape(seriesIndex, new Ellipse2D.Double(-5, -5, 10, 10));
            markerRenderer.setSeriesPaint(seriesIndex, new Color(0, 200, 0)); // Verde brillante
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(2));
            seriesIndex++;
        }

        // Salida SL - X roja
        if (exitLossSeries.getItemCount() > 0) {
            markerRenderer.setSeriesShape(seriesIndex,
                    new Polygon(new int[]{0, 10, 10, 0}, new int[]{0, 10, 0, 10}, 4));
            markerRenderer.setSeriesPaint(seriesIndex, new Color(200, 0, 0)); // Rojo brillante
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(2));
            seriesIndex++;
        }

        // Otras salidas - cuadrado azul
        if (exitOtherSeries.getItemCount() > 0) {
            markerRenderer.setSeriesShape(seriesIndex, new Rectangle2D.Double(-5, -5, 10, 10));
            markerRenderer.setSeriesPaint(seriesIndex, Color.BLUE);
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(2));
            seriesIndex++;
        }

        // Líneas de TP - línea verde discontinua
        if (tpLinesSeries.getItemCount() > 0) {
            markerRenderer.setSeriesLinesVisible(seriesIndex, true);
            markerRenderer.setSeriesShapesVisible(seriesIndex, false);
            markerRenderer.setSeriesPaint(seriesIndex, new Color(0, 255, 0, 200)); // Verde transparente
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(1, BasicStroke.CAP_BUTT,
                    BasicStroke.JOIN_BEVEL, 0, new float[]{5, 5}, 0));
            seriesIndex++;
        }

        // Líneas de SL - línea roja discontinua
        if (slLinesSeries.getItemCount() > 0) {
            markerRenderer.setSeriesLinesVisible(seriesIndex, true);
            markerRenderer.setSeriesShapesVisible(seriesIndex, false);
            markerRenderer.setSeriesPaint(seriesIndex, new Color(255, 0, 0, 200)); // Rojo transparente
            markerRenderer.setSeriesStroke(seriesIndex, new BasicStroke(1, BasicStroke.CAP_BUTT,
                    BasicStroke.JOIN_BEVEL, 0, new float[]{5, 5}, 0));
        }

        // Agregar el renderer al plot
        plot.setDataset(datasetIndex, datasetMarkers);
        plot.setRenderer(datasetIndex, markerRenderer);
    }

    private static int findClosestCandleIndex(List<CandleSimple> candles, long time) {
        int left = 0;
        int right = candles.size() - 1;

        while (left <= right) {
            int mid = left + (right - left) / 2;
            long midTime = candles.get(mid).openTime();

            if (midTime == time) {
                return mid;
            } else if (midTime < time) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }

        // Si no encontramos exacto, devolvemos el más cercano
        if (right < 0) return 0;
        if (left >= candles.size()) return candles.size() - 1;

        long leftTime = candles.get(left).openTime();
        long rightTime = candles.get(right).openTime();

        return Math.abs(leftTime - time) < Math.abs(rightTime - time) ? left : right;
    }

    private static void addLegend(JFreeChart chart, List<BackTestEngine.CompleteTrade> trades) {
        // Crear leyenda personalizada si hay trades
        if (trades != null && !trades.isEmpty()) {
            // Contar estadísticas
            long longWins = trades.stream()
                    .filter(t -> t.getDirection() == DireccionOperation.LONG && t.getPnl() > 0)
                    .count();
            long longLosses = trades.stream()
                    .filter(t -> t.getDirection() == DireccionOperation.LONG && t.getPnl() <= 0)
                    .count();
            long shortWins = trades.stream()
                    .filter(t -> t.getDirection() == DireccionOperation.SHORT && t.getPnl() > 0)
                    .count();
            long shortLosses = trades.stream()
                    .filter(t -> t.getDirection() == DireccionOperation.SHORT && t.getPnl() <= 0)
                    .count();

            String legendText = String.format(
                    "<html><b>Resumen de Trades:</b><br>" +
                            "Total: %d trades<br>" +
                            "LONG: %d (✓%d ✗%d)<br>" +
                            "SHORT: %d (✓%d ✗%d)<br>" +
                            "<br><b>Leyenda:</b><br>" +
                            "▲ Entrada LONG<br>" +
                            "▼ Entrada SHORT<br>" +
                            "● Salida TP<br>" +
                            "✗ Salida SL<br>" +
                            "■ Otra Salida<br>" +
                            "---- Línea TP<br>" +
                            "---- Línea SL</html>",
                    trades.size(),
                    longWins + longLosses, longWins, longLosses,
                    shortWins + shortLosses, shortWins, shortLosses
            );

            JLabel legendLabel = new JLabel(legendText);
            legendLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

            // Podrías agregar esta leyenda al frame principal si lo deseas
        }
    }

    private static JPanel createTradeInfoPanel(List<BackTestEngine.CompleteTrade> trades) {
        JPanel panel = new JPanel(new BorderLayout());
        panel.setPreferredSize(new Dimension(300, 700));
        panel.setBorder(BorderFactory.createTitledBorder("Detalles de Trades"));

        // Crear modelo de tabla
        String[] columnNames = {"#", "Dir", "Entry", "TP", "SL", "Exit", "PnL%", "Razón"};
        Object[][] data = new Object[trades.size()][8];

        DecimalFormat priceFormat = new DecimalFormat("#0.0000");
        DecimalFormat pnlFormat = new DecimalFormat("+0.00%;-0.00%");

        for (int i = 0; i < trades.size(); i++) {
            BackTestEngine.CompleteTrade trade = trades.get(i);

            String direction = trade.getDirection() == DireccionOperation.LONG ? "LONG" :
                    trade.getDirection() == DireccionOperation.SHORT ? "SHORT" : "NEUT";

            String exitReason = trade.getExitReason().name()
                    .replace("LONG_", "")
                    .replace("SHORT_", "")
                    .replace("_", " ");

            data[i][0] = i + 1;
            data[i][1] = direction;
            data[i][2] = priceFormat.format(trade.getEntryPrice());
            data[i][3] = priceFormat.format(trade.getTpPrice());
            data[i][4] = priceFormat.format(trade.getSlPrice());
            data[i][5] = priceFormat.format(trade.getExitPrice());
            data[i][6] = pnlFormat.format(trade.getPnlPercent());
            data[i][7] = exitReason;
        }

        JTable table = new JTable(data, columnNames);
        table.setAutoCreateRowSorter(true);

        // Colorear filas según PnL
        table.setDefaultRenderer(Object.class, new DefaultTableCellRenderer() {
            @Override
            public Component getTableCellRendererComponent(JTable table, Object value,
                                                           boolean isSelected, boolean hasFocus, int row, int column) {
                Component c = super.getTableCellRendererComponent(table, value,
                        isSelected, hasFocus, row, column);

                if (column == 6) { // Columna PnL%
                    String pnlText = value.toString();
                    if (pnlText.startsWith("+")) {
                        c.setForeground(new Color(0, 150, 0)); // Verde para ganancias
                    } else if (pnlText.startsWith("-")) {
                        c.setForeground(Color.RED); // Rojo para pérdidas
                    }
                } else if (column == 1) { // Columna Dirección
                    String dir = value.toString();
                    if ("LONG".equals(dir)) {
                        c.setForeground(new Color(0, 100, 0)); // Verde oscuro
                    } else if ("SHORT".equals(dir)) {
                        c.setForeground(new Color(150, 0, 0)); // Rojo oscuro
                    }
                }

                return c;
            }
        });

        JScrollPane scrollPane = new JScrollPane(table);
        panel.add(scrollPane, BorderLayout.CENTER);

        // Agregar estadísticas rápidas
        if (!trades.isEmpty()) {
            double totalPnL = trades.stream().mapToDouble(t -> t.getPnlPercent()).sum();
            double avgPnL = totalPnL / trades.size();
            long wins = trades.stream().filter(t -> t.getPnl() > 0).count();
            double winRate = (double) wins / trades.size() * 100;

            JLabel statsLabel = new JLabel(String.format(
                    "<html><b>Estadísticas:</b><br>" +
                            "Win Rate: %.1f%%<br>" +
                            "PnL Total: %.2f%%<br>" +
                            "PnL Promedio: %.2f%%</html>",
                    winRate, totalPnL * 100, avgPnL * 100
            ));
            statsLabel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));
            panel.add(statsLabel, BorderLayout.SOUTH);
        }

        return panel;
    }
}
