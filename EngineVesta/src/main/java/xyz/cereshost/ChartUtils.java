package xyz.cereshost;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartFrame;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.ValueAxis;
import org.jfree.chart.plot.*;
import org.jfree.chart.renderer.xy.CandlestickRenderer;
import org.jfree.chart.renderer.xy.XYItemRenderer;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.DefaultHighLowDataset;
import org.jfree.data.xy.OHLCDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.EngineUtils;
import xyz.cereshost.trading.Trading;

import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import java.awt.*;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Rectangle2D;
import java.text.DecimalFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

import static org.jfree.chart.axis.NumberAxis.createIntegerTickUnits;
import static xyz.cereshost.engine.PredictionEngine.THRESHOLD_RELATIVE;

public class ChartUtils {


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

        try {
            // Preparar datos para JFreeChart
            int itemCount = candles.size();
            Date[] dates = new Date[itemCount];
            double[] highs = new double[itemCount];
            double[] lows = new double[itemCount];
            double[] opens = new double[itemCount];
            double[] closes = new double[itemCount];
            double[] volumes = new double[itemCount];

            for (int i = 0; i < itemCount; i++) {
                Candle candle = candles.get(i);
                dates[i] = new Date(candle.openTime());
                opens[i] = candle.open();
                highs[i] = candle.high();
                lows[i] = candle.low();
                closes[i] = candle.close();
                volumes[i] = candle.volumeBase();
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

            // Ajusta el zoon
            double minPrice = Arrays.stream(lows).min().orElse(0);
            double maxPrice = Arrays.stream(highs).max().orElse(0);

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

    /**
     * Muestra la distribución de datos con dos salidas (TP y SL)
     */
    public static void showTPSLDistribution(String title, float[][] y, String symbol) {
        try {
            // Extraer TP y SL de los datos de salida
            List<Float> tpValues = new ArrayList<>();
            List<Float> slValues = new ArrayList<>();
            int j = 0;
            for (float[] floats : y) {
                if (floats.length >= 2) {
                    float tp = floats[0];
                    float sl = floats[1];
                    tpValues.add(tp);
                    slValues.add(-sl);
                }
            }

            // Crear dataset para TP y SL
            XYSeries tpSeries = new XYSeries("TP (Take Profit)");
            XYSeries slSeries = new XYSeries("SL (Stop Loss)");

            for (int i = 0; i < tpValues.size(); i++) {
                tpSeries.add(i, tpValues.get(i));
                slSeries.add(i, slValues.get(i));
            }

            XYSeriesCollection dataset = new XYSeriesCollection();
            dataset.addSeries(tpSeries);
            dataset.addSeries(slSeries);

            // Crear gráfico
            JFreeChart chart = ChartFactory.createScatterPlot(
                    title + " - " + symbol + " (Distribución de TP y SL)",
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

    public static void showDirectionDistribution(String title, float[][] y, String symbol) {
        try {
            // Contar ocurrencias de cada dirección
            int longCount = 0;
            int neutralCount = 0;
            int shortCount = 0;

            // TP y SL promedios por dirección
            double longTpSum = 0;
            double longSlSum = 0;
            double neutralTpSum = 0;
            double neutralSlSum = 0;
            double shortTpSum = 0;
            double shortSlSum = 0;

            for (float[] row : y) {
                if (row.length >= 5) {
                    // Las posiciones: 0=TP, 1=SL, 2=Long, 3=Neutral, 4=Short
                    float tp = row[0];
                    float sl = row[1];
                    float probLong = row[2];
                    float probNeutral = row[3];
                    float probShort = row[4];

                    // Determinar dirección predominante (one-hot encoding)
                    if (probLong >= probNeutral && probLong >= probShort) {
                        longCount++;
                        longTpSum += tp;
                        longSlSum += sl;
                    } else if (probNeutral >= probLong && probNeutral >= probShort) {
                        neutralCount++;
                        neutralTpSum += tp;
                        neutralSlSum += sl;
                    } else if (probShort >= probLong && probShort >= probNeutral) {
                        shortCount++;
                        shortTpSum += tp;
                        shortSlSum += sl;
                    }
                }
            }

            // Calcular promedios
            double longTpAvg = longCount > 0 ? longTpSum / longCount : 0;
            double longSlAvg = longCount > 0 ? longSlSum / longCount : 0;
            double neutralTpAvg = neutralCount > 0 ? neutralTpSum / neutralCount : 0;
            double neutralSlAvg = neutralCount > 0 ? neutralSlSum / neutralCount : 0;
            double shortTpAvg = shortCount > 0 ? shortTpSum / shortCount : 0;
            double shortSlAvg = shortCount > 0 ? shortSlSum / shortCount : 0;

            // Crear dataset para el gráfico de barras
            DefaultCategoryDataset countDataset = new DefaultCategoryDataset();
            countDataset.addValue(longCount, "Cantidad", "Long");
            countDataset.addValue(neutralCount, "Cantidad", "Neutral");
            countDataset.addValue(shortCount, "Cantidad", "Short");

            // Crear dataset para TP/SL por dirección
            DefaultCategoryDataset tpSlDataset = new DefaultCategoryDataset();
            tpSlDataset.addValue(longTpAvg * 10000, "TP (×10⁴)", "Long");
            tpSlDataset.addValue(longSlAvg * 10000, "SL (×10⁴)", "Long");
            tpSlDataset.addValue(neutralTpAvg * 10000, "TP (×10⁴)", "Neutral");
            tpSlDataset.addValue(neutralSlAvg * 10000, "SL (×10⁴)", "Neutral");
            tpSlDataset.addValue(shortTpAvg * 10000, "TP (×10⁴)", "Short");
            tpSlDataset.addValue(shortSlAvg * 10000, "SL (×10⁴)", "Short");

            // Crear gráfico de barras para cantidad
            JFreeChart countChart = ChartFactory.createBarChart(
                    title + " - Distribución de Direcciones - " + symbol,
                    "Dirección",
                    "Cantidad de Muestras",
                    countDataset,
                    PlotOrientation.VERTICAL,
                    true,  // Incluir leyenda
                    true,  // Tooltips
                    false  // URLs
            );

            // Configurar colores para cantidad
            CategoryPlot countPlot = countChart.getCategoryPlot();
            countPlot.getRenderer().setSeriesPaint(0, new Color(65, 105, 225)); // Azul Royal
            countPlot.setRangeGridlinePaint(Color.GRAY);

            // Crear gráfico de barras para TP/SL
            JFreeChart tpSlChart = ChartFactory.createBarChart(
                    title + " - TP/SL Promedio por Dirección - " + symbol,
                    "Dirección",
                    "Valor Promedio (Log Return ×10⁴)",
                    tpSlDataset,
                    PlotOrientation.VERTICAL,
                    true,  // Incluir leyenda
                    true,  // Tooltips
                    false  // URLs
            );

            // Configurar colores para TP/SL
            CategoryPlot tpSlPlot = tpSlChart.getCategoryPlot();
            tpSlPlot.getRenderer().setSeriesPaint(0, Color.green);
            tpSlPlot.getRenderer().setSeriesPaint(1, Color.red);

            // Crear panel con pestañas
            JTabbedPane tabbedPane = new JTabbedPane();
            tabbedPane.addTab("Distribución", new ChartPanel(countChart));
            tabbedPane.addTab("TP/SL por Dirección", new ChartPanel(tpSlChart));

            // Mostrar estadísticas en consola
//                int totalSamples = longCount + neutralCount + shortCount;
//                Vesta.info("\n📊 Estadísticas de Dirección para " + symbol + ":");
//                Vesta.info("  Total muestras: " + totalSamples);
//                Vesta.info("  Long: " + longCount + " (" + String.format("%.1f", (longCount * 100.0 / totalSamples)) + "%)");
//                Vesta.info("  Neutral: " + neutralCount + " (" + String.format("%.1f", (neutralCount * 100.0 / totalSamples)) + "%)");
//                Vesta.info("  Short: " + shortCount + " (" + String.format("%.1f", (shortCount * 100.0 / totalSamples)) + "%)");
//                Vesta.info("\n📈 TP/SL Promedio (Log Return ×10⁴):");
//                Vesta.info("  Long - TP: " + String.format("%.4f", longTpAvg * 10000) +
//                        ", SL: " + String.format("%.4f", longSlAvg * 10000) +
//                        ", Ratio: " + String.format("%.2f", longSlAvg > 0 ? longTpAvg / longSlAvg : 0));
//                Vesta.info("  Neutral - TP: " + String.format("%.4f", neutralTpAvg * 10000) +
//                        ", SL: " + String.format("%.4f", neutralSlAvg * 10000) +
//                        ", Ratio: " + String.format("%.2f", neutralSlAvg > 0 ? neutralTpAvg / neutralSlAvg : 0));
//                Vesta.info("  Short - TP: " + String.format("%.4f", shortTpAvg * 10000) +
//                        ", SL: " + String.format("%.4f", shortSlAvg * 10000) +
//                        ", Ratio: " + String.format("%.2f", shortSlAvg > 0 ? shortTpAvg / shortSlAvg : 0));

            // Mostrar en ventana
            JFrame frame = new JFrame("Análisis de Direcciones - " + symbol);
            frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
            frame.getContentPane().add(tabbedPane);
            frame.setSize(1000, 700);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);

        } catch (Exception e) {
            Vesta.error("Error mostrando distribución de direcciones: " + e.getMessage());
            e.printStackTrace();
        }
    }

    /**
     * Muestra la comparación de predicciones vs reales para TP y SL
     */
    public static void showPredictionComparison(
            String title,
            List<EngineUtils.ResultPrediction> results
    ) {
        if (results == null || results.isEmpty()) {
            Vesta.error("No hay resultados para mostrar");
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
            results.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::tpDiff));
            for (EngineUtils.ResultPrediction r : results) {
                actualTP.add(r.realTP());
                predictedTP.add(r.predTP());
                tpError.add(Math.abs(r.realTP() - r.predTP()));
            }
            results.sort(Comparator.comparingDouble(EngineUtils.ResultPrediction::lsDiff));
            for (EngineUtils.ResultPrediction r : results) {
                actualSL.add(r.realSL());
                predictedSL.add(r.predSL());
                slError.add(Math.abs(r.realSL() - r.predSL()));
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
    public static void plotRatioDistribution(String title, List<EngineUtils.ResultPrediction> results) {
        if (results == null || results.isEmpty()) return;

        // Calcular ratios
        List<Float> realRatios = new ArrayList<>();
        List<Float> predRatios = new ArrayList<>();

        for (EngineUtils.ResultPrediction r : results) {
            if (r.realSL() != 0) {
                realRatios.add(r.realTP() / r.realSL());
            }else {
                realRatios.add(0f);
            }
            if (r.predSL() != 0) {
                predRatios.add(r.predTP() / r.predSL());
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

        try {
            // Ordenar por ratio (de menor a mayor)
            List<BackTestEngine.CompleteTrade> sortedStats = new ArrayList<>(extraStats);
            sortedStats.sort(Comparator.comparingDouble(BackTestEngine.CompleteTrade::ratio));

            // Crear dataset para ROI vs Ratio
            XYSeriesCollection dataset = new XYSeriesCollection();
            XYSeries roiSeries = new XYSeries("ROI por Ratio");
            XYSeries cumulativeROISeries = new XYSeries("ROI Acumulado");

            double cumulativeROI = 0;
            for (BackTestEngine.CompleteTrade stat : sortedStats) {
                double roi = stat.pnlPercent() * 100; // Convertir a porcentaje
                float ratio = stat.ratio();

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

            // Calcular estadísticas
            double avgRatio = sortedStats.stream()
                    .mapToDouble(BackTestEngine.CompleteTrade::ratio)
                    .average()
                    .orElse(0);

            double positiveROICount = sortedStats.stream()
                    .filter(s -> s.pnlPercent() > 0)
                    .count();

            double winRate = (positiveROICount / sortedStats.size()) * 100;

//            Vesta.info("Estadísticas Ratio vs ROI:");
//            Vesta.info("  Ratio promedio: " + avgRatio);
//            Vesta.info("  Ratio mínimo: " + sortedStats.get(0).ratio());
//            Vesta.info("  Ratio máximo: " + sortedStats.get(sortedStats.size() - 1).ratio());
//            Vesta.info("  Win Rate: " + String.format("%.2f", winRate) + "%");
//            Vesta.info("  ROI total acumulado: " + String.format("%.2f", cumulativeROI) + "%");

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

        try {
            // Crear lista combinada de datos
            List<TPSLROIData> dataList = new ArrayList<>();
            for (int i = 0; i < ExtraData.size(); i++) {
                BackTestEngine.CompleteTrade stat = ExtraData.get(i);
                // Calcular magnitud del TP (en porcentaje de log return)
                float tpMagnitude = Math.abs(stat.tpPercent() * 10000); // Convertir a puntos base
                float slMagnitude = Math.abs(stat.slPercent() * 10000); // Convertir a puntos base
                float roi = (float) (stat.pnlPercent() * 100); // Convertir a porcentaje

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
    public static void plotPredictionVsRealScatter(List<EngineUtils.ResultPrediction> predictions, String title) {
        try {
            XYSeries longSeries = new XYSeries("LONG");
            XYSeries neutralSeries = new XYSeries("NEUTRAL");
            XYSeries shortSeries = new XYSeries("SHORT");

            for (EngineUtils.ResultPrediction pred : predictions) {
                float real = pred.realDir();
                float predVal = pred.predDir();

                if (real > THRESHOLD_RELATIVE) {
                    longSeries.add(real, predVal);
                } else if (real < -THRESHOLD_RELATIVE) {
                    shortSeries.add(real, predVal);
                } else {
                    neutralSeries.add(real, predVal);
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
    public static void plotErrorDistributionByDirection(List<EngineUtils.ResultPrediction> predictions, String title) {
        try {
            List<Float> longErrors = new ArrayList<>();
            List<Float> neutralErrors = new ArrayList<>();
            List<Float> shortErrors = new ArrayList<>();

            for (EngineUtils.ResultPrediction pred : predictions) {
                float error = pred.dirDiff();

                if (pred.realDir() > THRESHOLD_RELATIVE) {
                    longErrors.add(error);
                } else if (pred.realDir() < -THRESHOLD_RELATIVE) {
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
     * Clase auxiliar para bins de magnitud
     */
    private static class MagnitudeBin {
        float min;
        float max;
        int totalSamples;
        int correctSamples;
        double avgError;

        MagnitudeBin(float min, float max) {
            this.min = min;
            this.max = max;
            this.totalSamples = 0;
            this.correctSamples = 0;
            this.avgError = 0;
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
            plot.setRenderer(renderer);

            // Agregar anotaciones para cada trade
            if (trades != null && !trades.isEmpty()) {
                addTradeAnnotations(plot, trades, timeToIndex, candles);
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

    private static void addTradeAnnotations(XYPlot plot, List<BackTestEngine.CompleteTrade> trades,
                                            Map<Long, Integer> timeToIndex, List<CandleSimple> candles) {

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
            int entryIndex = findClosestCandleIndex(candles, trade.entryTime());
            int exitIndex = findClosestCandleIndex(candles, trade.exitTime());

            if (entryIndex >= 0) {
                CandleSimple entryCandle = candles.get(entryIndex);
                double entryX = entryCandle.openTime();

                // Agregar marker de entrada
                if (trade.direction() == Trading.DireccionOperation.LONG) {
                    entryLongSeries.add(entryX, trade.entryPrice());
                } else if (trade.direction() == Trading.DireccionOperation.SHORT) {
                    entryShortSeries.add(entryX, trade.entryPrice());
                }

                // Agregar líneas de TP y SL
                tpLinesSeries.add(entryX, trade.tpPrice());
                tpLinesSeries.add(exitIndex >= 0 ? candles.get(exitIndex).openTime() : entryX + 3600000, trade.tpPrice());

                slLinesSeries.add(entryX, trade.slPrice());
                slLinesSeries.add(exitIndex >= 0 ? candles.get(exitIndex).openTime() : entryX + 3600000, trade.slPrice());
            }

            // Agregar marker de salida
            if (exitIndex >= 0) {
                CandleSimple exitCandle = candles.get(exitIndex);
                double exitX = exitCandle.openTime();

                switch (trade.exitReason()) {
                    case LONG_TAKE_PROFIT:
                    case SHORT_TAKE_PROFIT:
                        exitProfitSeries.add(exitX, trade.exitPrice());
                        break;
                    case LONG_STOP_LOSS:
                    case SHORT_STOP_LOSS:
                        exitLossSeries.add(exitX, trade.exitPrice());
                        break;
                    default:
                        exitOtherSeries.add(exitX, trade.exitPrice());
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
        plot.setDataset(1, datasetMarkers);
        plot.setRenderer(1, markerRenderer);
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
                    .filter(t -> t.direction() == Trading.DireccionOperation.LONG && t.pnl() > 0)
                    .count();
            long longLosses = trades.stream()
                    .filter(t -> t.direction() == Trading.DireccionOperation.LONG && t.pnl() <= 0)
                    .count();
            long shortWins = trades.stream()
                    .filter(t -> t.direction() == Trading.DireccionOperation.SHORT && t.pnl() > 0)
                    .count();
            long shortLosses = trades.stream()
                    .filter(t -> t.direction() == Trading.DireccionOperation.SHORT && t.pnl() <= 0)
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

            String direction = trade.direction() == Trading.DireccionOperation.LONG ? "LONG" :
                    trade.direction() == Trading.DireccionOperation.SHORT ? "SHORT" : "NEUT";

            String exitReason = trade.exitReason().name()
                    .replace("LONG_", "")
                    .replace("SHORT_", "")
                    .replace("_", " ");

            data[i][0] = i + 1;
            data[i][1] = direction;
            data[i][2] = priceFormat.format(trade.entryPrice());
            data[i][3] = priceFormat.format(trade.tpPrice());
            data[i][4] = priceFormat.format(trade.slPrice());
            data[i][5] = priceFormat.format(trade.exitPrice());
            data[i][6] = pnlFormat.format(trade.pnlPercent());
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
            double totalPnL = trades.stream().mapToDouble(t -> t.pnlPercent()).sum();
            double avgPnL = totalPnL / trades.size();
            long wins = trades.stream().filter(t -> t.pnl() > 0).count();
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