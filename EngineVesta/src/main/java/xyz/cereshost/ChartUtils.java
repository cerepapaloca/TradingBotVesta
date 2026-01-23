package xyz.cereshost;

import org.jetbrains.annotations.NotNull;
import org.jfree.chart.*;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.CategoryPlot;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.DefaultHighLowDataset;
import org.jfree.data.xy.OHLCDataset;
import org.jfree.data.xy.XYSeriesCollection;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.engine.EngineUtils;

import javax.swing.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

public class ChartUtils {


    public static void plot(
            String title,
            String xLabel,
            List<DataPlot> plots
    ) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (DataPlot plot : plots) {
            List<Float> values = plot.values();
            String seriesName = plot.yLabel();

            for (int i = 0; i < values.size(); i++) {
                dataset.addValue(values.get(i), seriesName, String.valueOf(i + 1));
            }
        }

        JFreeChart chart = ChartFactory.createLineChart(
                title,
                xLabel,
                "Value",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );


        ChartFrame frame = new ChartFrame(title, chart);
        frame.pack();
        frame.setVisible(true);
    }

    public record DataPlot(
            String yLabel,
            List<Float> values
    ) {}

    public static class CandleChartUtils {

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

        public static void showPriceComparison(
                String title,
                List<Float> actualPrices,
                List<Float> predictedPrices
        ) {
            if (actualPrices.size() != predictedPrices.size()) {
                Vesta.error("Los arrays de precios no tienen el mismo tamaño");
                return;
            }

            try {
                // Dataset
                XYSeriesCollection dataset = getXySeriesCollection(actualPrices, predictedPrices);

                // Crear gráfico
                JFreeChart chart = ChartFactory.createXYLineChart(
                        title,
                        "Muestra",
                        "Precio",
                        dataset
                );

                XYPlot plot = chart.getXYPlot();

                double min = Double.POSITIVE_INFINITY;
                double max = Double.NEGATIVE_INFINITY;

                for (int i = 0; i < actualPrices.size(); i++) {
                    double a = actualPrices.get(i);
                    double p = predictedPrices.get(i);
                    min = Math.min(min, Math.min(a, p));
                    max = Math.max(max, Math.max(a, p));
                }

                // margen del 2%
                double padding = (max - min) * 0.02;
                if (padding == 0) padding = max * 0.001; // por si todo es constante

                NumberAxis rangeAxis = (NumberAxis) plot.getRangeAxis();
                rangeAxis.setAutoRange(false);
                rangeAxis.setLowerBound(min - padding);
                rangeAxis.setUpperBound(max + padding);

                // ===== MEJORAS VISUALES OPCIONALES =====
                rangeAxis.setAutoTickUnitSelection(true);
                plot.setDomainGridlinesVisible(true);
                plot.setRangeGridlinesVisible(true);

                // Mostrar
                ChartPanel chartPanel = new ChartPanel(chart);
                chartPanel.setPreferredSize(new Dimension(1200, 600));

                JFrame frame = new JFrame("Comparación Predicciones");
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.getContentPane().add(chartPanel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);

                Vesta.info("Mostrando comparación de " + actualPrices.size() + " predicciones");

            } catch (Exception e) {
                Vesta.error("Error mostrando comparación: " + e.getMessage());
                e.printStackTrace();
            }
        }

        private static @NotNull XYSeriesCollection getXySeriesCollection(List<Float> actualPrices, List<Float> predictedPrices) {
            org.jfree.data.xy.XYSeries actualSeries = new org.jfree.data.xy.XYSeries("Actual");
            org.jfree.data.xy.XYSeries predictedSeries = new org.jfree.data.xy.XYSeries("Predicho");

            for (int i = 0; i < actualPrices.size(); i++) {
                actualSeries.add(i, actualPrices.get(i));
                predictedSeries.add(i, predictedPrices.get(i));
            }

            XYSeriesCollection dataset = new XYSeriesCollection();
            dataset.addSeries(actualSeries);
            dataset.addSeries(predictedSeries);
            return dataset;
        }

        /**
         * Muestra la distribución de datos con dos salidas (TP y SL)
         */
        public static void showDataDistribution(String title, float[][] y, String symbol) {
            try {
                // Extraer TP y SL de los datos de salida
                List<Float> tpValues = new ArrayList<>();
                List<Float> slValues = new ArrayList<>();
                List<Float> ratioValues = new ArrayList<>();
                int j = 0;
                for (float[] floats : y) {
                    if (floats.length >= 2) {
                        float tp = floats[0];
                        float sl = floats[1];
                        tpValues.add(tp);
                        slValues.add(sl);
                        if (sl > 0) {
                            ratioValues.add(tp / sl);
                        }else {
                            ratioValues.add(0f);
                        }
                    }
                }

                // Crear dataset para TP y SL
                org.jfree.data.xy.XYSeries tpSeries = new org.jfree.data.xy.XYSeries("TP (Take Profit)");
                org.jfree.data.xy.XYSeries slSeries = new org.jfree.data.xy.XYSeries("SL (Stop Loss)");
                org.jfree.data.xy.XYSeries ratioSeries = new org.jfree.data.xy.XYSeries("Ratio TP/SL");

                for (int i = 0; i < tpValues.size(); i++) {
                    tpSeries.add(i, tpValues.get(i) * 10000);
                    slSeries.add(i, slValues.get(i) * 10000);
                    if (i < ratioValues.size()) {
                        ratioSeries.add(i, ratioValues.get(i));
                    }
                }

                org.jfree.data.xy.XYSeriesCollection dataset = new org.jfree.data.xy.XYSeriesCollection();
                dataset.addSeries(ratioSeries);
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
                plot.getRenderer().setSeriesPaint(0, Color.BLUE);   // Ratio en azul
                plot.getRenderer().setSeriesPaint(1, Color.GREEN);  // TP en verde
                plot.getRenderer().setSeriesPaint(2, Color.RED);    // SL en rojo

                // Mostrar en ventana
                ChartPanel chartPanel = new ChartPanel(chart);
                chartPanel.setPreferredSize(new Dimension(1200, 600));

                JFrame frame = new JFrame("Distribución de Datos - " + symbol);
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.getContentPane().add(chartPanel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);

                Vesta.info("Mostrando distribución de " + tpValues.size() + " muestras para " + symbol);
                Vesta.info("TP promedio: " + tpValues.stream().mapToDouble(f -> f).average().orElse(0));
                Vesta.info("SL promedio: " + slValues.stream().mapToDouble(f -> f).average().orElse(0));
                Vesta.info("Ratio TP/SL promedio: " + ratioValues.stream().mapToDouble(f -> f).average().orElse(0));

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

                // Crear un panel con 4 gráficos
                JPanel panel = new JPanel(new GridLayout(2, 2));

                // Gráfico 1: TP predicho vs real
                XYSeriesCollection datasetTP = new XYSeriesCollection();
                org.jfree.data.xy.XYSeries actualTPSeries = new org.jfree.data.xy.XYSeries("TP Real");
                org.jfree.data.xy.XYSeries predictedTPSeries = new org.jfree.data.xy.XYSeries("TP Predicho");
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
                panel.add(new ChartPanel(chartTP));

                // Gráfico 2: SL predicho vs real
                XYSeriesCollection datasetSL = new XYSeriesCollection();
                org.jfree.data.xy.XYSeries actualSLSeries = new org.jfree.data.xy.XYSeries("SL Real");
                org.jfree.data.xy.XYSeries predictedSLSeries = new org.jfree.data.xy.XYSeries("SL Predicho");
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
                panel.add(new ChartPanel(chartSL));

                // Gráfico 3: Error de predicción
                XYSeriesCollection datasetError = new XYSeriesCollection();
                org.jfree.data.xy.XYSeries tpErrorSeries = new org.jfree.data.xy.XYSeries("Error TP");
                org.jfree.data.xy.XYSeries slErrorSeries = new org.jfree.data.xy.XYSeries("Error SL");
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
                panel.add(new ChartPanel(chartError));

                // Gráfico 4: Ratio TP/SL real vs predicho
                XYSeriesCollection datasetRatio = new XYSeriesCollection();
                org.jfree.data.xy.XYSeries actualRatioSeries = new org.jfree.data.xy.XYSeries("Ratio Real");
                org.jfree.data.xy.XYSeries predictedRatioSeries = new org.jfree.data.xy.XYSeries("Ratio Predicho");
                for (int i = 0; i < results.size(); i++) {
                    EngineUtils.ResultPrediction r = results.get(i);
                    float actualRatio = r.realSL() != 0 ? r.realTP() / r.realSL() : 0;
                    float predictedRatio = r.predSL() != 0 ? r.predTP() / r.predSL() : 0;
                    actualRatioSeries.add(i, actualRatio);
                    predictedRatioSeries.add(i, predictedRatio);
                }
                datasetRatio.addSeries(actualRatioSeries);
                datasetRatio.addSeries(predictedRatioSeries);

                JFreeChart chartRatio = ChartFactory.createXYLineChart(
                        "Ratio TP/SL",
                        "Muestra",
                        "Ratio",
                        datasetRatio
                );
                ((XYPlot) chartRatio.getPlot()).getRenderer().setSeriesPaint(0, Color.MAGENTA);
                ((XYPlot) chartRatio.getPlot()).getRenderer().setSeriesPaint(1, Color.YELLOW);
                ChartPanel cp = new ChartPanel(chartRatio);
                cp.setSize(800, 600);
                panel.add(cp);
                // Mostrar ventana
                JFrame frame = new JFrame(title);
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.getContentPane().add(new JScrollPane(panel));
                frame.setSize(1600, 1200);
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);

                Vesta.info("Mostrando comparación de " + results.size() + " predicciones");

            } catch (Exception e) {
                Vesta.error("Error mostrando comparación: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    /**
     * Muestra la precisión del modelo por magnitud del movimiento
     */
    public static void plotPrecisionByMagnitude(String title, List<EngineUtils.ResultPrediction> results) {
        if (results == null || results.isEmpty()) return;

        // Ordenar por magnitud del TP real
        List<EngineUtils.ResultPrediction> sorted = new ArrayList<>(results);
        sorted.sort(Comparator.comparingDouble(r -> Math.abs(r.realTP())));

        int numBins = 10;
        int binSize = Math.max(1, sorted.size() / numBins);

        DefaultCategoryDataset datasetDirection = new DefaultCategoryDataset();
        DefaultCategoryDataset datasetRatio = new DefaultCategoryDataset();

        for (int i = 0; i < numBins; i++) {
            int start = i * binSize;
            int end = (i == numBins - 1) ? sorted.size() : Math.min(sorted.size(), (i + 1) * binSize);

            if (start >= end) continue;

            List<EngineUtils.ResultPrediction> subList = sorted.subList(start, end);

            // Calcular métricas
            long directionalHits = subList.stream()
                    .filter(r -> (r.predTP() > r.predSL()) == (r.realTP() > r.realSL()))
                    .count();

            long profitableHits = subList.stream()
                    .filter(r -> {
                        boolean predProfitable = r.predTP() > r.predSL();
                        boolean realProfitable = r.realTP() > r.realSL();
                        return predProfitable && realProfitable;
                    })
                    .count();

            // Magnitud promedio del TP real en este bin
            double avgMagnitude = subList.stream()
                    .mapToDouble(EngineUtils.ResultPrediction::realTP)
                    .average().orElse(0);

            // Calcular porcentajes
            double directionalAccuracy = (double) directionalHits / subList.size() * 100.0;
            double profitableAccuracy = (double) profitableHits / subList.size() * 100.0;

            String binLabel = String.format("%.4f", avgMagnitude * 100) + "%";
            datasetDirection.addValue(directionalAccuracy, "Precisión Direccional", binLabel);
            datasetRatio.addValue(profitableAccuracy, "Trades Rentables", binLabel);
        }

        // Crear gráficos
        JFreeChart chartDirection = ChartFactory.createBarChart(
                title + " - Precisión Direccional",
                "Magnitud del TP Real (%)",
                "Precisión (%)",
                datasetDirection,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        JFreeChart chartRatio = ChartFactory.createBarChart(
                title + " - Trades Rentables",
                "Magnitud del TP Real (%)",
                "Precisión (%)",
                datasetRatio,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );

        // Configurar rangos
        CategoryPlot plot1 = chartDirection.getCategoryPlot();
        NumberAxis rangeAxis1 = (NumberAxis) plot1.getRangeAxis();
        rangeAxis1.setRange(0.0, 100.0);

        CategoryPlot plot2 = chartRatio.getCategoryPlot();
        NumberAxis rangeAxis2 = (NumberAxis) plot2.getRangeAxis();
        rangeAxis2.setRange(0.0, 100.0);

        // Mostrar en panel dividido
        JPanel panel = new JPanel(new GridLayout(2, 1));
        panel.add(new ChartPanel(chartDirection));
        panel.add(new ChartPanel(chartRatio));

        JFrame frame = new JFrame(title);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(panel);
        frame.pack();
        frame.setSize(1200, 800);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
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

        org.jfree.data.xy.XYSeries realSeries = new org.jfree.data.xy.XYSeries("Ratio Real TP/SL");
        org.jfree.data.xy.XYSeries predSeries = new org.jfree.data.xy.XYSeries("Ratio Predicho TP/SL");

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
        plot.addRangeMarker(new org.jfree.chart.plot.ValueMarker(2.0, Color.RED, new BasicStroke(2.0f)));

        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new Dimension(1200, 600));

        JFrame frame = new JFrame("Distribución de Ratios");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().add(chartPanel);
        frame.pack();
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }
}