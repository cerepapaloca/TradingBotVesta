package xyz.cereshost;

import org.jetbrains.annotations.NotNull;
import org.jfree.chart.*;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.DefaultHighLowDataset;
import org.jfree.data.xy.OHLCDataset;
import org.jfree.data.xy.XYSeriesCollection;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;

import javax.swing.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
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

        public static void showDataDistribution(String title, float[][][] X, float[] y, String symbol) {
            try {
                // Extraer precios de cierre de la última vela de cada muestra
                List<Float> closes = new ArrayList<>();
                List<Float> targets = new ArrayList<>();

                for (int i = 0; i < X.length; i++) {
                    // Suponiendo que el precio de cierre está en alguna posición
                    // Ajusta según tu estructura de datos
                    if (X[i].length > 0 && X[i][X[i].length-1].length > 0) {
                        closes.add(X[i][X[i].length-1][3]); // Índice 3 = close?
                        targets.add(y[i]);
                    }
                }

                // Crear dataset
                org.jfree.data.xy.XYSeries series = new org.jfree.data.xy.XYSeries("Datos");
                for (int i = 0; i < closes.size(); i++) {
                    series.add(i, closes.get(i));
                }

                org.jfree.data.xy.XYSeriesCollection dataset = new org.jfree.data.xy.XYSeriesCollection();
                dataset.addSeries(series);

                // Crear gráfico
                JFreeChart chart = ChartFactory.createScatterPlot(
                        title + " - " + symbol + " (Distribución de Datos)",
                        "Índice de Muestra",
                        "Precio de Cierre",
                        dataset
                );

                // Mostrar en ventana
                ChartPanel chartPanel = new ChartPanel(chart);
                chartPanel.setPreferredSize(new Dimension(1200, 600));

                JFrame frame = new JFrame("Distribución de Datos - " + symbol);
                frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                frame.getContentPane().add(chartPanel);
                frame.pack();
                frame.setLocationRelativeTo(null);
                frame.setVisible(true);

                Vesta.info("Mostrando distribución de " + closes.size() + " muestras para " + symbol);

            } catch (Exception e) {
                Vesta.error("Error mostrando distribución: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }
}

