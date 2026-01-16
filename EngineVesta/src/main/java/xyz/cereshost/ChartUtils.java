package xyz.cereshost;

import org.jfree.chart.*;
import org.jfree.chart.axis.DateAxis;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.category.DefaultCategoryDataset;
import org.jfree.data.xy.DefaultHighLowDataset;
import org.jfree.data.xy.OHLCDataset;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;

import javax.swing.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
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

    public class CandleChartUtils {

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

        public static void showPriceComparison(String title, List<Double> actualPrices,
                                               List<Double> predictedPrices, List<String> labels) {
            if (actualPrices.size() != predictedPrices.size()) {
                Vesta.error("Los arrays de precios no tienen el mismo tamaño");
                return;
            }

            try {
                // Crear dataset para línea
                org.jfree.data.xy.XYSeries actualSeries = new org.jfree.data.xy.XYSeries("Actual");
                org.jfree.data.xy.XYSeries predictedSeries = new org.jfree.data.xy.XYSeries("Predicho");

                for (int i = 0; i < actualPrices.size(); i++) {
                    actualSeries.add(i, actualPrices.get(i));
                    predictedSeries.add(i, predictedPrices.get(i));
                }

                org.jfree.data.xy.XYSeriesCollection dataset = new org.jfree.data.xy.XYSeriesCollection();
                dataset.addSeries(actualSeries);
                dataset.addSeries(predictedSeries);

                // Crear gráfico de línea
                JFreeChart chart = ChartFactory.createXYLineChart(
                        title,
                        "Muestra",
                        "Precio",
                        dataset
                );

                // Mostrar en ventana
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

        public static void showDataDistribution(String title, float[][][] X, float[] y, String symbol) {
            try {
                // Extraer precios de cierre de la última vela de cada muestra
                List<Double> closes = new ArrayList<>();
                List<Double> targets = new ArrayList<>();

                for (int i = 0; i < X.length; i++) {
                    // Suponiendo que el precio de cierre está en alguna posición
                    // Ajusta según tu estructura de datos
                    if (X[i].length > 0 && X[i][X[i].length-1].length > 0) {
                        closes.add((double) X[i][X[i].length-1][3]); // Índice 3 = close?
                        targets.add((double) y[i]);
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

