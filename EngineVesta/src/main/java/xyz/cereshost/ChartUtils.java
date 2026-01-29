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
import xyz.cereshost.engine.BackTestEngine;
import xyz.cereshost.engine.EngineUtils;

import javax.swing.*;
import java.awt.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.List;

public class ChartUtils {


    public static DefaultCategoryDataset plot(
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

        return dataset;
    }

    public record DataPlot(
            String yLabel,
            List<Float> values
    ) {}

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
            org.jfree.data.xy.XYSeries tpSeries = new org.jfree.data.xy.XYSeries("TP (Take Profit)");
            org.jfree.data.xy.XYSeries slSeries = new org.jfree.data.xy.XYSeries("SL (Stop Loss)");

            for (int i = 0; i < tpValues.size(); i++) {
                tpSeries.add(i, tpValues.get(i));
                slSeries.add(i, slValues.get(i));
            }

            org.jfree.data.xy.XYSeriesCollection dataset = new org.jfree.data.xy.XYSeriesCollection();
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
            countPlot.setBackgroundPaint(Color.WHITE);
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

            tabbedPane.addTab("TP Predicho/Real", new ChartPanel(chartTP));


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
            tabbedPane.addTab("SL Predicho/Real", new ChartPanel(chartSL));

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

    public static void plotRatioVsROI(String title, List<BackTestEngine.ExtraDataPlot> extraStats) {
        if (extraStats == null || extraStats.isEmpty()) return;

        try {
            // Ordenar por ratio (de menor a mayor)
            List<BackTestEngine.ExtraDataPlot> sortedStats = new ArrayList<>(extraStats);
            sortedStats.sort(Comparator.comparingDouble(BackTestEngine.ExtraDataPlot::ratio));

            // Crear dataset para ROI vs Ratio
            XYSeriesCollection dataset = new XYSeriesCollection();
            org.jfree.data.xy.XYSeries roiSeries = new org.jfree.data.xy.XYSeries("ROI por Ratio");
            org.jfree.data.xy.XYSeries cumulativeROISeries = new org.jfree.data.xy.XYSeries("ROI Acumulado");

            double cumulativeROI = 0;
            for (int i = 0; i < sortedStats.size(); i++) {
                BackTestEngine.ExtraDataPlot stat = sortedStats.get(i);
                float roi = stat.pnlPercent() * 100; // Convertir a porcentaje
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
            plot.addRangeMarker(new org.jfree.chart.plot.ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));

            // Añadir línea en Ratio=1 (TP=SL) para referencia
            plot.addDomainMarker(new org.jfree.chart.plot.ValueMarker(1, Color.GREEN, new BasicStroke(1.0f)));

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
                    .mapToDouble(BackTestEngine.ExtraDataPlot::ratio)
                    .average()
                    .orElse(0);

            double positiveROICount = sortedStats.stream()
                    .filter(s -> s.pnlPercent() > 0)
                    .count();

            double winRate = (positiveROICount / sortedStats.size()) * 100;

            Vesta.info("Estadísticas Ratio vs ROI:");
            Vesta.info("  Ratio promedio: " + avgRatio);
            Vesta.info("  Ratio mínimo: " + sortedStats.get(0).ratio());
            Vesta.info("  Ratio máximo: " + sortedStats.get(sortedStats.size() - 1).ratio());
            Vesta.info("  Win Rate: " + String.format("%.2f", winRate) + "%");
            Vesta.info("  ROI total acumulado: " + String.format("%.2f", cumulativeROI) + "%");

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
            List<BackTestEngine.ExtraDataPlot> ExtraData
    ) {
        if ( ExtraData == null || ExtraData.isEmpty()) return;

        try {
            // Crear lista combinada de datos
            List<TPSLROIData> dataList = new ArrayList<>();
            for (int i = 0; i < ExtraData.size(); i++) {
                BackTestEngine.ExtraDataPlot stat = ExtraData.get(i);
                // Calcular magnitud del TP (en porcentaje de log return)
                float tpMagnitude = Math.abs(stat.tpPercent() * 10000); // Convertir a puntos base
                float slMagnitude = Math.abs(stat.slPercent() * 10000); // Convertir a puntos base
                float roi = stat.pnlPercent() * 100; // Convertir a porcentaje

                dataList.add(new TPSLROIData(tpMagnitude, slMagnitude, roi));
            }

            // Ordenar por magnitud del TP (menor a mayor)
            dataList.sort(Comparator.comparingDouble(TPSLROIData::tpMagnitude));

            // Crear datasets
            XYSeriesCollection datasetTP = new XYSeriesCollection();
            XYSeriesCollection datasetSL = new XYSeriesCollection();
            XYSeriesCollection datasetRatio = new XYSeriesCollection();

            org.jfree.data.xy.XYSeries tpSeries = new org.jfree.data.xy.XYSeries("ROI vs Magnitud TP");
            org.jfree.data.xy.XYSeries slSeries = new org.jfree.data.xy.XYSeries("ROI vs Magnitud SL");
            org.jfree.data.xy.XYSeries ratioSeries = new org.jfree.data.xy.XYSeries("ROI vs Ratio TP/SL");

            org.jfree.data.xy.XYSeries cumTPROI = new org.jfree.data.xy.XYSeries("ROI Acumulado TP");
            org.jfree.data.xy.XYSeries cumSLROI = new org.jfree.data.xy.XYSeries("ROI Acumulado SL");

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
            plotTP.addRangeMarker(new org.jfree.chart.plot.ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
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
            plotSL.addRangeMarker(new org.jfree.chart.plot.ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
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
            plotRatio.addRangeMarker(new org.jfree.chart.plot.ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
            plotRatio.addDomainMarker(new org.jfree.chart.plot.ValueMarker(1, Color.GREEN, new BasicStroke(1.0f)));
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
            org.jfree.data.xy.XYSeries highROI = new org.jfree.data.xy.XYSeries("ROI Alto (>5%)");
            org.jfree.data.xy.XYSeries mediumROI = new org.jfree.data.xy.XYSeries("ROI Medio (0-5%)");
            org.jfree.data.xy.XYSeries lowROI = new org.jfree.data.xy.XYSeries("ROI Bajo (-5-0%)");
            org.jfree.data.xy.XYSeries negativeROI = new org.jfree.data.xy.XYSeries("ROI Negativo (<-5%)");

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
            plot.addDomainMarker(new org.jfree.chart.plot.ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));
            plot.addRangeMarker(new org.jfree.chart.plot.ValueMarker(0, Color.GRAY, new BasicStroke(1.0f)));

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
}