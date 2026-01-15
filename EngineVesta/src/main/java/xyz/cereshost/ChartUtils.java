package xyz.cereshost;

import org.jfree.chart.*;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

import java.util.List;

public class ChartUtils {

    public static void plot(
            String title,
            String yLabel,
            List<Float> values
    ) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < values.size(); i++) {
            dataset.addValue(values.get(i), yLabel, String.valueOf(i + 1));
        }

        JFreeChart chart = ChartFactory.createLineChart(
                title,
                "Epoch",
                yLabel,
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
}

