package xyz.cereshost.metrics;

public class MinDiffEvaluator extends AbstractDiffEvaluator {

    public MinDiffEvaluator() {
        this("min_diff");
    }

    public MinDiffEvaluator(String name) {
        super(name, 1);
    }
}
