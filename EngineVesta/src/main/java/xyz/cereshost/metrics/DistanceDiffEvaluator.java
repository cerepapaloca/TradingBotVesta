package xyz.cereshost.metrics;

public class DistanceDiffEvaluator extends AbstractDiffEvaluator {

    public DistanceDiffEvaluator() {
        this("distance_diff");
    }

    public DistanceDiffEvaluator(String name) {
        super(name, 0);
    }
}
