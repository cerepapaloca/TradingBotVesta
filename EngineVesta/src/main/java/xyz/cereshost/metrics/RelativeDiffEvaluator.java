package xyz.cereshost.metrics;

public class RelativeDiffEvaluator extends AbstractDiffEvaluator {

    public RelativeDiffEvaluator() {
        this("relative_diff");
    }

    public RelativeDiffEvaluator(String name) {
        super(name, 2);
    }
}
