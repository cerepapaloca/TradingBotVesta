package xyz.cereshost;

// Agrega esta clase
public class EarlyStopping {
    private int patience;
    private double bestLoss = Double.POSITIVE_INFINITY;
    private int wait = 0;

    public EarlyStopping(int patience) {
        this.patience = patience;
    }

    public boolean shouldStop(double currentLoss) {
        if (currentLoss < bestLoss) {
            bestLoss = currentLoss;
            wait = 0;
        } else {
            wait++;
        }
        return wait >= patience;
    }
}