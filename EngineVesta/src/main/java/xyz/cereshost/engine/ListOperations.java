package xyz.cereshost.engine;

import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.List;

public class ListOperations {

    private final ArrayList<BackTestEngine.OpenOperation> openOperations = new ArrayList<>();
    private final ArrayList<BackTestEngine.CloseOperation> closeOperations = new ArrayList<>();

    private final BackTestEngine backTestEngine;

    public ListOperations(BackTestEngine backTestEngine) {
        this.backTestEngine = backTestEngine;
    }

    public int sizeOpens(){
        return openOperations.size();
    }
    public int sizeClosed(){
        return closeOperations.size();
    }

    public void add(BackTestEngine.OpenOperation openOperation) {
        openOperations.add(openOperation);
    }

    public @NotNull List<BackTestEngine.OpenOperation> iteratorOpens() {
        return new ArrayList<>(openOperations);
    }

    public @NotNull List<BackTestEngine.CloseOperation> iteratorCloses() {
        return new ArrayList<>(closeOperations);
    }

    public void close(BackTestEngine.CloseOperation closeOperation) {
        closeOperations.add(closeOperation);
    }

    public void computeCloses() {
        for (BackTestEngine.CloseOperation closeOperation : closeOperations) {
            openOperations.remove(closeOperation.openOperationLastEstate());
            backTestEngine.computeFinal(closeOperation);
        }
        closeOperations.clear();
    }
}
