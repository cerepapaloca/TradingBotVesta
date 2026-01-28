package xyz.cereshost.trading;

import lombok.Data;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.market.Market;

import java.util.List;
import java.util.UUID;

import static xyz.cereshost.trading.Trading.DireccionOperation.SHORT;

public interface Trading {
    // Abrir una operación
    void open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage);

    // Cerrar una operación manualmente
    void close(ExitReason reason, UUID uuidOpenOperation);

    int closeSize();

    int openSize();

    @NotNull List<OpenOperation> getOpens();

    @NotNull List<CloseOperation> getCloses();
        // Sincronizar estado con el mercado (API o Simulación)
    void updateState(String symbol);

    Market getMarket();

    double getAvailableBalance();

     default void log(String s){};

    @Data
    abstract class CloseOperation {


        private final double exitPrice;
        private final ExitReason reason;
        private final long exitTime;
        private final long entryTime;
        private final UUID uuidOpenOperation;

        public CloseOperation(double exitPrice, long exitTime, long entryTime, ExitReason reason, UUID uuidOpenOperation) {
            this.exitPrice = exitPrice;
            this.reason = reason;
            this.exitTime = exitTime;
            this.entryTime = entryTime;
            this.uuidOpenOperation = uuidOpenOperation;
        }
    }

    @Data
    abstract class OpenOperation {
        private final UUID uuid = UUID.randomUUID();
        private double tpPercent;
        private double slPercent;
        private int countCandles = 0;
        private double lastExitPrices;

        private long entryTime;

        private final double entryPrice;
        private final DireccionOperation direccion;
        private final double amountInitUSDT;
        private final int leverage;

        // Ojo el TP y SL en Porcentajes ABS sin apalancar
        public OpenOperation(double entryPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            this.tpPercent = tpPercent;
            this.slPercent = slPercent;

            this.entryPrice = entryPrice;
            this.direccion = direccion;
            this.amountInitUSDT = amountUSDT;
            this.leverage = leverage;
        }

        public double getSlPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? slPercent : -slPercent)*0.01));
        }

        public double getTpPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? -tpPercent : tpPercent)*0.01));
        }

        public void next() {
            countCandles++;
        }
    }

    enum ExitReason {
        LONG_TAKE_PROFIT,
        LONG_STOP_LOSS,
        SHORT_TAKE_PROFIT,
        SHORT_STOP_LOSS,
        TIMEOUT,
        STRATEGY,
        NO_DATA_ERROR
    }

    enum DireccionOperation {
        SHORT,
        LONG,
        // OJO no se puede operar con neutral solo es una forma de identificar operacion sin movimiento ósea no hacer
        NEUTRAL
    }

}
