package xyz.cereshost.vesta.core.trading;

import lombok.Data;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.core.message.Notifiable;

import java.util.*;

import static xyz.cereshost.vesta.core.trading.Trading.DireccionOperation.LONG;
import static xyz.cereshost.vesta.core.trading.Trading.DireccionOperation.SHORT;

public interface Trading extends Notifiable {
    // Abrir una operación
    OpenOperation open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage);

    // Cerrar una operación manualmente
    void close(ExitReason reason, OpenOperation openOperation);

    int closeSize();

    int openSize();

    default boolean hasOpenOperation() {
        return openSize() > 0;
    }

    @NotNull List<OpenOperation> getOpens();

    @NotNull List<CloseOperation> getCloses();

    Market getMarket();

    double getAvailableBalance();

    double getCurrentPrice();

    default void log(String s){};

    @Data
    abstract class CloseOperation {

        private final double exitPrice;
        private final ExitReason reason;
        private final long exitTime;
        private final long entryTime;
        private final UUID uuidOpenOperation;
        private final OpenOperation openOperation;

        public CloseOperation(double exitPrice, long exitTime, long entryTime, ExitReason reason, OpenOperation openOperation) {
            this.exitPrice = exitPrice;
            this.reason = reason;
            this.exitTime = exitTime;
            this.entryTime = entryTime;
            this.uuidOpenOperation = openOperation.uuid;
            this.openOperation = openOperation;
        }
    }

    @SuppressWarnings("unused")
    @Data
    abstract class OpenOperation {
        private final UUID uuid = UUID.randomUUID();
        private double tpPercent;
        private double slPercent;
        private int minutesOpen = 0;
        private double lastExitPrices;

        private long entryTime;

        private final double entryPrice;
        private final double originalTpPercent;
        private final double originalSlPercent;
        private final DireccionOperation direccion;
        private final double amountInitUSDT;
        private final int leverage;
        private final Trading trading;
        public final Set<String> flags = new HashSet<>();

        // Ojo el TP y SL en Porcentajes ABS sin apalancar
        public OpenOperation(@NotNull Trading trading, double entryPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            this.tpPercent = tpPercent;
            this.slPercent = slPercent;

            this.originalTpPercent = tpPercent;
            this.originalSlPercent = slPercent;

            this.entryPrice = entryPrice;
            this.direccion = direccion;
            this.amountInitUSDT = amountUSDT;
            this.leverage = leverage;
            this.trading = trading;
        }

        public double getSlPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? slPercent : -slPercent)*0.01));
        }

        public double getTpPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? -tpPercent : tpPercent)*0.01));
        }

        public double getDiffPercent() {
            double currentPrice = trading.getCurrentPrice();
            return ((currentPrice - entryPrice) / currentPrice) * 100;
        }

        public double getRoiRaw() {
            if (isUpDireccion()){
                return getDiffPercent();
            } else {
                return -getDiffPercent();
            }
        }

        public boolean isProfit(){
            return getRoiRaw() > 0;
        }

        public boolean isUpDireccion(){
            return direccion == LONG;
        }

        /**
         * Suma 1+ a la cantidad de velas que lleva abierto la operación
         */

        public void next() {
            minutesOpen++;
        }
    }

    enum ExitReason {
        LONG_TAKE_PROFIT,
        LONG_STOP_LOSS,
        SHORT_TAKE_PROFIT,
        SHORT_STOP_LOSS,
        TIMEOUT,
        STRATEGY,
        STRATEGY_INVERSION,
        NO_DATA_ERROR;

        public boolean isTakeProfit() {
            return this == LONG_TAKE_PROFIT || this == SHORT_TAKE_PROFIT;
        }

        public boolean isStopLoss() {
            return this == LONG_STOP_LOSS || this == SHORT_STOP_LOSS;
        }
    }

    enum DireccionOperation {
        SHORT,
        LONG,
        // OJO no se puede operar con neutral solo es una forma de identificar operacion sin movimiento ósea no hacer nada
        NEUTRAL
    }

}
