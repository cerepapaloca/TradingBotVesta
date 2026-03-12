package xyz.cereshost.vesta.core.trading;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.Getter;
import lombok.experimental.Delegate;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.core.exception.OperationFilled;
import xyz.cereshost.vesta.core.message.Notifiable;

import java.util.*;

import static xyz.cereshost.vesta.core.trading.DireccionOperation.LONG;
import static xyz.cereshost.vesta.core.trading.DireccionOperation.SHORT;

public interface TradingManager extends Notifiable {
    /**
     * Abre una operacion en futuros perpetuos
     * @param tpPercent porcentaje para él limite Take Profit
     * @param slPercent porcentaje para él limite Stop Loss
     * @param direccion Si la operación es Short/Venta o una operación Larga/Compra
     * @param amountUSD Cantidad de USDT/USDC a operar
     * @param leverage La cantidad de apaleamiento
     * @return la nueva instancia de la operación
     */
    @Nullable OpenOperation open(double tpPercent, double slPercent, @NotNull DireccionOperation direccion, double amountUSD, int leverage);

    /**
     * Cierras una Operación previamente Abierta
     * @param reason Razón de la salida
     * @param openOperation La operacion que se va a cerrar
     */
    @Nullable CloseOperation close(ExitReason reason, OpenOperation openOperation);

    @Nullable LimiteOperation limit(double entryPrice, double tpPercent, double slPercent, @NotNull DireccionOperation direccion, double amountUSD, int leverage);

    int closeSize();

    int openSize();

    @NotNull List<OpenOperation> getOpens();

    @NotNull List<CloseOperation> getCloses();

    Market getMarket();

    double getAvailableBalance();

    /**
     * Obtienes el preio actual del mercado
     * @return el precio absoluto actual
     */

    double getCurrentPrice();

    /**
     * La hora actual (En el backtest se usa la hora del propio backtest)
     * @return La hora actual
     */

    long getCurrentTime();

    default void log(String s){}

    default boolean hasOpenOperation() {
        return openSize() > 0;
    }

    @Data
    @EqualsAndHashCode(callSuper = true)
    abstract class CloseOperation extends OpenOperation {

        private final double exitPrice;
        private final long exitTime;
        private final ExitReason reason;

        @Delegate
        private final OpenOperation openOperation;

        public CloseOperation(double exitPrice, long exitTime, ExitReason reason, OpenOperation openOperation) {
            super(openOperation.getTradingManager(),
                    openOperation.getEntryPrice(),
                    openOperation.getTpPercent(),
                    openOperation.getSlPercent(),
                    openOperation.getDireccion(),
                    openOperation.getAmountInitUSDT(),
                    openOperation.getLeverage()
            );
            this.exitPrice = exitPrice;
            this.exitTime = exitTime;
            this.reason = reason;
            this.openOperation = openOperation;
        }
        /* Methods que implica una modificación a la operación */
        @Override
        public void close(ExitReason reason){
            throw new OperationFilled(this);
        }
        @Override
        public void close(){
            throw new OperationFilled(this);
        }
        @Override
        public void setTpPercent(double tpPercent) {
            throw new OperationFilled(this);
        }
        @Override
        public void setSlPercent(double slPercent) {
            throw new OperationFilled(this);
        }
        @Override
        public void nextMinute(){
            throw new OperationFilled(this);
        }
    }

    @EqualsAndHashCode(callSuper = true)
    @Data
    abstract class OpenOperation extends LimiteOperation {
        private double tpPercent;
        private double slPercent;

        private final long entryTime;
        /**
         * Lista de banderas para identificar una operación
         */
        public final Set<String> flags = new HashSet<>();

        // Ojo el TP y SL en Porcentajes ABS sin apalancar
        public OpenOperation(@NotNull TradingManager tradingManager, double entryPrice, double tpPercent, double slPercent, @NotNull DireccionOperation direccion, double amountUSDT, int leverage) {
            super(entryPrice, direccion, tpPercent, slPercent, amountUSDT, leverage, tradingManager);
            this.tpPercent = tpPercent;
            this.slPercent = slPercent;

            this.entryTime = tradingManager.getCurrentTime();
        }

        public void close(ExitReason reason) {
            tradingManager.close(reason, this);
        }

        public void close() {
            tradingManager.close(ExitReason.STRATEGY, this);
        }

        /**
         * Obtienes el precio absoluto del limíte de Stop Loss
         * @return Precio
         */
        public double getSlPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? slPercent : -slPercent)*0.01));
        }

        /**
         * Obtienes el precio absoluto del limíte de Take Profit
         * @return Precio
         */
        public double getTpPrice() {
            return entryPrice + (entryPrice * ((direccion.equals(SHORT) ? -tpPercent : tpPercent)*0.01));
        }

        /**
         * Obtienes la diferencia porcentual desde el precio de apertura
         * @return diferencia porcentual -/+
         */
        public double getDiffPercent() {
            double currentPrice = tradingManager.getCurrentPrice();
            return ((currentPrice - entryPrice) / currentPrice) * 100;
        }

        /**
         * Obtienes el roi de la operación sin comisión y sin apalancamiénto
         * @return El Roi crudo
         */
        public double getRoiRaw() {
            if (isUpDireccion()){
                return getDiffPercent();
            } else {
                return -getDiffPercent();
            }
        }

        /**
         * Devuelve si la operacion en este instante de tiempo está en positivo
         * <strong>No tiene en cuenta la comisión y el apalancamiénto</strong>
         * @return true si es rentable, false en caso contrario
         */
        public boolean isProfit(){
            return getRoiRaw() > 0;
        }

        public boolean isUpDireccion(){
            return direccion == LONG;
        }

        /**
         * Devuelve true si la operacion aún sigue abierta o solo está en modo lectura
         * @return true si está abierto y modificable false en caso contrario
         */
        public boolean isOpen(){
            return !(this instanceof CloseOperation);
        }

        /**
         * La cantidad de minutos que lleva abierto una operación
         */
        private int minutesOpen = 0;

        /**
         * Suma 1+ a la cantidad de minutos que lleva abierto la operación
         */
        public void nextMinute() {
            minutesOpen++;
        }

        @Override
        public void setTimeInForce(TimeInForce timeInForce) {
            throw new UnsupportedOperationException();
        }
    }

    @Data
    abstract class LimiteOperation {
        protected final UUID uuid = UUID.randomUUID();
        protected final double entryPrice;
        @NotNull
        protected final DireccionOperation direccion;
        protected final double originalTpPercent;
        protected final double originalSlPercent;
        protected final double amountInitUSDT;
        protected final int leverage;
        @NotNull
        protected final TradingManager tradingManager;

        @NotNull
        protected TimeInForce timeInForce = TimeInForce.GTC;
    }

    /**
     * Razones para salir de una operación
     */
    enum ExitReason {
        /**
         * Cierre al cruzar él limite Take Profit en operación Long
         */
        LONG_TAKE_PROFIT,
        /**
         * Cierre al cruzar él limite Stop Loss en operación Long
         */
        LONG_STOP_LOSS,
        /**
         * Cierre al cruzar él limite Take Profit en operación short
         */
        SHORT_TAKE_PROFIT,
        /**
         * Cierre al cruzar él limite Stop Loss en operación short
         */
        SHORT_STOP_LOSS,
        /**
         * Se cierra por el exceder tiempo máximo (Se tiene añadir de forma manual)
         */
        TIMEOUT,
        /**
         * Cierra por qué la estrategia lo indica
         */
        STRATEGY,
        /**
         * Cierra por qué la estrategia lo indica (Se debe abrir una operacion inversa)
         */
        STRATEGY_INVERSION,
        /**
         * Cierre en caso de error
         */
        NO_DATA_ERROR;

        public boolean isTakeProfit() {
            return this == LONG_TAKE_PROFIT || this == SHORT_TAKE_PROFIT;
        }

        public boolean isStopLoss() {
            return this == LONG_STOP_LOSS || this == SHORT_STOP_LOSS;
        }
    }

}
