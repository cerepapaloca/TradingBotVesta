package xyz.cereshost.vesta.core.trading;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.core.message.MediaNotification;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

public class TradingBinance implements Trading {

    private static final double POSITION_EPSILON = 1.0e-8;
    private final BinanceApi binanceApi;
    private int lastLeverage = 1;
    private double lastBalance = 0;
    @Setter
    private TradingTickLoop tradingTickLoop;
    @Getter @Setter @NotNull
    private MediaNotification mediaNotification;

    // Mapa para vincular tu UUID interno con los IDs de órdenes de Binance
    private final ConcurrentHashMap<UUID, BinanceOpenOperation> activeOperations = new ConcurrentHashMap<>();
    private final List<CloseOperation> closedOperations = Collections.synchronizedList(new ArrayList<>());
    private final Market market;

    public TradingBinance(@NotNull BinanceApi binanceApi, @Nullable MediaNotification mediaNotification, @NotNull Market market) {
        this.binanceApi = binanceApi;
        this.market = market;
        this.mediaNotification = Objects.requireNonNullElse(mediaNotification, MediaNotification.empty());
    }

    @Override
    public OpenOperation open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
        String symbol = getMarket().getSymbol();
        try {
            CountDownLatch latch = new CountDownLatch(3);

            tradingTickLoop.getExecutor().execute(() -> {
                if (lastLeverage != leverage) {
                    binanceApi.changeLeverage(symbol, leverage);
                }
                lastLeverage = leverage;
                latch.countDown();
            });
            AtomicReference<Double> safeAmountUSDT = new AtomicReference<>(amountUSDT);
            tradingTickLoop.getExecutor().execute(() -> {
                double balance = getAvailableBalance();
                if (balance <= 0) {
                    Vesta.error("Balance insuficiente o no detectado para operar en " + symbol);
                    return;
                }

                if (safeAmountUSDT.get() >= balance) {
                    safeAmountUSDT.set(balance * 0.98);
                } else {
                    safeAmountUSDT.set(safeAmountUSDT.get() * 0.99);
                }
                latch.countDown();
            });
            AtomicReference<Double> currentPrice = new AtomicReference<>(0d);
            tradingTickLoop.getExecutor().execute(() -> {
                currentPrice.set(binanceApi.getTickerPrice(symbol));
                latch.countDown();
            });
            latch.await();
            double quantity = (safeAmountUSDT.get() * leverage) / currentPrice.get();
            String qtyStr = binanceApi.formatQuantity(symbol, quantity);

            if (Double.parseDouble(qtyStr) <= 0) {
                Vesta.error("La cantidad calculada es 0. Revisa el balance o apalancamiento.");
                return null;
            }

            BinanceOpenOperation op = new BinanceOpenOperation(
                    this, currentPrice.get(), tpPercent, slPercent, direccion, amountUSDT, leverage
            );
            String colorGreen = "\u001B[32m";
            String colorRed = "\u001B[31m";
            String reset = "\u001B[0m";
            String displayDireccion = direccion == DireccionOperation.LONG ? colorGreen + direccion.name() + reset : colorRed + direccion.name() + reset;
            Vesta.info(String.format("🔓 Abriendo operacion %s, con un margen: %.3f$. " + colorRed + " TP %.2f%% (%.4f$) " + colorGreen + "SL %.2f%% (%.4f$)" + reset + " (%s)",
                    displayDireccion, amountUSDT, tpPercent, op.getTpPrice(), slPercent, op.getSlPrice(), op.getUuid()
            ));
            info("Abriendo operacion **%s** con un margen: %.3f$. **TP %.2f%%** (%.4f$) **SL %.2f%%** (%.4f$)",  direccion.name(), amountUSDT,
                    tpPercent, op.getTpPrice(),
                    slPercent, op.getSlPrice()
            );
            // obtener Precios
            double tpPrice = op.getTpPrice();
            double slPrice = op.getSlPrice();

            String side = (direccion == DireccionOperation.LONG) ? "BUY" : "SELL";
            long entryOrderId = binanceApi.placeOrder(symbol, side, "MARKET", qtyStr, null, false, false);
            op.setEntryBinanceId(entryOrderId);

            String closeSide = (side.equals("BUY")) ? "SELL" : "BUY";

            tradingTickLoop.getExecutor().execute(() ->{
                long slOrderId = binanceApi.placeAlgoOrder(symbol, closeSide, "STOP_MARKET", null, slPrice, true, true);
                op.setSlBinanceId(slOrderId);
                op.setSlIsAlgo(true);
            });

            tradingTickLoop.getExecutor().execute(() -> {
                long tpOrderId = binanceApi.placeAlgoOrder(symbol, closeSide, "TAKE_PROFIT_MARKET", null, tpPrice, true, true);
                op.setTpBinanceId(tpOrderId);
                op.setTpIsAlgo(true);
            });

            activeOperations.put(op.getUuid(), op);
            return op;
        } catch (InterruptedException e) {
            Vesta.sendErrorException("Binance Error Open Async", e);
            return null;
        }
    }

    @Override
    public void close(ExitReason reason, OpenOperation openOperation) {
        BinanceOpenOperation op = activeOperations.get(openOperation.getUuid());
        if (op == null) return;

        String symbol = getMarket().getSymbol();
        try {
            CountDownLatch latch = new CountDownLatch(2);
            Vesta.info("🔒 Cerrando operacion: %s por %s", op.getUuid(), reason);
            info("Cerrando operacion: %s por %s", op.getUuid(), reason);

            // 1. Cancelar SL y TP pendientes
            tradingTickLoop.getExecutor().execute(() -> {
                binanceApi.cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
                latch.countDown();
            });
            tradingTickLoop.getExecutor().execute(() -> {
                binanceApi.cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
                latch.countDown();
            });
            latch.await();
            // 2. Cerrar posición (Market opuesto)
            // Usamos closePosition=true para cerrar cualquier remanente con seguridad
            String closeSide = (op.getDireccion() == DireccionOperation.LONG) ? "SELL" : "BUY";

            double quantity = (op.getAmountInitUSDT() * op.getLeverage()) / op.getEntryPrice();
            String qtyStr = binanceApi.formatQuantity(symbol, quantity);

            tradingTickLoop.getExecutor().execute(() -> {
                binanceApi.placeOrder(symbol, closeSide, "MARKET", qtyStr, null, true, false);
            });

            // 3. Registrar cierre
            double exitPrice = binanceApi.getTickerPrice(symbol);
            CloseOperation closeOp = new BinanceCloseOperation(
                    exitPrice, System.currentTimeMillis(), op.getEntryTime(), reason, op
            );

            closedOperations.add(closeOp);
            activeOperations.remove(op.getUuid());
        } catch (InterruptedException e) {
            Vesta.sendErrorException("Binance Error Open Async", e);
        }
    }

    @Override
    public int closeSize() {
        return closedOperations.size();
    }

    @Override
    public int openSize() {
        return activeOperations.size();
    }

    @Override
    public @NotNull List<OpenOperation> getOpens() {
        return new ArrayList<>(activeOperations.values());
    }

    @Override
    public @NotNull List<CloseOperation> getCloses() {
        return new ArrayList<>(closedOperations);
    }

    @Override
    public Market getMarket() {
        return market;
    }

    @Override
    public double getAvailableBalance() {
        if (lastBalance == 0) {
            lastBalance = binanceApi.getBalance(market.getSymbol());
        }
        return lastBalance;
    }

    @Override
    public double getCurrentPrice() {
        return binanceApi.getTickerPrice(getMarket().getSymbol());
    }

    @Override
    public void log(String log){
        Vesta.info("📃 " + log);
    }

    public void updateOrdensSimple() {
        String symbol = getMarket().getSymbol();
        lastBalance = 0;
        try {
            if (!hasOpenPositionOnBinance(symbol)) {
                reconcileNoPositionOnBinance(symbol, "syncWithBinance");
            }
        } catch (Exception e) {
            Vesta.error("Binance sync error: " + e.getMessage());
        }
    }

    public void updateOrdens(String symbol) {
        lastBalance = 0;
        if (activeOperations.isEmpty()) {
            return;
        }

        try {
            if (!hasOpenPositionOnBinance(symbol)) {
                reconcileNoPositionOnBinance(symbol, "updateState");
                return;
            }
        } catch (Exception e) {
            Vesta.error("Error checking Binance position: " + e.getMessage());
            return;
        }

        List<Map.Entry<UUID, BinanceOpenOperation>> snapshot = new ArrayList<>(activeOperations.entrySet());
        for (Map.Entry<UUID, BinanceOpenOperation> entry : snapshot) {
            UUID uuid = entry.getKey();
            BinanceOpenOperation op = entry.getValue();
            if (!activeOperations.containsKey(uuid)) {
                continue;
            }

            try {
                if (binanceApi.checkOrderFilled(symbol, op.getSlBinanceId(), op.isSlIsAlgo())) {
                    Vesta.info("Binance: SL detectado ejecutado para " + op.getUuid());
                    binanceApi.cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
                    closeAndRemoveOperation(op,
                            op.getDireccion() == DireccionOperation.LONG ? ExitReason.LONG_STOP_LOSS : ExitReason.SHORT_STOP_LOSS,
                            "sl_filled");
                    continue;
                }

                if (binanceApi.checkOrderFilled(symbol, op.getTpBinanceId(), op.isTpIsAlgo())) {
                    Vesta.info("Binance: TP detectado ejecutado para " + op.getUuid());
                    binanceApi.cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
                    closeAndRemoveOperation(op,
                            op.getDireccion() == DireccionOperation.LONG ? ExitReason.LONG_TAKE_PROFIT : ExitReason.SHORT_TAKE_PROFIT,
                            "tp_filled");
                }
            } catch (Exception e) {
                Vesta.error("Error updating state: " + e.getMessage());
            }
        }
    }

    private boolean hasOpenPositionOnBinance(String symbol) {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        JsonNode positions = binanceApi.sendSignedRequest("GET", "/fapi/v2/positionRisk", params);

        if (positions == null || positions.isNull()) {
            return false;
        }

        if (positions.isArray()) {
            for (JsonNode position : positions) {
                if (!symbol.equalsIgnoreCase(position.path("symbol").asText())) {
                    continue;
                }
                if (Math.abs(position.path("positionAmt").asDouble(0.0)) > POSITION_EPSILON) {
                    return true;
                }
            }
            return false;
        }

        if (positions.isObject() && symbol.equalsIgnoreCase(positions.path("symbol").asText(symbol))) {
            return Math.abs(positions.path("positionAmt").asDouble(0.0)) > POSITION_EPSILON;
        }

        return false;
    }

    private void reconcileNoPositionOnBinance(String symbol, String source) {
        if (activeOperations.isEmpty()) {
            return;
        }

        Vesta.warning("No hay posición activa en Binance para " + symbol + ". Reconciliando estado local (" + source + ").");
        List<BinanceOpenOperation> snapshot = new ArrayList<>(activeOperations.values());
        for (BinanceOpenOperation op : snapshot) {
//            binanceApi.cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
//            binanceApi.cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
            closeAndRemoveOperation(op, ExitReason.STRATEGY, "binance_position_closed");
        }
    }

    private void closeAndRemoveOperation(BinanceOpenOperation op, ExitReason reason, String source) {
        if (!activeOperations.remove(op.getUuid(), op)) {
            return;
        }

//        double exitPrice = op.getEntryPrice();
//        try {
//            exitPrice = binanceApiRest.getTickerPrice(getMarket().getSymbol());
//        } catch (Exception e) {
//            Vesta.warning("No se pudo obtener precio de salida en reconciliación: " + e.getMessage());
//        }

        BinanceCloseOperation close = new BinanceCloseOperation(op, reason);
        closedOperations.add(close);

        if (tradingTickLoop != null && tradingTickLoop.getStrategy() != null) {
            tradingTickLoop.getStrategy().closeOperation(close, this);
        }
        Vesta.info("Operación reconciliada por Binance: " + op.getUuid() + " (" + source + ")");
    }

    @Getter
    @Setter
    public static class BinanceOpenOperation extends OpenOperation {
        private long entryBinanceId;          // ID de la orden de entrada (normal)
        private long tpBinanceId;             // ID de la orden de TP (puede ser normal o algo)
        private long slBinanceId;             // ID de la orden de SL (puede ser normal o algo)
        private boolean tpIsAlgo;             // true si TP es una orden algorítmica
        private boolean slIsAlgo;
        private final TradingBinance tradingBinance;

        public BinanceOpenOperation(TradingBinance binance, double entryPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            super(binance, entryPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
            this.tradingBinance = binance;
        }

        @Override
        public synchronized void setTpPercent(double tpPercent) {
            updateProtectionOrder(true, tpPercent, "TAKE_PROFIT_MARKET");
        }

        @Override
        public synchronized void setSlPercent(double slPercent) {
            updateProtectionOrder(false, slPercent, "STOP_MARKET");
        }

        private void updateProtectionOrder(boolean isTpOrder, double newPercent, @NotNull String orderType) {
            if (!Double.isFinite(newPercent)) {
                Vesta.warning("Valor inválido para %s en %s: %s",
                        isTpOrder ? "TP" : "SL", getUuid(), newPercent);
                return;
            }

            if (!tradingBinance.activeOperations.containsKey(getUuid())) {
                Vesta.warning("No se puede actualizar %s: operación %s no está activa",
                        isTpOrder ? "TP" : "SL", getUuid());
                return;
            }

            String symbol = tradingBinance.getMarket().getSymbol();
            String closeSide = (getDireccion() == DireccionOperation.LONG) ? "SELL" : "BUY";

            double oldPercent = isTpOrder ? getTpPercent() : getSlPercent();
            long oldOrderId = isTpOrder ? getTpBinanceId() : getSlBinanceId();
            boolean oldIsAlgo = isTpOrder ? isTpIsAlgo() : isSlIsAlgo();

            if (isTpOrder) {
                super.setTpPercent(newPercent);
            } else {
                super.setSlPercent(newPercent);
            }
            double triggerPrice = isTpOrder ? getTpPrice() : getSlPrice();

            try {
                tradingBinance.binanceApi.cancelOrder(symbol, oldOrderId, oldIsAlgo);
                long newOrderId = tradingBinance.binanceApi.placeAlgoOrder(
                        symbol, closeSide, orderType, null, triggerPrice, true, true
                );

                if (isTpOrder) {
                    setTpBinanceId(newOrderId);
                    setTpIsAlgo(true);
                } else {
                    setSlBinanceId(newOrderId);
                    setSlIsAlgo(true);
                }

                Vesta.info("Actualizado %s para %s a %.4f%% (trigger %.4f)",
                        isTpOrder ? "TP" : "SL", getUuid(), newPercent, triggerPrice);
            } catch (Exception e) {
                if (isTpOrder) {
                    super.setTpPercent(oldPercent);
                    setTpBinanceId(oldOrderId);
                    setTpIsAlgo(oldIsAlgo);
                } else {
                    super.setSlPercent(oldPercent);
                    setSlBinanceId(oldOrderId);
                    setSlIsAlgo(oldIsAlgo);
                }
                Vesta.sendErrorException("Binance Error Update " + (isTpOrder ? "TP" : "SL"), e);
            }
        }
    }

    public static class BinanceCloseOperation extends CloseOperation {

        public BinanceCloseOperation(@NotNull BinanceOpenOperation op, @NotNull ExitReason reason) {
            super((reason.toString().contains("STOP")) ? op.getSlPrice() : op.getTpPrice(),
                    System.currentTimeMillis(),
                    op.getEntryTime(),
                    reason,
                    op
            );
        }

        public BinanceCloseOperation(double exitPrice, long exitTime, long entryTime, ExitReason reason, OpenOperation openOperation) {
            super(exitPrice, exitTime, entryTime, reason, openOperation);
        }
    }

}
