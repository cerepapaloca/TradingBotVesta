package xyz.cereshost.trading;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.message.MediaNotification;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

public class TradingBinance implements Trading {

    private static final double POSITION_EPSILON = 1.0e-8;
    private final BinanceApi binanceApiRest;
    private int lastLeverage = 1;
    private double lastBalance = 0;
    @Setter
    private TradingLoopBinance tradingLoopBinance;
    @Getter @Setter @NotNull
    private MediaNotification mediaNotification;

    // Mapa para vincular tu UUID interno con los IDs de órdenes de Binance
    private final ConcurrentHashMap<UUID, BinanceOpenOperation> activeOperations = new ConcurrentHashMap<>();
    private final List<CloseOperation> closedOperations = Collections.synchronizedList(new ArrayList<>());
    private final Market market;

    public TradingBinance(@NotNull BinanceApi binanceApiRest, @Nullable MediaNotification mediaNotification, @NotNull Market market) {
        this.binanceApiRest = binanceApiRest;
        this.market = market;
        this.mediaNotification = Objects.requireNonNullElse(mediaNotification, MediaNotification.empty());
    }

    @Override
    public void open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
        String symbol = getMarket().getSymbol();
        try {
            CountDownLatch latch = new CountDownLatch(3);

            tradingLoopBinance.getExecutor().execute(() -> {
                if (lastLeverage != leverage) {
                    binanceApiRest.changeLeverage(symbol, leverage);
                }
                lastLeverage = leverage;
                latch.countDown();
            });
            AtomicReference<Double> safeAmountUSDT = new AtomicReference<>(amountUSDT);
            tradingLoopBinance.getExecutor().execute(() -> {
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
            tradingLoopBinance.getExecutor().execute(() -> {
                currentPrice.set(binanceApiRest.getTickerPrice(symbol));
                latch.countDown();
            });
            latch.await();
            double quantity = (safeAmountUSDT.get() * leverage) / currentPrice.get();
            String qtyStr = binanceApiRest.formatQuantity(symbol, quantity);

            if (Double.parseDouble(qtyStr) <= 0) {
                Vesta.error("La cantidad calculada es 0. Revisa el balance o apalancamiento.");
                return;
            }

            BinanceOpenOperation op = new BinanceOpenOperation(
                    currentPrice.get(), tpPercent, slPercent, direccion, amountUSDT, leverage
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
            long entryOrderId = binanceApiRest.placeOrder(symbol, side, "MARKET", qtyStr, null, false, false);
            op.setEntryBinanceId(entryOrderId);

            String closeSide = (side.equals("BUY")) ? "SELL" : "BUY";

            tradingLoopBinance.getExecutor().execute(() ->{
                long slOrderId = binanceApiRest.placeAlgoOrder(symbol, closeSide, "STOP_MARKET", null, slPrice, true, true);
                op.setSlBinanceId(slOrderId);
                op.setSlIsAlgo(true);
            });

            tradingLoopBinance.getExecutor().execute(() -> {
                long tpOrderId = binanceApiRest.placeAlgoOrder(symbol, closeSide, "TAKE_PROFIT_MARKET", null, tpPrice, true, true);
                op.setTpBinanceId(tpOrderId);
                op.setTpIsAlgo(true);
            });

            activeOperations.put(op.getUuid(), op);
        } catch (InterruptedException e) {
            Vesta.sendErrorException("Binance Error Open Async", e);
        }
    }

    @Override
    public void close(ExitReason reason, UUID uuidOpenOperation) {
        BinanceOpenOperation op = activeOperations.get(uuidOpenOperation);
        if (op == null) return;

        String symbol = getMarket().getSymbol();
        try {
            CountDownLatch latch = new CountDownLatch(2);
            Vesta.info("🔒 Cerrando operacion: %s por %s", uuidOpenOperation, reason);
            info("Cerrando operacion: %s por %s", uuidOpenOperation, reason);

            // 1. Cancelar SL y TP pendientes
            tradingLoopBinance.getExecutor().execute(() -> {
                binanceApiRest.cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
                latch.countDown();
            });
            tradingLoopBinance.getExecutor().execute(() -> {
                binanceApiRest.cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
                latch.countDown();
            });
            latch.await();
            // 2. Cerrar posición (Market opuesto)
            // Usamos closePosition=true para cerrar cualquier remanente con seguridad
            String closeSide = (op.getDireccion() == DireccionOperation.LONG) ? "SELL" : "BUY";

            double quantity = (op.getAmountInitUSDT() * op.getLeverage()) / op.getEntryPrice();
            String qtyStr = binanceApiRest.formatQuantity(symbol, quantity);

            tradingLoopBinance.getExecutor().execute(() -> {
                binanceApiRest.placeOrder(symbol, closeSide, "MARKET", qtyStr, null, true, false);
            });

            // 3. Registrar cierre
            double exitPrice = binanceApiRest.getTickerPrice(symbol);
            CloseOperation closeOp = new BinanceCloseOperation(
                    exitPrice, System.currentTimeMillis(), op.getEntryTime(), reason, op.getUuid()
            );

            closedOperations.add(closeOp);
            activeOperations.remove(uuidOpenOperation);
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
        if (lastBalance != 0){
            return lastBalance;
        }
        // 1. Consultar cuenta (v3 devuelve el objeto con el campo 'assets')
        JsonNode root = binanceApiRest.sendSignedRequest("GET", "/fapi/v3/account", new TreeMap<>());
        // 2. Determinar qué moneda base estamos usando (USDT o USDC)
        // Si el símbolo es "BNBUSDC", buscamos "USDC". Si es "BNBUSDT", buscamos "USDT".
        String symbol = getMarket().getSymbol();
        String quoteAsset = symbol.endsWith("USDC") ? "USDC" : "USDT";

        // 3. Acceder al array de 'assets'
        if (root.has("assets") && root.get("assets").isArray()) {
            JsonNode assets = root.get("assets");

            for (JsonNode assetNode : assets) {
                String assetName = assetNode.get("asset").asText();

                if (quoteAsset.equalsIgnoreCase(assetName)) {
                    double balance = assetNode.get("availableBalance").asDouble();
                    Vesta.info("💰 Balance detectado para " + quoteAsset + ": " + balance);
                    lastBalance = balance;
                    return balance;
                }
            }
        }

        // 4. Backup: Si por alguna razón no se encuentra en el array,
        // intentar tomar el availableBalance general del root
        if (root.has("availableBalance")) {
            double balance = root.get("availableBalance").asDouble();
            lastBalance = balance;
            return balance;
        }
        return 0.0;
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
                if (binanceApiRest.checkOrderFilled(symbol, op.getSlBinanceId(), op.isSlIsAlgo())) {
                    Vesta.info("Binance: SL detectado ejecutado para " + op.getUuid());
                    binanceApiRest.cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
                    closeAndRemoveOperation(op,
                            op.getDireccion() == DireccionOperation.LONG ? ExitReason.LONG_STOP_LOSS : ExitReason.SHORT_STOP_LOSS,
                            "sl_filled");
                    continue;
                }

                if (binanceApiRest.checkOrderFilled(symbol, op.getTpBinanceId(), op.isTpIsAlgo())) {
                    Vesta.info("Binance: TP detectado ejecutado para " + op.getUuid());
                    binanceApiRest.cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
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
        JsonNode positions = binanceApiRest.sendSignedRequest("GET", "/fapi/v2/positionRisk", params);

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
            binanceApiRest.cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
            binanceApiRest.cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
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

        if (tradingLoopBinance != null && tradingLoopBinance.getStrategy() != null) {
            tradingLoopBinance.getStrategy().closeOperation(close);
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

        public BinanceOpenOperation(double entryPrice, double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
            super(entryPrice, tpPercent, slPercent, direccion, amountUSDT, leverage);
        }
    }

    public static class BinanceCloseOperation extends CloseOperation {

        public BinanceCloseOperation(@NotNull BinanceOpenOperation op, @NotNull ExitReason reason) {
            super((reason.toString().contains("STOP")) ? op.getSlPrice() : op.getTpPrice(),
                    System.currentTimeMillis(),
                    op.getEntryTime(),
                    reason,
                    op.getUuid()
            );
        }

        public BinanceCloseOperation(double exitPrice, long exitTime, long entryTime, ExitReason reason, UUID uuidOpenOperation) {
            super(exitPrice, exitTime, entryTime, reason, uuidOpenOperation);
        }
    }

}
