package xyz.cereshost.trading;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.TradingLoopBinance;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicReference;

public class TradingBinance implements Trading {

    private final String apiKey;
    private final String secretKey;
    private final String baseUrl; // https://fapi.binance.com
    private int lastLeverage = 1;
    private double lastBalance = 0;
    @Setter
    private TradingLoopBinance tradingLoopBinance;

    // Mapa para vincular tu UUID interno con los IDs de órdenes de Binance
    private final ConcurrentHashMap<UUID, BinanceOpenOperation> activeOperations = new ConcurrentHashMap<>();
    private final List<CloseOperation> closedOperations = Collections.synchronizedList(new ArrayList<>());

    private final HttpClient client = HttpClient.newHttpClient();

    @Setter
    private Market market;

    public TradingBinance(String apiKey, String secretKey, boolean isTestNet, Market market) {
        this.apiKey = apiKey;
        this.secretKey = secretKey;
        this.baseUrl = isTestNet ? "https://testnet.binancefuture.com" : "https://fapi.binance.com";
        this.market = market;
    }

    @Override
    public void open(double tpPercent, double slPercent, DireccionOperation direccion, double amountUSDT, int leverage) {
        String symbol = getMarket().getSymbol();
        try {
            CountDownLatch latch = new CountDownLatch(3);

            tradingLoopBinance.getExecutor().execute(() -> {
                if (lastLeverage != leverage) {
                    try {
                        changeLeverage(symbol, leverage);
                    } catch (Exception e) {
                        tradingLoopBinance.stop(e);
                    }
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
                try {
                    currentPrice.set(getTickerPrice(symbol));
                } catch (Exception e) {
                    tradingLoopBinance.stop(e);
                }
                latch.countDown();
            });
            latch.await();
            double quantity = (safeAmountUSDT.get() * leverage) / currentPrice.get();
            String qtyStr = formatQuantity(symbol, quantity);

            if (Double.parseDouble(qtyStr) <= 0) {
                Vesta.error("La cantidad calculada es 0. Revisa el balance o apalancamiento.");
                return;
            }

            String colorGreen = "\u001B[32m";
            String colorRed = "\u001B[31m";
            String reset = "\u001B[0m";
            String displayDireccion = direccion == DireccionOperation.LONG ? colorGreen + direccion.name() + reset : colorRed + direccion.name() + reset;
            Vesta.info("🔓 Abriendo %s, con un margen: %.3f$. " + colorRed + " TP %.2f" + colorRed + " SL %.2f", displayDireccion, amountUSDT, tpPercent, slPercent);

            BinanceOpenOperation op = new BinanceOpenOperation(
                    currentPrice.get(), tpPercent, slPercent, direccion, amountUSDT, leverage
            );
            op.setEntryTime(System.currentTimeMillis());

            // Enviar Orden de Entrada (MARKET)
            String side = (direccion == DireccionOperation.LONG) ? "BUY" : "SELL";
            long entryOrderId = placeOrder(symbol, side, "MARKET", qtyStr, null, false, false);
            op.setEntryBinanceId(entryOrderId);

            // Calcular Precios
            double tpPrice = op.getTpPrice();
            double slPrice = op.getSlPrice();

//            // Verificar dirección correcta de SL y TP
//            if (direccion == DireccionOperation.LONG) {
//                if (slPrice >= currentPrice.get()) {
//                    slPrice = currentPrice.get() * 0.99; // 1% por debajo
//                    Vesta.warning("Ajustando SL para LONG: " + slPrice);
//                }
//                if (tpPrice <= currentPrice.get()) {
//                    tpPrice = currentPrice.get() * 1.01; // 1% por encima
//                    Vesta.warning("Ajustando TP para LONG: " + tpPrice);
//                }
//            } else {
//                if (slPrice <= currentPrice.get()) {
//                    slPrice = currentPrice.get() * 1.01; // 1% por encima
//                    Vesta.warning("Ajustando SL para SHORT: " + slPrice);
//                }
//                if (tpPrice >= currentPrice.get()) {
//                    tpPrice = currentPrice.get() * 0.99; // 1% por debajo
//                    Vesta.warning("Ajustando TP para SHORT: " + tpPrice);
//                }
//            }

            String closeSide = (side.equals("BUY")) ? "SELL" : "BUY";

            tradingLoopBinance.getExecutor().execute(() ->{
                long slOrderId = 0;
                try {
                    slOrderId = placeOrder(symbol, closeSide, "STOP_MARKET", null, slPrice, true, true);
                } catch (Exception e) {
                    tradingLoopBinance.stop(e);
                }
                op.setSlBinanceId(slOrderId);
                op.setSlIsAlgo(true);
            });

            tradingLoopBinance.getExecutor().execute(() -> {
                long tpOrderId = 0;
                try {
                    tpOrderId = placeOrder(symbol, closeSide, "TAKE_PROFIT_MARKET", null, tpPrice, true, true);
                } catch (Exception e) {
                    tradingLoopBinance.stop(e);
                }
                op.setTpBinanceId(tpOrderId);
                op.setTpIsAlgo(true);
            });

            activeOperations.put(op.getUuid(), op);


        } catch (Exception e) {
            Vesta.error("Binance Error Open Async: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // --- Modificación en el método close ---
    @Override
    public void close(ExitReason reason, UUID uuidOpenOperation) {
        BinanceOpenOperation op = activeOperations.get(uuidOpenOperation);
        if (op == null) return;

        String symbol = getMarket().getSymbol();
        try {
            CountDownLatch latch = new CountDownLatch(2);
            Vesta.info("🔒 Cerrando operación " + uuidOpenOperation + " por " + reason);

            // 1. Cancelar SL y TP pendientes
            tradingLoopBinance.getExecutor().execute(() -> {
                cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
                latch.countDown();
            });
            tradingLoopBinance.getExecutor().execute(() -> {
                cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
                latch.countDown();
            });
            latch.await();
            // 2. Cerrar posición (Market opuesto)
            // Usamos closePosition=true para cerrar cualquier remanente con seguridad
            String closeSide = (op.getDireccion() == DireccionOperation.LONG) ? "SELL" : "BUY";

            double quantity = (op.getAmountInitUSDT() * op.getLeverage()) / op.getEntryPrice();
            String qtyStr = formatQuantity(symbol, quantity);

            tradingLoopBinance.getExecutor().execute(() -> {
                try {
                    placeOrder(symbol, closeSide, "MARKET", qtyStr, null, true, false);
                } catch (Exception e) {
                    tradingLoopBinance.stop(e);
                }
            });

            // 3. Registrar cierre
            double exitPrice = getTickerPrice(symbol);
            CloseOperation closeOp = new CloseOperationReal(
                    exitPrice, System.currentTimeMillis(), op.getEntryTime(), reason, op.getUuid()
            );

            closedOperations.add(closeOp);
            activeOperations.remove(uuidOpenOperation);

        } catch (Exception e) {
            Vesta.error("Binance Error Close: " + e.getMessage());
        }
    }

    private void recordClose(@NotNull BinanceOpenOperation op, @NotNull ExitReason reason) {
        // En un caso real, obtendríamos el precio real de ejecución de la orden
        double estimatedExit = (reason.toString().contains("STOP")) ? op.getSlPrice() : op.getTpPrice();

        CloseOperation closeOp = new CloseOperationReal(
                estimatedExit,
                System.currentTimeMillis(),
                op.getEntryTime(),
                reason,
                op.getUuid()
        );
        closedOperations.add(closeOp);
    }

    // --- Clases Internas extendiendo las de Trading.java ---

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

    public static class CloseOperationReal extends CloseOperation {
        public CloseOperationReal(double exitPrice, long exitTime, long entryTime, ExitReason reason, UUID uuidOpenOperation) {
            super(exitPrice, exitTime, entryTime, reason, uuidOpenOperation);
        }
    }

    // --- Métodos Privados de API Binance (REST) ---

    private long placeOrder(String symbol, String side, String type, String quantity, Double stopPrice, boolean reduceOnly, boolean closePosition) throws Exception {
        // Si es una orden condicional, usar el endpoint de órdenes algorítmicas
        if ("STOP_MARKET".equals(type) || "TAKE_PROFIT_MARKET".equals(type)) {
            return placeAlgoOrder(symbol, side, type, quantity, stopPrice, reduceOnly, closePosition);
        }

        // Para órdenes no condicionales (MARKET, LIMIT), seguir usando el endpoint tradicional
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("side", side);
        params.put("type", type);
        if (closePosition) {
            params.put("closePosition", "true");
        } else if (quantity != null) {
            params.put("quantity", quantity);
            if (reduceOnly) params.put("reduceOnly", "true");
        }
        if (stopPrice != null) params.put("stopPrice", formatPrice(symbol, stopPrice));

        Vesta.info("Enviando orden REST: " + type + " " + side +
                " Qty:" + (quantity != null ? quantity : "null") +
                " Stop:" + stopPrice +
                " reduceOnly:" + reduceOnly +
                " closePosition:" + closePosition);

        String response = sendSignedRequest("POST", "/fapi/v1/order", params);
        // ... (parseo igual que antes)
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(response);
        if (root.has("code") && root.get("code").asInt() != 0) {
            throw new RuntimeException("Error Binance (" + root.get("code") + "): " + root.get("msg").asText());
        }
        if (root.has("orderId")) {
            return root.get("orderId").asLong();
        } else {
            throw new RuntimeException("Respuesta desconocida: " + response);
        }
    }

    private long placeAlgoOrder(String symbol, String side, String type,
                                String quantity, Double stopPrice,
                                boolean reduceOnly, boolean closePosition) throws Exception {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("algoType", "CONDITIONAL");          // Obligatorio para órdenes condicionales
        params.put("symbol", symbol);
        params.put("side", side);
        params.put("type", type);                       // STOP_MARKET, TAKE_PROFIT_MARKET
        params.put("timeInForce", "GTC");               // Recomendado para condicionales

        // Si se quiere cerrar la posición completa, se usa closePosition y NO se envía quantity
        if (closePosition) {
            params.put("closePosition", "true");
        } else if (quantity != null) {
            params.put("quantity", quantity);
            if (reduceOnly) {
                params.put("reduceOnly", "true");
            }
        }

        // Precio de activación (obligatorio para STOP_MARKET/TAKE_PROFIT_MARKET)
        if (stopPrice != null) {
            params.put("triggerPrice", formatPrice(symbol, stopPrice));
        }

        // WorkingType (opcional, pero recomendado)
        params.put("workingType", "MARK_PRICE");

        // Usar el endpoint de órdenes algorítmicas
        String response = sendSignedRequest("POST", "/fapi/v1/algoOrder", params);

        // Parsear la respuesta
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root = mapper.readTree(response);
        if (root.has("code") && root.get("code").asInt() != 0) {
            throw new RuntimeException("Error Binance (" + root.get("code") + "): " + root.get("msg").asText());
        }
        if (root.has("algoId")) {
            return root.get("algoId").asLong();   // ¡Devuelve el algoId, no orderId!
        } else {
            throw new RuntimeException("Respuesta desconocida: " + response);
        }
    }

    @Override
    public void updateState(String symbol) {
        Iterator<Map.Entry<UUID, BinanceOpenOperation>> it = activeOperations.entrySet().iterator();
        while (it.hasNext()) {
            tradingLoopBinance.getExecutor().execute(() -> {
                Map.Entry<UUID, BinanceOpenOperation> entry = it.next();
                BinanceOpenOperation op = entry.getValue();

                try {
                    // Verificar estado del SL
                    if (checkOrderFilled(symbol, op.getSlBinanceId(), op.isSlIsAlgo())) {
                        Vesta.info("Binance: SL detectado ejecutado para " + op.getUuid());
                        // Cancelar el TP restante
                        cancelOrder(symbol, op.getTpBinanceId(), op.isTpIsAlgo());
                        // Registrar cierre
                        recordClose(op, op.getDireccion() == DireccionOperation.LONG ?
                                ExitReason.LONG_STOP_LOSS : ExitReason.SHORT_STOP_LOSS);
                        it.remove();
                        return;
                    }

                    // Verificar estado del TP
                    if (checkOrderFilled(symbol, op.getTpBinanceId(), op.isTpIsAlgo())) {
                        Vesta.info("Binance: TP detectado ejecutado para " + op.getUuid());
                        // Cancelar el SL restante
                        cancelOrder(symbol, op.getSlBinanceId(), op.isSlIsAlgo());
                        // Registrar cierre
                        recordClose(op, op.getDireccion() == DireccionOperation.LONG ?
                                ExitReason.LONG_TAKE_PROFIT : ExitReason.SHORT_TAKE_PROFIT);
                        it.remove();
                    }

                } catch (Exception e) {
                    Vesta.error("Error updating state: " + e.getMessage());
                }
            });
        }
    }

    private void cancelOrder(String symbol, long orderId, boolean isAlgoOrder) {
        try {
            if (orderId == 0) return;
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            if (isAlgoOrder) {
                params.put("algoId", String.valueOf(orderId));
                sendSignedRequest("DELETE", "/fapi/v1/algoOrder", params);
            } else {
                params.put("orderId", String.valueOf(orderId));
                sendSignedRequest("DELETE", "/fapi/v1/order", params);
            }
        } catch (Exception e) {
            Vesta.warning("No se pudo cancelar orden " + orderId + ": " + e.getMessage());  // CORREGIDO
        }
    }

    // NUEVO: Método para verificar el estado de cualquier orden (normal o algorítmica)
    private boolean checkOrderFilled(String symbol, long orderId, boolean isAlgoOrder) throws Exception {
        if (orderId == 0) return false;

        if (isAlgoOrder) {
            // Para órdenes algorítmicas
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            params.put("algoId", String.valueOf(orderId));

            String response = sendSignedRequest("GET", "/fapi/v1/algoOrder", params);
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response);

            // Verificar si hay error
            if (root.has("code") && root.get("code").asInt() != 0) {
                // Orden no encontrada, probablemente ya fue ejecutada
                return false;
            }

            // El estado puede ser "FILLED" o "FINISHED" para órdenes ejecutadas
            if (root.has("algoStatus")) {
                String status = root.get("algoStatus").asText();
                return "FILLED".equals(status) || "FINISHED".equals(status);
            }
            return false;
        } else {
            // Para órdenes normales
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            params.put("orderId", String.valueOf(orderId));

            String response = sendSignedRequest("GET", "/fapi/v1/order", params);
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response);

            // Verificar si hay error
            if (root.has("code") && root.get("code").asInt() != 0) {
                // Orden no encontrada, probablemente ya fue ejecutada
                return false;
            }

            if (root.has("status")) {
                return "FILLED".equals(root.get("status").asText());
            }
            return false;
        }
    }

    public void syncWithBinance() {
        String symbol = getMarket().getSymbol();
        lastBalance = 0;
        try {
            // 1. Obtener posiciones actuales
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            String response = sendSignedRequest("GET", "/fapi/v2/positionRisk", params);
            ObjectMapper mapper = new ObjectMapper();
            JsonNode positions = mapper.readTree(response);

            // Si no hay posición abierta pero tenemos operaciones activas, limpiar
            if (positions.isArray() && positions.size() == 0 && !activeOperations.isEmpty()) {
                Vesta.warning("No hay posición en Binance pero tenemos operaciones activas. Limpiando...");
                activeOperations.clear();
                return;
            }

            // 2. Verificar cada posición
            for (JsonNode position : positions) {
                String posSymbol = position.get("symbol").asText();
                if (symbol.equals(posSymbol)) {
                    double positionAmt = position.get("positionAmt").asDouble();
                    if (Math.abs(positionAmt) <= 0) {
                        // Posición cerrada en Binance
                        Vesta.warning("Posición cerrada en Binance. Limpiando operaciones locales...");
                        activeOperations.clear();
                    }
                }
            }

        } catch (Exception e) {
            Vesta.error("Error sincronizando con Binance: " + e.getMessage());
        }
    }

    private void closeAllExistingPositions(String symbol) {
        try {
            // 1. Obtener posiciones actuales
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            String response = sendSignedRequest("GET", "/fapi/v2/positionRisk", params);
            ObjectMapper mapper = new ObjectMapper();
            JsonNode positions = mapper.readTree(response);

            for (JsonNode position : positions) {
                String posSymbol = position.get("symbol").asText();
                if (symbol.equals(posSymbol)) {
                    double positionAmt = position.get("positionAmt").asDouble();
                    if (Math.abs(positionAmt) > 0) {
                        Vesta.warning("Cerrando posición existente: " + positionAmt + " " + symbol);

                        String side = positionAmt > 0 ? "SELL" : "BUY";
                        TreeMap<String, String> closeParams = new TreeMap<>();
                        closeParams.put("symbol", symbol);
                        closeParams.put("side", side);
                        closeParams.put("type", "MARKET");
                        closeParams.put("quantity", String.valueOf(Math.abs(positionAmt)));
                        closeParams.put("reduceOnly", "true");

                        sendSignedRequest("POST", "/fapi/v1/order", closeParams);
                    }
                }
            }

            // 2. Cancelar todas las órdenes abiertas
            TreeMap<String, String> cancelParams = new TreeMap<>();
            cancelParams.put("symbol", symbol);
            sendSignedRequest("DELETE", "/fapi/v1/allOpenOrders", cancelParams);

        } catch (Exception e) {
            Vesta.error("Error cerrando posiciones existentes: " + e.getMessage());
        }
    }


    private void changeLeverage(String symbol, int leverage) throws Exception {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("leverage", String.valueOf(leverage));
        sendSignedRequest("POST", "/fapi/v1/leverage", params);
    }

    private double getTickerPrice(String symbol) throws Exception {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        String response = sendRequest("GET", "/fapi/v1/ticker/price", params); // Public endpoint
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root;
        try {
            root = mapper.readTree(response);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        return root.get("price").asDouble();
    }

    // --- Utilidades HTTP y Firma ---

    private String sendSignedRequest(String method, String endpoint, TreeMap<String, String> params) throws Exception {
        params.put("timestamp", String.valueOf(System.currentTimeMillis()));
        params.put("recvWindow", "10000"); // Aumentamos margen de error de tiempo
        String queryString = buildQueryString(params);
        String signature = hmacSha256(queryString, secretKey);
        String finalUrl = baseUrl + endpoint + "?" + queryString + "&signature=" + signature;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(finalUrl))
                .header("X-MBX-APIKEY", apiKey)
                .method(method, HttpRequest.BodyPublishers.noBody())
                .build();

        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        return response.body();
    }

    private String sendRequest(String method, String endpoint, TreeMap<String, String> params) throws Exception {
        params.put("timestamp", String.valueOf(System.currentTimeMillis()));
        params.put("recvWindow", "10000");
        String queryString = buildQueryString(params);
        String finalUrl = baseUrl + endpoint + "?" + queryString;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(finalUrl))
                .method(method, HttpRequest.BodyPublishers.noBody())
                .build();
        return client.send(request, HttpResponse.BodyHandlers.ofString()).body();
    }

    private String buildQueryString(TreeMap<String, String> params) {
        StringJoiner sj = new StringJoiner("&");
        for (Map.Entry<String, String> entry : params.entrySet()) {
            sj.add(entry.getKey() + "=" + entry.getValue());
        }
        return sj.toString();
    }

    private String hmacSha256(String data, String secret) throws Exception {
        Mac sha256_HMAC = Mac.getInstance("HmacSHA256");
        SecretKeySpec secret_key = new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256");
        sha256_HMAC.init(secret_key);
        byte[] raw = sha256_HMAC.doFinal(data.getBytes(StandardCharsets.UTF_8));
        StringBuilder hex = new StringBuilder(2 * raw.length);
        for (byte b : raw) {
            hex.append(String.format("%02x", b));
        }
        return hex.toString();
    }

    // IMPORTANTE: Debes ajustar la precisión según el mercado (exchangeInfo)
    private String formatQuantity(String symbol, double qty) {
        if (symbol.startsWith("BTC")) return String.format(Locale.US, "%.3f", qty);
        if (symbol.startsWith("ETH")) return String.format(Locale.US, "%.2f", qty);
        return String.format(Locale.US, "%.0f", qty); // Default int
    }

    private String formatPrice(String symbol, double price) {
        if (symbol.startsWith("BTC")) return String.format(Locale.US, "%.1f", price);
        return String.format(Locale.US, "%.2f", price);
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
        try {
            // 1. Consultar cuenta (v3 devuelve el objeto con el campo 'assets')
            String response = sendSignedRequest("GET", "/fapi/v3/account", new TreeMap<>());
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response);

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

        } catch (Exception e) {
            Vesta.error("Error al obtener balance de Binance: " + e.getMessage());
        }
        return 0.0;
    }
}
