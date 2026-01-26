package xyz.cereshost.trading;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jfree.data.json.impl.JSONObject;
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

public class TradingBinance implements Trading {

    private final String apiKey;
    private final String secretKey;
    private final String baseUrl; // https://fapi.binance.com

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
            // 1. Configurar Apalancamiento en Binance
            changeLeverage(symbol, leverage);

            // 2. Obtener precio actual para calcular cantidad (Binance requiere cantidad en Moneda Base, ej BTC)
            double currentPrice = getTickerPrice(symbol);

            // Cantidad = (USDT * Leverage) / Precio
            double quantity = (amountUSDT * leverage) / currentPrice;
            String qtyStr = formatQuantity(symbol, quantity);

            Vesta.info("Binance: Abriendo " + direccion + " en " + symbol + " Qty: " + qtyStr);

            // 3. Crear Objeto Local (Hereda de tu OpenOperation)
            BinanceOpenOperation op = new BinanceOpenOperation(
                    currentPrice, tpPercent, slPercent, direccion, amountUSDT, leverage
            );
            op.setEntryTime(System.currentTimeMillis());

            // 4. Enviar Orden de Mercado (ENTRY)
            String side = (direccion == DireccionOperation.LONG) ? "BUY" : "SELL";
            long entryOrderId = placeOrder(symbol, side, "MARKET", qtyStr, null, false);
            op.setEntryBinanceId(entryOrderId);

            // 5. Calcular Precios TP/SL basados en el precio de entrada estimado
            double tpPrice = op.getTpPrice();
            double slPrice = op.getSlPrice();

            // 6. Enviar Stop Loss (STOP_MARKET - Reduce Only)
            String closeSide = (side.equals("BUY")) ? "SELL" : "BUY";
            long slOrderId = placeOrder(symbol, closeSide, "STOP_MARKET", qtyStr, slPrice, true);
            op.setSlBinanceId(slOrderId);

            // 7. Enviar Take Profit (TAKE_PROFIT_MARKET - Reduce Only)
            long tpOrderId = placeOrder(symbol, closeSide, "TAKE_PROFIT_MARKET", qtyStr, tpPrice, true);
            op.setTpBinanceId(tpOrderId);

            // Guardar en memoria
            activeOperations.put(op.getUuid(), op);
            Vesta.info("Binance: Operación abierta exitosamente. UUID: " + op.getUuid());

        } catch (Exception e) {
            Vesta.error("Binance Error Open: " + e.getMessage());
            e.printStackTrace();
        }
    }

    @Override
    public void close(ExitReason reason, UUID uuidOpenOperation) {
        BinanceOpenOperation op = activeOperations.get(uuidOpenOperation);
        if (op == null) {
            Vesta.error("Intento de cerrar operación no existente: " + uuidOpenOperation);
            return;
        }

        String symbol = getMarket().getSymbol();
        try {
            Vesta.info("Binance: Cerrando operación " + uuidOpenOperation + " por " + reason);

            // 1. Cancelar órdenes pendientes (SL y TP) para que no queden huerfanas
            cancelOrder(symbol, op.getSlBinanceId());
            cancelOrder(symbol, op.getTpBinanceId());

            // 2. Cerrar la posición inmediatamente (Market Order opuesta)
            // Calculamos la cantidad igual que en la entrada (o consultamos posición real)
            double quantity = (op.getAmountInitUSDT() * op.getLeverage()) / op.getEntryPrice();
            String qtyStr = formatQuantity(symbol, quantity);

            String closeSide = (op.getDireccion() == DireccionOperation.LONG) ? "SELL" : "BUY";

            // Ejecutar cierre
            placeOrder(symbol, closeSide, "MARKET", qtyStr, null, false);

            // 3. Mover a lista de cerrados
            double exitPrice = getTickerPrice(symbol); // Precio aproximado de salida
            CloseOperation closeOp = new CloseOperationReal(
                    exitPrice,
                    System.currentTimeMillis(),
                    op.getEntryTime(),
                    reason,
                    op.getUuid()
            );

            closedOperations.add(closeOp);
            activeOperations.remove(uuidOpenOperation);

        } catch (Exception e) {
            Vesta.error("Binance Error Close: " + e.getMessage());
        }
    }

    @Override
    public void updateState(String symbol) {
        // Lógica de sincronización:
        // Consultar si las órdenes de TP o SL fueron llenadas ("FILLED") en Binance

        Iterator<Map.Entry<UUID, BinanceOpenOperation>> it = activeOperations.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<UUID, BinanceOpenOperation> entry = it.next();
            BinanceOpenOperation op = entry.getValue();

            try {
                // Verificar estado del SL
                if (checkOrderFilled(symbol, op.getSlBinanceId())) {
                    Vesta.info("Binance: SL detectado ejecutado para " + op.getUuid());
                    // Cancelar el TP restante
                    cancelOrder(symbol, op.getTpBinanceId());
                    // Registrar cierre
                    recordClose(op, ExitReason.SHORT_STOP_LOSS); // O LONG_STOP_LOSS dinámico
                    it.remove();
                    continue;
                }

                // Verificar estado del TP
                if (checkOrderFilled(symbol, op.getTpBinanceId())) {
                    Vesta.info("Binance: TP detectado ejecutado para " + op.getUuid());
                    // Cancelar el SL restante
                    cancelOrder(symbol, op.getSlBinanceId());
                    // Registrar cierre
                    recordClose(op, ExitReason.LONG_TAKE_PROFIT); // O SHORT dinámico
                    it.remove();
                }

            } catch (Exception e) {
                Vesta.error("Error updating state: " + e.getMessage());
            }
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
        private long entryBinanceId;
        private long tpBinanceId;
        private long slBinanceId;

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

    private long placeOrder(String symbol, String side, String type, String quantity, Double stopPrice, boolean reduceOnly) throws Exception {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("side", side);
        params.put("type", type);
        params.put("quantity", quantity);
        if (stopPrice != null) params.put("stopPrice", formatPrice(symbol, stopPrice));
        if (reduceOnly) params.put("reduceOnly", "true");
        params.put("timestamp", String.valueOf(System.currentTimeMillis()));

        String response = sendSignedRequest("POST", "/fapi/v1/order", params);
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root;
        try {
            root = mapper.readTree(response);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        if (root.has("orderId")) {
            return root.get("orderId").asLong();
        } else {
            throw new RuntimeException("Error colocando orden: " + response);
        }
    }

    private void cancelOrder(String symbol, long orderId) {
        try {
            if (orderId == 0) return;
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            params.put("orderId", String.valueOf(orderId));
            params.put("timestamp", String.valueOf(System.currentTimeMillis()));
            sendSignedRequest("DELETE", "/fapi/v1/order", params);
        } catch (Exception e) {
            // Ignoramos si la orden ya no existe (ej. ya se llenó)
            Vesta.waring("No se pudo cancelar orden " + orderId + ": " + e.getMessage());
        }
    }

    private boolean checkOrderFilled(String symbol, long orderId) throws Exception {
        if (orderId == 0) return false;
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("orderId", String.valueOf(orderId));
        params.put("timestamp", String.valueOf(System.currentTimeMillis()));

        String response = sendSignedRequest("GET", "/fapi/v1/order", params);
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root;
        try {
            root = mapper.readTree(response);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        return "FILLED".equals(root.get("status").asText());
    }

    private void changeLeverage(String symbol, int leverage) throws Exception {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("leverage", String.valueOf(leverage));
        params.put("timestamp", String.valueOf(System.currentTimeMillis()));
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
        try {
            // Consultar balances de la cuenta de futuros
            String response = sendSignedRequest("GET", "/fapi/v2/balance", new TreeMap<>());
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response);

            if (root.isArray()) {
                for (JsonNode assetNode : root) {
                    // Buscamos el activo USDT
                    if ("USDT".equalsIgnoreCase(assetNode.get("asset").asText())) {
                        return assetNode.get("withdrawAvailable").asDouble();
                    }
                }
            }
        } catch (Exception e) {
            Vesta.error("Error al obtener balance de Binance: " + e.getMessage());
        }
        return 0.0;
    }
}
