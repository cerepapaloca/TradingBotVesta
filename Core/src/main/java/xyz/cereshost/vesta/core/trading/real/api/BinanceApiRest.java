package xyz.cereshost.vesta.core.trading.real.api;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.vesta.common.Vesta;
import xyz.cereshost.vesta.core.exception.BinanceApiRequestException;
import xyz.cereshost.vesta.core.exception.BinanceApiSignedRequestException;
import xyz.cereshost.vesta.core.exception.BinanceCodeException;
import xyz.cereshost.vesta.core.exception.BinanceCodeWeakException;
import xyz.cereshost.vesta.core.io.IOdata;
import xyz.cereshost.vesta.core.message.MediaNotification;
import xyz.cereshost.vesta.core.trading.DireccionOperation;
import xyz.cereshost.vesta.core.trading.TimeInForce;
import xyz.cereshost.vesta.core.trading.TradingManager;
import xyz.cereshost.vesta.core.trading.TypeOrder;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.List;
import java.util.TreeMap;
import java.util.function.Consumer;

/**
 * ApiRest de Binance
 * <a href="https://developers.binance.com/docs/binance-spot-api-docs/rest-api/general-api-information">...</a>
 */

@Getter
@Setter
public final class BinanceApiRest implements BinanceApi {

    private final String apiKey;
    private final String secretKey;
    private final String baseUrl;
    // Errores "idempotentes" que no deben detener el loop (ej: cancelar una orden ya cerrada).
    private static final List<Integer> WEAK_ERROS_CODE = List.of(-2021, -2011);
    private final HttpClient client = HttpClient.newHttpClient();

    @NotNull private MediaNotification mediaNotification = MediaNotification.empty();
    @NotNull private Consumer<Exception> exceptionHandler = e -> {};

    public BinanceApiRest(boolean isTestNet) throws IOException {
        IOdata.ApiKeysBinance apiKeysBinance = IOdata.loadApiKeysBinance();
        this.apiKey = apiKeysBinance.key();
        this.secretKey = apiKeysBinance.secret();
        this.baseUrl = isTestNet ? "https://testnet.binancefuture.com" : "https://fapi.binance.com";
    }

    public BinanceApiRest(String apiKey, String secretKey, boolean isTestNet) {
        this.apiKey = apiKey;
        this.secretKey = secretKey;
        this.baseUrl = isTestNet ? "https://testnet.binancefuture.com" : "https://fapi.binance.com";
    }

    @Override
    public long placeOrder(String symbol,
                           DireccionOperation side,
                           TypeOrder type,
                           TimeInForce timeInForce,
                           String quantity,
                           Double stopPrice,
                           boolean reduceOnly,
                           boolean closePosition
    ) {
        // Para órdenes no condicionales (MARKET, LIMIT), seguir usando el endpoint tradicional
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("side", side.getSide());
        params.put("type", type.name());
        params.put("timeInForce", timeInForce.name());
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

        JsonNode root = sendSignedRequest("POST", "/fapi/v1/order", params);
        return root.get("orderId").asLong();
    }

    @Override
    public long placeAlgoOrder(String symbol,
                               DireccionOperation side,
                               TypeOrder type,
                               TimeInForce timeInForce,
                               String quantity,
                               Double stopPrice,
                               boolean reduceOnly,
                               boolean closePosition
    ) {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("algoType", "CONDITIONAL");          // Obligatorio para órdenes condicionales
        params.put("symbol", symbol);
        params.put("side", side.getSide());
        params.put("type", type.name());                       // STOP_MARKET, TAKE_PROFIT_MARKET
        params.put("timeInForce", timeInForce.name());         // Recomendado para condicionales

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
        JsonNode root = sendSignedRequest("POST", "/fapi/v1/algoOrder", params);
        return root.get("algoId").asLong();
    }

    @Override
    public void cancelOrder(String symbol, long orderId, boolean isAlgoOrder) {
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
            Vesta.warning("No se pudo cancelar orden " + orderId + ": " + e.getMessage());
        }
    }

    @Override
    public boolean checkOrderFilled(String symbol, long orderId, boolean isAlgoOrder) {
        if (orderId == 0) return false;

        if (isAlgoOrder) {
            // Para órdenes algorítmicas
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            params.put("algoId", String.valueOf(orderId));

            JsonNode root = sendSignedRequest("GET", "/fapi/v1/algoOrder", params);
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

            JsonNode root = sendSignedRequest("GET", "/fapi/v1/order", params);
            if (root.has("status")) {
                return "FILLED".equals(root.get("status").asText());
            }
            return false;
        }
    }

    @Override
    public void closeAll(String symbol) {
        try {
            // 1. Obtener posiciones actuales
            TreeMap<String, String> params = new TreeMap<>();
            params.put("symbol", symbol);
            JsonNode positions = sendSignedRequest("GET", "/fapi/v2/positionRisk", params);

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

    @Override
    public void changeLeverage(String symbol, int leverage) {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        params.put("leverage", String.valueOf(leverage));
        sendSignedRequest("POST", "/fapi/v1/leverage", params);
    }

    @Override
    public double getTickerPrice(String symbol) {
        TreeMap<String, String> params = new TreeMap<>();
        params.put("symbol", symbol);
        JsonNode root = sendRequest("GET", "/fapi/v1/ticker/price", params); // Public endpoint
        return root.get("price").asDouble();
    }

    @Override
    public double getBalance(String symbol) {
        // 1. Consultar cuenta (v3 devuelve el objeto con el campo 'assets')
        JsonNode root = sendSignedRequest("GET", "/fapi/v3/account", new TreeMap<>());
        // 2. Determinar qué moneda base estamos usando (USDT o USDC)
        // Si el símbolo es "BNBUSDC", buscamos "USDC". Si es "BNBUSDT", buscamos "USDT".
        String quoteAsset = symbol.endsWith("USDC") ? "USDC" : "USDT";

        // 3. Acceder al array de 'assets'
        if (root.has("assets") && root.get("assets").isArray()) {
            JsonNode assets = root.get("assets");

            for (JsonNode assetNode : assets) {
                String assetName = assetNode.get("asset").asText();

                if (quoteAsset.equalsIgnoreCase(assetName)) {
                    double balance = assetNode.get("availableBalance").asDouble();
                    Vesta.info("💰 Balance detectado para " + quoteAsset + ": " + balance);
                    return balance;
                }
            }
        }

        // 4. Backup: Si por alguna razón no se encuentra en el array,
        // intentar tomar el availableBalance general del root
        if (root.has("availableBalance")) {
            return root.get("availableBalance").asDouble();
        }
        return 0.0;
    }

    @Override
    public JsonNode sendSignedRequest(String method, String endpoint, TreeMap<String, String> params) throws BinanceApiSignedRequestException {
        params.put("timestamp", String.valueOf(System.currentTimeMillis()));
        params.put("recvWindow", "20000");
        try {
            String queryString = buildQueryString(params);
            String signature = hmacSha256(queryString, secretKey);
            String finalUrl = baseUrl + endpoint + "?" + queryString + "&signature=" + signature;

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(finalUrl))
                    .header("X-MBX-APIKEY", apiKey)
                    .method(method, HttpRequest.BodyPublishers.noBody())
                    .build();

            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            // Parsear la respuesta
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(response.body());
            checkRepose(root, method, endpoint, params.get("symbol"));
            return root;
        }catch (Exception e) {
            exceptionHandler.accept(e);
            if (e instanceof BinanceCodeException binanceCodeException) {
                throw new BinanceApiSignedRequestException(e, binanceCodeException.getCode());
            }else {
                throw new BinanceApiSignedRequestException(e, -1);
            }
        }
    }

    @Override
    public JsonNode sendRequest(String method, String endpoint, TreeMap<String, String> params) throws BinanceApiRequestException {
        try {
            params.put("timestamp", String.valueOf(System.currentTimeMillis()));
            params.put("recvWindow", "20000");
            String queryString = buildQueryString(params);
            String finalUrl = baseUrl + endpoint + "?" + queryString;

            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(finalUrl))
                    .method(method, HttpRequest.BodyPublishers.noBody())
                    .build();
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(client.send(request, HttpResponse.BodyHandlers.ofString()).body());
            checkRepose(root, method, endpoint, params.get("symbol"));
            return root;
        } catch (Exception e) {
            exceptionHandler.accept(e);
            throw new BinanceApiRequestException(e);
        }
    }

    @Override
    public void checkRepose(@NotNull JsonNode node, @NotNull String method, @NotNull String endpoint, @Nullable String symbol) throws BinanceCodeException {
        if (node.has("code") && node.get("code").asInt() != 0) {
            int code = node.get("code").asInt();
            if (code == 200) {
                Vesta.info("✅ operacion Ok (%s:%s)", method, endpoint);
            }else {
                if (WEAK_ERROS_CODE.contains(code)) {
                    if (code == -2021 && symbol != null) closeAll(symbol);
                    BinanceCodeWeakException e = new BinanceCodeWeakException(node, method, endpoint);
                    mediaNotification.waring("Error al hacer una petición: **%s**. Revisa la consola para más información", e.getMessage());
                    Vesta.sendWaringException("Error al hacer una petición" , e);
                    throw e;
                }else {
                    if (symbol != null) closeAll(symbol);
                    BinanceCodeException exception = new BinanceCodeException(node, method, endpoint);
                    mediaNotification.error("Error al hacer una petición: **%s**. Revisa la consola para más información", exception.getMessage());
                    exceptionHandler.accept(exception);
                    throw exception;
                }
            }
        }
    }
}
