package xyz.cereshost.trading;

import com.fasterxml.jackson.databind.JsonNode;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.exception.BinanceCodeException;
import xyz.cereshost.message.Notifiable;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.util.Locale;
import java.util.Map;
import java.util.StringJoiner;
import java.util.TreeMap;
import java.util.function.Consumer;

public interface BinanceApi extends Notifiable {

    long placeAlgoOrder(String symbol,
                               String side,
                               String type,
                               String quantity,
                               Double stopPrice,
                               boolean reduceOnly,
                               boolean closePosition
    );


    long placeOrder(String symbol,
                    String side,
                    String type,
                    String quantity,
                    Double stopPrice,
                    boolean reduceOnly,
                    boolean closePosition
    );

    void cancelOrder(String symbol, long orderId, boolean isAlgoOrder);

    boolean checkOrderFilled(String symbol, long orderId, boolean isAlgoOrder);

    void closeAll(String symbol);

    void changeLeverage(String symbol, int leverage);

    double getTickerPrice(String symbol);

    JsonNode sendSignedRequest(String method, String endpoint, TreeMap<String, String> params);

    JsonNode sendRequest(String method, String endpoint, TreeMap<String, String> params);

    void checkRepose(JsonNode node, String method, String endpoint, String symbol) throws BinanceCodeException;

    void setExceptionHandler(Consumer<Exception> consumer);

    default String buildQueryString(@NotNull TreeMap<String, String> params) {
        StringJoiner sj = new StringJoiner("&");
        for (Map.Entry<String, String> entry : params.entrySet()) {
            sj.add(entry.getKey() + "=" + entry.getValue());
        }
        return sj.toString();
    }

    default String hmacSha256(@NotNull String data, @NotNull String secret) throws NoSuchAlgorithmException, InvalidKeyException {
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

    default String formatQuantity(@NotNull String symbol, double qty) {
        if (symbol.startsWith("BTC")) return String.format(Locale.US, "%.3f", qty);
        if (symbol.startsWith("ETH")) return String.format(Locale.US, "%.2f", qty);
        if (symbol.startsWith("XRP")) return String.format(Locale.US, "%.1f", qty);

        return String.format(Locale.US, "%.0f", qty); // Default int
    }

    default String formatPrice(@NotNull String symbol, double price) {
        if (symbol.startsWith("BTC")) return String.format(Locale.US, "%.1f", price);
        if (symbol.startsWith("XRP")) return String.format(Locale.US, "%.4f", price);

        return String.format(Locale.US, "%.2f", price);
    }

}
