package xyz.cereshost.vesta.core.exception;

import com.fasterxml.jackson.databind.JsonNode;
import lombok.Getter;

import java.text.MessageFormat;

@Getter
public class BinanceCodeException extends RuntimeException {
    private final int code;

    public BinanceCodeException(JsonNode rootNode, String method, String endpoint) {
        super(MessageFormat.format("Error Binance ({0}) ({1}:{2}): {3}", rootNode.get("code"), method, endpoint, rootNode.get("msg").asText()));
        code = rootNode.get("code").asInt();
    }
}
