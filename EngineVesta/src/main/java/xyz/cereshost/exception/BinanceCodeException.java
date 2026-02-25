package xyz.cereshost.exception;

import com.fasterxml.jackson.databind.JsonNode;

import java.text.MessageFormat;

public class BinanceCodeException extends RuntimeException {
    public BinanceCodeException(JsonNode rootNode,String method, String endpoint) {
        super(MessageFormat.format("Error Binance ({0}) ({1}:{2}): {3}", rootNode.get("code"), method, endpoint, rootNode.get("msg").asText()));
    }
}
