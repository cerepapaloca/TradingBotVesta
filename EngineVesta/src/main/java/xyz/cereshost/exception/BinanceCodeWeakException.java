package xyz.cereshost.exception;

import com.fasterxml.jackson.databind.JsonNode;

import java.text.MessageFormat;

public class BinanceCodeWeakException extends BinanceCodeException {
    public BinanceCodeWeakException(JsonNode rootNode, String method, String endpoint) {
        super(rootNode, method, endpoint);
    }
}
