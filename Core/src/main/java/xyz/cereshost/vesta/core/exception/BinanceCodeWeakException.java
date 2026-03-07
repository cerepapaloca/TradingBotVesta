package xyz.cereshost.vesta.core.exception;

import com.fasterxml.jackson.databind.JsonNode;

public class BinanceCodeWeakException extends BinanceCodeException {
    public BinanceCodeWeakException(JsonNode rootNode, String method, String endpoint) {
        super(rootNode, method, endpoint);
    }
}
