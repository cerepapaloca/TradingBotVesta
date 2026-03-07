package xyz.cereshost.vesta.core.exception;

public class BinanceApiRequestException extends RuntimeException {
    public BinanceApiRequestException(Exception e) {
        super(e);
    }
}
