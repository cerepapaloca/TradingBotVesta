package xyz.cereshost.vesta.core.exception;

public class BinanceApiSignedRequestException extends BinanceApiRequestException {
    private int code;
    public BinanceApiSignedRequestException(Exception e, int code) {
        super(e);
    }
}
