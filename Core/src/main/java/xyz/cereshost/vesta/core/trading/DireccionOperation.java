package xyz.cereshost.vesta.core.trading;

import lombok.Getter;

/**
 * Direcciones de operación
 */
@Getter
public enum DireccionOperation {
    /**
     * Operación Corta/Vender
     */
    SHORT("SELL"),
    /**
     * Operación Larga/Comprar
     */
    LONG("BUY"),
    /**
     * En caso qué sea lateral
     * <strong>No se puede operar con neutral solo es una forma de identificar operacion sin movimiento ósea no hacer nada</strong>
     */
    NEUTRAL("invalid");

    private final String side;
    DireccionOperation(String side) {
        this.side = side;
    }

    public DireccionOperation inverse(){
        return switch (this){
            case LONG -> SHORT;
            case SHORT -> LONG;
            default -> NEUTRAL;
        };
    }
}
