package xyz.cereshost.vesta.common.market;

public record Volumen(double quoteVolume,
                      double baseVolume,
                      double takerBuyQuoteVolume,
                      double sellQuoteVolume,
                      double deltaUSDT,
                      double buyRatio
) {

}
