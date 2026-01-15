package xyz.cereshost.common.market;

public record Volumen(double quoteVolume,
                      double takerBuyQuoteVolume,
                      double sellQuoteVolume,
                      double deltaUSDT,
                      double buyRatio
) {

}
