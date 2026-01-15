package xyz.cereshost.market;

public record Volumen(double quoteVolume,
                      double takerBuyQuoteVolume,
                      double sellQuoteVolume,
                      double deltaUSDT,
                      double buyRatio
) {

}
