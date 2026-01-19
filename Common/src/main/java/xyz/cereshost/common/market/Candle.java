package xyz.cereshost.common.market;

public record Candle(
        long openTime,       // inicio del minuto
        double open,
        double high,
        double low,
        double close,

        double volumeBase,
        double quoteVolume,
        double buyQuoteVolume,
        double sellQuoteVolume,

        double deltaUSDT,
        double buyRatio,

        double bidLiquidity,
        double askLiquidity,
        double depthImbalance,
        double midPrice,
        double spread,

        double rsi8,
        double resi16,
        double macdVal,
        double macdSignal,
        double macdHist
) {}