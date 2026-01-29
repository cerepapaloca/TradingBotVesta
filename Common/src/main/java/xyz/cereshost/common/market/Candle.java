package xyz.cereshost.common.market;

public record Candle(
        long openTime,       // inicio del minuto
        double open,
        double high,
        double low,
        double close,

        float direccion,

        double amountTrades,

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


        double rsi4,
        double rsi8,
        double rsi16,

        double macdVal,
        double macdSignal,
        double macdHist,

        double nvi,

        double upperBand,
        double middleBand,
        double lowerBand,
        double bandwidth,
        double percentB
) {}