package xyz.cereshost.vesta.common.market;

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

        double volRatioToMean,
        double volZscore,
        double volPerAtr,

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
        double percentB,

        double atr14,

        float superTrendSlow,
        float superTrendMedium,
        float superTrendFast,
        float emaSlow,
        float emaFast
) {
    public double getDiffPercent(){
        return ((close - open)/open)*100;
    }

    public double getDiffPercentAbs(){
        return Math.abs(getDiffPercent());
    }

    public boolean isStrongCandle() {
        double range = high - low;
        if (range == 0) return false;

        double body = Math.abs(close - open);

        double upperWick = high - Math.max(open, close);
        double lowerWick = Math.min(open, close) - low;

        double bodyRatio = body / range;

        // cuerpo debe ser grande
        if (bodyRatio < 0.6) return false;

        // sombras pequeñas
        if (upperWick / range > 0.25) return false;
        if (lowerWick / range > 0.25) return false;

        return true;
    }

    public boolean isWeakCandle() {

        double range = high - low;
        if (range == 0) return true;

        double body = Math.abs(close - open);
        double bodyRatio = body / range;

        return bodyRatio < 0.3;
    }

    public boolean isBullish(){
        return close > open;
    }

    public double highBody(){
        return Math.max(open, close);
    }
    public double lowBody(){
        return Math.min(open, close);
    }
}
