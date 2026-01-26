package xyz.cereshost;

import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@UtilityClass
public class FinancialCalculation {

    public static double @NotNull [] computeRSI(@NotNull List<Double> price, int period) {
        int n = price.size();
        double[] rsi = new double[n];

        if (n <= period) {
            Arrays.fill(rsi, Double.NaN);
            return rsi;
        }

        double gain = 0.0;
        double loss = 0.0;

        for (int i = 1; i <= period; i++) {
            double diff = price.get(i) - price.get(i - 1);
            if (diff >= 0) gain += diff;
            else loss -= diff;
        }

        double avgGain = gain / period;
        double avgLoss = loss / period;

        for (int i = 0; i < period; i++) {
            rsi[i] = Double.NaN;
        }

        rsi[period] = avgLoss == 0
                ? 100.0
                : 100.0 - (100.0 / (1.0 + (avgGain / avgLoss)));

        for (int i = period + 1; i < n; i++) {
            double diff = price.get(i) - price.get(i - 1);
            double g = diff > 0 ? diff : 0;
            double l = diff < 0 ? -diff : 0;

            avgGain = (avgGain * (period - 1) + g) / period;
            avgLoss = (avgLoss * (period - 1) + l) / period;

            if (avgLoss == 0) {
                rsi[i] = 100.0;
            } else {
                double rs = avgGain / avgLoss;
                rsi[i] = 100.0 - (100.0 / (1.0 + rs));
            }
        }

        return rsi;
    }

    /**
     * Calcula la EMA (Exponential Moving Average) alineada con los precios.
     * Devuelve un array de la misma longitud que prices, con Double.NaN
     * para los índices previos al primer valor calculable.
     */
    public static double[] computeEMA(List<Double> prices, int period) {
        int n = prices.size();
        double[] ema = new double[n];
        Arrays.fill(ema, Double.NaN);
        if (n == 0 || period <= 0 || n < period) return ema;

        // alpha = 2 / (period + 1)
        double alpha = 2.0 / (period + 1.0);

        // primer EMA = SMA del primer "period" valores (en índice period-1)
        double sum = 0.0;
        for (int i = 0; i < period; i++) sum += prices.get(i);
        double prevEma = sum / period;
        ema[period - 1] = prevEma;

        // después, EMA recursiva
        for (int i = period; i < n; i++) {
            double price = prices.get(i);
            prevEma = alpha * (price - prevEma) + prevEma;
            ema[i] = prevEma;
        }
        return ema;
    }

    /**
     * Calcula MACD, Signal y Histogram.
     * shortPeriod: e.g. 12
     * longPeriod: e.g. 26
     * signalPeriod: e.g. 9
     *
     * Devuelve MACDResult con arrays alineados (Double.NaN donde no hay valor).
     */
    public static MACDResult computeMACD(List<Double> closes, int shortPeriod, int longPeriod, int signalPeriod) {
        int n = closes.size();
        double[] macd = new double[n];
        double[] signal = new double[n];
        double[] hist = new double[n];
        Arrays.fill(macd, Double.NaN);
        Arrays.fill(signal, Double.NaN);
        Arrays.fill(hist, Double.NaN);

        if (n == 0 || longPeriod <= 0 || shortPeriod <= 0 || longPeriod <= shortPeriod) {
            return new MACDResult(macd, signal, hist);
        }

        // EMAs
        double[] emaShort = computeEMA(closes, shortPeriod);
        double[] emaLong = computeEMA(closes, longPeriod);

        // MACD line = EMA_short - EMA_long (válido cuando ambos EMA no son NaN)
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(emaShort[i]) && !Double.isNaN(emaLong[i])) {
                macd[i] = emaShort[i] - emaLong[i];
            } else {
                macd[i] = Double.NaN;
            }
        }

        // Para calcular la signal line (EMA del MACD), necesitamos lista de MACD válidos.
        // Pero para mantener alineamiento, aplicamos computeEMA sobre la lista completa
        // transformando NaN a 0 hasta que haya valores; sin embargo la forma correcta:
        // crear lista de macdValid (sólo donde macd != NaN) y luego reinsertar alineado.
        List<Double> macdValid = new ArrayList<>();
        List<Integer> macdIdx = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (!Double.isNaN(macd[i])) {
                macdValid.add(macd[i]);
                macdIdx.add(i);
            }
        }

        if (macdValid.size() >= signalPeriod) {
            double[] signalCompact = computeEMA(macdValid, signalPeriod);
            // mapear de vuelta a array completo
            for (int k = 0; k < signalCompact.length; k++) {
                int idx = macdIdx.get(k);
                signal[idx] = signalCompact[k];
                if (!Double.isNaN(macd[idx]) && !Double.isNaN(signal[idx])) {
                    hist[idx] = macd[idx] - signal[idx];
                } else {
                    hist[idx] = Double.NaN;
                }
            }
        }
        // Si no hay suficientes valores para signal, quedan NaN.

        return new MACDResult(macd, signal, hist);
    }

    public record MACDResult(double[] macd, double[] signal, double[] histogram) {}

    public static double[] computeNVI(
            List<Double> closes,
            List<Double> volumes,
            double initialValue
    ) {
        int n = closes.size();
        double[] nvi = new double[n];
        Arrays.fill(nvi, Double.NaN);

        if (n == 0 || volumes.size() != n) return nvi;

        nvi[0] = initialValue; // típico: 1000

        for (int i = 1; i < n; i++) {
            double prevNvi = nvi[i - 1];
            double prevClose = closes.get(i - 1);
            double currClose = closes.get(i);
            double prevVol = volumes.get(i - 1);
            double currVol = volumes.get(i);

            if (prevClose == 0 || prevVol == 0) {
                nvi[i] = prevNvi;
                continue;
            }

            if (currVol < prevVol) {
                double priceChange = (currClose - prevClose) / prevClose;
                nvi[i] = prevNvi + priceChange * prevNvi;
            } else {
                nvi[i] = prevNvi;
            }
        }

        return nvi;
    }

    public static BollingerBandsResult computeBollingerBands(List<Double> closes, int period, double multiplier) {
        if (closes.size() < period) {
            throw new IllegalArgumentException(
                    String.format("Se necesitan al menos %d velas, solo hay %d", period, closes.size())
            );
        }

        int size = closes.size();
        double[] upperBand = new double[size];
        double[] middleBand = new double[size]; // SMA
        double[] lowerBand = new double[size];
        double[] bandwidth = new double[size]; // Ancho de banda en porcentaje
        double[] percentB = new double[size];  // %B indicator

        // Calcular para cada punto desde (period-1) hasta el final
        for (int i = period - 1; i < size; i++) {
            // Calcular SMA (media móvil simple)
            double sum = 0;
            for (int j = i - period + 1; j <= i; j++) {
                sum += closes.get(j);
            }
            double sma = sum / period;
            middleBand[i] = sma;

            // Calcular desviación estándar
            double sumSquaredDiff = 0;
            for (int j = i - period + 1; j <= i; j++) {
                double diff = closes.get(j) - sma;
                sumSquaredDiff += diff * diff;
            }
            double stdDev = Math.sqrt(sumSquaredDiff / period);

            // Calcular bandas
            upperBand[i] = sma + (stdDev * multiplier);
            lowerBand[i] = sma - (stdDev * multiplier);

            // Calcular métricas adicionales
            double currentPrice = closes.get(i);

            // Ancho de banda en porcentaje
            bandwidth[i] = ((upperBand[i] - lowerBand[i]) / sma) * 100;

            // %B indicator (posicion dentro de la banda)
            if (upperBand[i] != lowerBand[i]) {
                percentB[i] = (currentPrice - lowerBand[i]) / (upperBand[i] - lowerBand[i]);
            } else {
                percentB[i] = 0.5; // Si las bandas son iguales (raro)
            }
        }

        // Los primeros (period-1) elementos no tienen cálculo
        for (int i = 0; i < period - 1; i++) {
            upperBand[i] = Double.NaN;
            middleBand[i] = Double.NaN;
            lowerBand[i] = Double.NaN;
            bandwidth[i] = Double.NaN;
            percentB[i] = Double.NaN;
        }

        return new BollingerBandsResult(upperBand, middleBand, lowerBand, bandwidth, percentB);
    }

    public record BollingerBandsResult(double[] upperBand, double[] middleBand, double[] lowerBand, double[] bandwidth, double[] percentB) {};
}
