package xyz.cereshost;

import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.CandleSimple;

import java.util.*;

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

    /**
     * Calcula ATR usando el método Wilder (suavizado) para un periodo dado.
     * Retorna un array double[] con la ATR por índice (misma longitud que candles).
     * Para los primeros elementos donde no hay ATR suficiente se devuelve 0.
     */
    public static double[] computeATRWilder(List<CandleSimple> candles, int period) {
        int n = candles.size();
        double[] atr = new double[n];
        if (n == 0) return atr;

        // 1) calcular TRs
        double[] tr = new double[n];
        tr[0] = candles.get(0).high() - candles.get(0).low(); // primer TR como high-low
        for (int i = 1; i < n; i++) {
            CandleSimple cur = candles.get(i);
            CandleSimple prev = candles.get(i - 1);
            double highLow = cur.high() - cur.low();
            double highPrevClose = Math.abs(cur.high() - prev.close());
            double lowPrevClose = Math.abs(cur.low() - prev.close());
            tr[i] = Math.max(highLow, Math.max(highPrevClose, lowPrevClose));
        }

        // 2) ATR inicial: simple moving average de los primeros 'period' TRs
        if (n <= period) {
            // si no hay suficientes periodos, usar SMA de lo que haya
            double sum = 0;
            for (int i = 0; i < n; i++) sum += tr[i];
            double sma = (n > 0) ? sum / n : 0;
            for (int i = 0; i < n; i++) atr[i] = sma;
            return atr;
        }

        double sumFirst = 0;
        for (int i = 1; i <= period; i++) { // normalmente se empieza desde el índice 1
            sumFirst += tr[i];
        }
        double prevAtr = sumFirst / period;
        // asignar ATR para el índice period (alineación común)
        atr[period] = prevAtr;

        // 3) Wilder smoothing
        for (int i = period + 1; i < n; i++) {
            double curAtr = (prevAtr * (period - 1) + tr[i]) / period;
            atr[i] = curAtr;
            prevAtr = curAtr;
        }

        // Para índices < period puedes opcionalmente rellenar con la primera ATR calculada
        for (int i = 0; i < period; i++) {
            atr[i] = atr[period]; // o 0 si prefieres
        }

        return atr;
    }

    /**
     * Calcula rolling mean y std (Welford o ventana simple) sobre el volumen base.
     * Returns double[][] where [0] = means, [1] = stds (same length as candles).
     * For the first <window elements we fill with the first computed mean/std or 0.
     */
    public static double[][] computeRollingMeanStd(List<CandleSimple> candles, int window) {
        int n = candles.size();
        double[] means = new double[n];
        double[] stds = new double[n];
        if (n == 0) return new double[][]{means, stds};

        Deque<Double> windowVals = new ArrayDeque<>(window);
        double sum = 0;
        double sumSq = 0;

        for (int i = 0; i < n; i++) {
            double v = candles.get(i).volumen().baseVolume();
            windowVals.addLast(v);
            sum += v;
            sumSq += v * v;
            if (windowVals.size() > window) {
                double old = windowVals.removeFirst();
                sum -= old;
                sumSq -= old * old;
            }
            int k = windowVals.size();
            double mean = (k > 0) ? sum / k : 0;
            double variance = (k > 1) ? Math.max(0, (sumSq - (sum * sum) / k) / (k - 1)) : 0;
            double std = Math.sqrt(variance);
            means[i] = mean;
            stds[i] = std;
        }
        return new double[][]{means, stds};
    }

    public static Map<String, double[]> computeVolumeNormalizations(List<CandleSimple> candles, int window, double[] atrArray) {
        int n = candles.size();
        double[][] meanStd = computeRollingMeanStd(candles, window);
        double[] means = meanStd[0];
        double[] stds = meanStd[1];

        double[] ratio = new double[n];
        double[] zscore = new double[n];
        double[] perAtr = new double[n];

        for (int i = 0; i < n; i++) {
            double v = candles.get(i).volumen().baseVolume();
            double mean = means[i];
            double std = stds[i];
            // ratio to mean (avoid divide by zero)
            ratio[i] = (mean > 0) ? v / mean : 0.0;
            // z-score (if std 0 use 0), clip to [-3,3]
            double z = (std > 0) ? (v - mean) / std : 0.0;
            if (Double.isFinite(z)) {
                z = Math.max(-3.0, Math.min(3.0, z));
            } else {
                z = 0.0;
            }
            zscore[i] = z;
            // volume per ATR (ATR may be 0)
            double atr = (atrArray != null && i < atrArray.length) ? atrArray[i] : 0.0;
            perAtr[i] = (atr > 0) ? v / atr : 0.0;
        }

        Map<String, double[]> map = new HashMap<>();
        map.put("ratio", ratio);
        map.put("zscore", zscore);
        map.put("perAtr", perAtr);
        return map;
    }


}
