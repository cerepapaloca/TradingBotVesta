package xyz.cereshost.common;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import xyz.cereshost.common.market.Market;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class Vesta {

    public static final ConcurrentHashMap<String, Market> MARKETS = new ConcurrentHashMap<>();
    private static final Logger LOGGER = LogManager.getLogger(Vesta.class);
    public static final List<String> MARKETS_NAMES = List.of(
            "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT",
            "BTCUSDC", "ETHUSDC", "XRPUSDC", "BNBUSDC", "SOLUSDC", "DOGEUSDC"

    );

    public static void info(String message, Object... o) {
        LOGGER.info(String.format(message, o));
    }
    public static void info(String message) {
        LOGGER.info(message);
    }

    public static void warning(String message, Object... o) {
        LOGGER.warn(String.format(message, o));
    }


    public static void warning(String message) {
        LOGGER.warn(message);
    }

    public static void error(String message, Object... o) {
        LOGGER.error(String.format(message, o));
    }

    public static void error(String message) {
        LOGGER.error(message);
    }

    public static void sendErrorException(String message, Exception exception) {
        LOGGER.error(setFormatException(message, exception));
    }

    public static  void sendWaringException(String message, Exception exception) {
        LOGGER.warn(setFormatException(message, exception));
    }

    private static String setFormatException(String message, Exception exception) {
        StringBuilder builder = new StringBuilder();
        for (StackTraceElement element : exception.getStackTrace()) {
            builder.append(element.toString()).append("\n\t");
        }
        for (Throwable throwable : exception.getSuppressed()) {
            builder.append("[").append(throwable.getCause()).append("=").append(throwable.getMessage()).append("]").append("\n\t");
            for (StackTraceElement element : throwable.getStackTrace()) {
                builder.append(element.toString()).append("\n\t");
            }
        }
        return String.format("%s [%s=%s] \n\t%s", message, exception.getClass().getSimpleName(), exception.getMessage(), builder);
    }
}
