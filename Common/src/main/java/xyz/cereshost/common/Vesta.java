package xyz.cereshost.common;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import xyz.cereshost.common.market.Market;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class Vesta {

    public static final ConcurrentHashMap<String, Market> MARKETS = new ConcurrentHashMap<>();
    private static final Logger LOGGER = LogManager.getLogger(Vesta.class);
    public static final List<String> MARKETS_NAMES = List.of("BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT");

    public static void info(String message, Object... o) {
        LOGGER.info(String.format(message, o));
    }
    public static void info(String message) {
        LOGGER.info(message);
    }

    public static void waring(String message, Object... o) {
        LOGGER.warn(String.format(message, o));
    }


    public static void waring(String message) {
        LOGGER.warn(message);
    }

    public static void error(String message, Object... o) {
        LOGGER.error(String.format(message, o));
    }

    public static void error(String message) {
        LOGGER.error(message);
    }
}
