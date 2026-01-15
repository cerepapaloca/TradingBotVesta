package xyz.cereshost;

import com.google.gson.Gson;
import xyz.cereshost.market.Market;

import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

public class Utils {

    public static final Gson GSON = new Gson();
    public static final ConcurrentHashMap<String, Market> MARKETS = new ConcurrentHashMap<>();
    public static final List<String> MARKETS_NAMES = List.of("BTCUSDT", "ETHUSDT", "XRPUSDT");
}
