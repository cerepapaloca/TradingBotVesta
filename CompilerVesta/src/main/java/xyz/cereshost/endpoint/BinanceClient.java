package xyz.cereshost.endpoint;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import lombok.Getter;
import lombok.experimental.UtilityClass;
import xyz.cereshost.market.Depth;
import xyz.cereshost.market.Trade;
import xyz.cereshost.market.Volumen;

import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

@UtilityClass
public class BinanceClient {

    public static final String BASE_URL = "https://api.binance.com/api/v3/";
    public static final int LIMIT_DEPTH = 40;

    public Depth getDepth(String symbol) {
        String raw = getRequest(BASE_URL + "depth" + "?symbol=" + symbol + "&limit=" + LIMIT_DEPTH);
        Gson gson = new Gson();
        OrderBookRaw orderBookRaw = gson.fromJson(raw, OrderBookRaw.class);
        return new Depth(System.currentTimeMillis(),
                orderBookRaw.getBids().stream().map(list ->
                        new Depth.OrderLevel(Double.parseDouble(list.getFirst()), Double.parseDouble(list.get(1)))).toList(),
                orderBookRaw.getAsks().stream().map(list ->
                        new Depth.OrderLevel(Double.parseDouble(list.getFirst()), Double.parseDouble(list.get(1)))).toList()
        );
    }

    @Getter
    private class OrderBookRaw {
        private long lastUpdateId;
        private List<List<String>> bids;
        private List<List<String>> asks;
    }

    public Volumen getVolumen(String symbol) {
        String raw = getRequest(BASE_URL + "klines" + "?symbol=" + symbol + "&interval=1m");
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root;
        try {
            root = mapper.readTree(raw);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }

        // Primera vela
        JsonNode kline = root.get(0);

        double baseVolume = kline.get(5).asDouble();   // BTC
        double quoteVolume = kline.get(7).asDouble();  // USDT
        double takerBuyQuoteVolume = kline.get(10).asDouble(); // USDT agresivo

        double sellQuoteVolume = quoteVolume - takerBuyQuoteVolume;
        double deltaUSDT = takerBuyQuoteVolume - sellQuoteVolume;
        double buyRatio = takerBuyQuoteVolume / quoteVolume;

        return new Volumen(quoteVolume, takerBuyQuoteVolume, sellQuoteVolume, deltaUSDT, buyRatio);
    }

    public Deque<Trade> getTrades(String symbol) {
        String raw = getRequest(BASE_URL + "trades" + "?symbol=" + symbol + "&limit=" + 20);
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root;
        try {
            root = mapper.readTree(raw);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        Deque<Trade> trades = new ArrayDeque<>();
        for (JsonNode trade : root) {
            double quoteQty = trade.get("quoteQty").asDouble();
            double price = trade.get("price").asDouble();
            boolean isBuyerMaker = trade.get("isBuyerMaker").asBoolean();
            long id = trade.get("id").asLong();
            long time = trade.get("time").asLong();
            trades.add(new Trade(id, time, price, quoteQty, isBuyerMaker));
        }
        return trades;
    }

    private String getRequest(String url) {
        try {
            return HttpClient.newHttpClient()
                    .send(
                            HttpRequest.newBuilder()
                                    .uri(URI.create(url))
                                    .GET()
                                    .build(),
                            HttpResponse.BodyHandlers.ofString()
                    ).body();
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
