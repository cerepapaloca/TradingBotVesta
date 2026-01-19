package xyz.cereshost.endpoint;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.gson.Gson;
import lombok.Getter;
import lombok.experimental.UtilityClass;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.common.market.Depth;
import xyz.cereshost.common.market.Trade;
import xyz.cereshost.common.market.Volumen;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.List;

@UtilityClass
public class BinanceAPI {

    public static final int LIMIT_DEPTH = 40;

    public synchronized Depth getDepth(String symbol) {
        String raw = Utils.getRequest(Utils.BASE_URL_API + "depth" + "?symbol=" + symbol + "&limit=" + LIMIT_DEPTH);
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

    public synchronized List<CandleSimple> getCandleAndVolumen(String symbol) {
        String raw = Utils.getRequest(Utils.BASE_URL_API + "klines" + "?symbol=" + symbol + "&interval=1m&limit=" + 300);
        ObjectMapper mapper = new ObjectMapper();
        JsonNode root;
        try {
            root = mapper.readTree(raw);
        } catch (JsonProcessingException e) {
            throw new RuntimeException(e);
        }
        ArrayDeque<CandleSimple> deque = new ArrayDeque<>();

        for (int i = 0; i < 300; i++) {
            JsonNode kline = root.get(i);

            double baseVolume = kline.get(5).asDouble();
            double quoteVolume = kline.get(7).asDouble();  // USDT
            double takerBuyQuoteVolume = kline.get(10).asDouble(); // USDT agresivo

            double sellQuoteVolume = quoteVolume - takerBuyQuoteVolume;
            double deltaUSDT = takerBuyQuoteVolume - sellQuoteVolume;
            double buyRatio = takerBuyQuoteVolume / quoteVolume;
            deque.add(new CandleSimple(
                    kline.get(0).asLong(),
                    kline.get(1).asDouble(), // open
                    kline.get(2).asDouble(), // high
                    kline.get(3).asDouble(), // low
                    kline.get(4).asDouble(), // close
                    new Volumen(quoteVolume, baseVolume, takerBuyQuoteVolume, sellQuoteVolume, deltaUSDT, buyRatio)));
        }
        return List.of(deque.toArray(new CandleSimple[0]));
    }

    public synchronized Deque<Trade> getTrades(String symbol) {
        String raw = Utils.getRequest(Utils.BASE_URL_API + "trades" + "?symbol=" + symbol + "&limit=" + 400);
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


}
