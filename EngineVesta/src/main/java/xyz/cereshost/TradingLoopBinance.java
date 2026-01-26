package xyz.cereshost;

import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.strategy.TradingStrategy;
import xyz.cereshost.trading.TradingBinance;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

public class TradingLoopBinance {

    private static final long CANDLE_MS = 60_000;
    private static final long OFFSET = 5_000;
    private final String symbol;
    private final TradingBinance trading;
    private final PredictionEngine engine;
    private final TradingStrategy strategy;

    private static final ExecutorService WORKERS = Executors.newSingleThreadExecutor(r -> {
                Thread t = new Thread(r);
                t.setName("Candle-Worker");
                return t;
            });

    public TradingLoopBinance(String symbol, PredictionEngine engine, TradingStrategy tradingStrategy) {
        this.symbol = symbol;
        this.engine = engine;
        this.strategy = tradingStrategy;
        try {
            IOdata.loadMarkets(DataSource.LOCAL_NETWORK_MINIMAL, symbol);
            trading = new TradingBinance("A", "B", true, Vesta.MARKETS.get(symbol));
        } catch (InterruptedException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public void startCandleLoop() {
        new Thread(() -> {
            while (true) {
                try {
                    long serverTime = getBinanceServerTime() + OFFSET;

                    long nextCandle = ((serverTime / CANDLE_MS) + 1) * CANDLE_MS;

                    long sleep = Math.abs(nextCandle - (serverTime));
                    Vesta.info("Tiempo de espera: %.2fs", (float) sleep/1000);
                    if (sleep > 0) {
                        LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(sleep));
                    }

                    performTick();

                } catch (Exception e) {
                    e.printStackTrace();
                    LockSupport.parkNanos(TimeUnit.SECONDS.toNanos(1));
                }
            }
        }).start();
    }



    public static long getBinanceServerTime() throws Exception {
        URL url = new URL("https://api.binance.com/api/v3/time");
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("GET");

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(con.getInputStream()))) {

            String response = reader.readLine();
            return Long.parseLong(
                    response.replaceAll("\\D+", "")
            );
        }
    }

    private void performTick() {
        Market market;
        try{
            IOdata.loadMarkets(DataSource.LOCAL_NETWORK_MINIMAL, symbol);
            market = Vesta.MARKETS.get(symbol);
            if (market == null) return;
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
            return;
        }
        List<Candle> allCandles = BuilderData.to1mCandles(market);
        PredictionEngine.PredictionResult result = engine.predictNextPriceDetail(allCandles.subList(VestaEngine.LOOK_BACK, allCandles.size() - 1), symbol);
        trading.updateState(symbol);
        strategy.executeStrategy(result, trading);
    }
}
