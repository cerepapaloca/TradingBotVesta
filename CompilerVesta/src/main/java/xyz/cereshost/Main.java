package xyz.cereshost;

import lombok.Getter;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.endpoint.BinanceClient;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.packet.PacketHandler;
import xyz.cereshost.packet.RequestMarketListener;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.Optional;
import java.util.concurrent.*;
import java.util.concurrent.locks.LockSupport;

public class Main {
    private static final int TICK_SIZE = 10000;
    private static final int SAVE_INTERVAL = 15;
    @Getter
    private static PacketHandler packetHandler;
    private static final ScheduledExecutorService EXECUTOR = Executors.newScheduledThreadPool(6);

    public static void main(String[] args) throws Exception {

        new RequestMarketListener();
        packetHandler = new PacketHandler();
        packetHandler.upServer();

        for (String name : Vesta.MARKETS_NAMES) {
            Optional<Path> last = IOdata.getLastSnapshot(name);
            if (last.isPresent()) {
                String json = Files.readString(last.get());
                Market market = Utils.GSON.fromJson(json, Market.class);
                market.sortd();
                Vesta.MARKETS.put(name, market);
                Vesta.info("Loaded " + name);
            }
        }

        EXECUTOR.scheduleAtFixedRate(() -> {
            for (String symbol : Vesta.MARKETS_NAMES) Vesta.MARKETS.computeIfAbsent(symbol, Market::new).addCandles(BinanceClient.getCandleAndVolumen(symbol));
        }, 0, 60, TimeUnit.SECONDS);
        EXECUTOR.scheduleAtFixedRate(() -> {
            for (String symbol : Vesta.MARKETS_NAMES) Vesta.MARKETS.computeIfAbsent(symbol, Market::new).addDepth(BinanceClient.getDepth(symbol));
        }, 0, 5, TimeUnit.SECONDS);
        EXECUTOR.scheduleAtFixedRate(() -> {
            for (String symbol : Vesta.MARKETS_NAMES) Vesta.MARKETS.computeIfAbsent(symbol, Market::new).addTrade(BinanceClient.getTrades(symbol));
        }, 0, 30, TimeUnit.SECONDS);

        EXECUTOR.scheduleAtFixedRate(() -> {
            try {
                IOdata.saveData();
            } catch (Exception e) {
                e.printStackTrace();
            }
            System.gc();
        }, 0, 30, TimeUnit.SECONDS);

        EXECUTOR.scheduleAtFixedRate(() -> {
           Vesta.info("Datos recopilados: %.2fmb", ((Utils.getFolderSize(Paths.get("data").toFile()) / 1024) / 1024));
        }, 0, 1, TimeUnit.MINUTES);

        // Mantener en pausa anta que detenga el programa
        LockSupport.parkNanos(Long.MAX_VALUE);
    }

    public static void updateData(String symbol) throws InterruptedException {
        var executor = Executors.newFixedThreadPool(3);
        CountDownLatch latch = new CountDownLatch(3);
        Market market = Vesta.MARKETS.computeIfAbsent(symbol, Market::new);
        executor.submit(() -> {
            try {
                market.addTrade(BinanceClient.getTrades(symbol));
            } finally {
                latch.countDown();
            }
        });
        executor.submit(() -> {
            try {
                market.addDepth(BinanceClient.getDepth(symbol));
            } finally {
                latch.countDown();
            }
        });
        executor.submit(() -> {
            try {
                market.addCandles(BinanceClient.getCandleAndVolumen(symbol));
            } finally {
                latch.countDown();
            }
        });
        latch.await();
        market.sortd();
        executor.shutdown();
    }
}