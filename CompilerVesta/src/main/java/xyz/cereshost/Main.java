package xyz.cereshost;

import lombok.Getter;
import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.endpoint.BinanceAPI;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.packet.PacketHandler;
import xyz.cereshost.packet.RequestMarketListener;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

public class Main {

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
            for (String symbol : Vesta.MARKETS_NAMES) Vesta.MARKETS.computeIfAbsent(symbol, Market::new).addCandles(BinanceAPI.getCandleAndVolumen(symbol));
        }, 0, 60, TimeUnit.SECONDS);
        EXECUTOR.scheduleAtFixedRate(() -> {
            for (String symbol : Vesta.MARKETS_NAMES) Vesta.MARKETS.computeIfAbsent(symbol, Market::new).addDepth(BinanceAPI.getDepth(symbol));
        }, 0, 5, TimeUnit.SECONDS);
        EXECUTOR.scheduleAtFixedRate(() -> {
            for (String symbol : Vesta.MARKETS_NAMES) Vesta.MARKETS.computeIfAbsent(symbol, Market::new).addTrade(BinanceAPI.getTrades(symbol));
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
            Vesta.info("Datos recopilados: " + ((double) (Utils.getFolderSize(Paths.get("data").toFile()) / 1024) / 1024) + " mb");
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
                market.addTrade(BinanceAPI.getTrades(symbol));
            } finally {
                latch.countDown();
            }
        });
        executor.submit(() -> {
            try {
                market.addDepth(BinanceAPI.getDepth(symbol));
            } finally {
                latch.countDown();
            }
        });
        executor.submit(() -> {
            try {
                market.addCandles(BinanceAPI.getCandleAndVolumen(symbol));
            } finally {
                latch.countDown();
            }
        });
        latch.await();
        market.sortd();
        executor.shutdown();
    }
}