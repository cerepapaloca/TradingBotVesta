package xyz.cereshost;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.endpoint.BinanceClient;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.market.*;
import xyz.cereshost.utils.TaskReturn;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;

import static xyz.cereshost.Utils.MARKETS;
import static xyz.cereshost.Utils.MARKETS_NAMES;

public class Main {
    private static final int TICK_SIZE = 2000;
    private static final int SAVE_INTERVAL = 15;

    public static void main(String[] args) throws Exception {
        int i = 0;

        for (String name : MARKETS_NAMES) {
            Optional<Path> last = IOdata.getLastSnapshot(name);

            if (last.isPresent()) {
                String json = Files.readString(last.get());
                Market book = Utils.GSON.fromJson(json, Market.class);
                MARKETS.put(name, book);
                System.out.println("Loaded " + name);
            }
        }

        while (!Thread.currentThread().isInterrupted()) {
            long start = System.nanoTime();
            i++;

            runTick();
            if ((i % SAVE_INTERVAL) == 0){
                IOdata.saveData();
            }
            long end = System.nanoTime();
            long deltaMilis = TimeUnit.NANOSECONDS.toMillis(end - start);
            LockSupport.parkNanos(TimeUnit.MICROSECONDS.toNanos(TICK_SIZE - deltaMilis));
            //System.out.println("duración=" + ((int)deltaMilis)+ "/" + TICK_SIZE + "ms StopTime=" + (TICK_SIZE - deltaMilis));
        }
    }

    @SuppressWarnings("unchecked")
    public static void runTick() throws InterruptedException {
        for (String symbol : MARKETS_NAMES) {
            multiThreadBlocking(result -> {
                TickMarket tickMarket = new TickMarket((Volumen) result.get(0), (Depth) result.get(1));
                Market market = MARKETS.computeIfAbsent(symbol, Market::new);

                market.add(tickMarket);
                market.add((Deque<Trade>) result.get(2));
            }, () -> BinanceClient.getVolumen(symbol),() -> BinanceClient.getDepth(symbol),() -> BinanceClient.getTrades(symbol));
        }
    }

    public static void multiThreadBlocking(Consumer<List<?>> runnable, TaskReturn<?> @NotNull ... functions) throws InterruptedException {
        CountDownLatch latch = new CountDownLatch(functions.length);
        HashMap<Integer, Object> resultsMap = new HashMap<>();
        HashMap<Integer, TaskReturn<?>> functionMap = new HashMap<>();
        for (int i = 0; i < functions.length; i++) {
            functionMap.put(i, functions[i]);
        }
        for (Map.Entry<Integer, TaskReturn<?>> r : functionMap.entrySet()) {
            new Thread(() -> {
                resultsMap.put(r.getKey(), r.getValue().run());
                latch.countDown();
            }).start();
        }
        latch.await();
        List<Object> results = new ArrayList<>();
        for (int i = 0; i < resultsMap.size(); i++) {
            results.add(resultsMap.get(i));
        }
        runnable.accept(results);
    }

    public static void clearData(){
        MARKETS.clear();
    }
}