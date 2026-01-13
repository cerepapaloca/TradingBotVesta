package xyz.cereshost;

import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.endpoint.BinanceClient;
import xyz.cereshost.market.*;
import xyz.cereshost.utils.TaskReturn;

import java.util.*;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;

public class Main {

    private static final int TICK_SIZE = 2000;
    private static final List<String> MARKETS_NAMES = List.of("BTCUSDT", "ETHUSDT", "XRPUSDT");
    private static final HashMap<String, Market> MARKETS = new HashMap<>();

    public static void main(String[] args) throws InterruptedException {
        while (!Thread.currentThread().isInterrupted()) {
            long start = System.nanoTime();
            runTick();
            long end = System.nanoTime();
            long deltaMilis = TimeUnit.NANOSECONDS.toMillis(end - start);
            LockSupport.parkNanos(TimeUnit.MICROSECONDS.toNanos(TICK_SIZE - deltaMilis));
            System.out.println("duración=" + ((int)deltaMilis)+ "/" + TICK_SIZE + "ms StopTime=" + (TICK_SIZE - deltaMilis));
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

    @Contract(value = "_ -> new", pure = true)
    public static @NotNull ThreadFactory createThreadFactoryNamed(String name) {
        return new ThreadFactory() {
            private final AtomicInteger count = new AtomicInteger(0);
            @Override
            public Thread newThread(@NotNull Runnable r) {
                Thread t = new Thread(r);
                t.setName("Vesta-" + name + "-" + count.getAndIncrement());
                return t;
            }
        };
    }
}