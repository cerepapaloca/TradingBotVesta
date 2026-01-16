package xyz.cereshost;

import xyz.cereshost.common.Utils;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.*;
import xyz.cereshost.endpoint.BinanceClient;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.utils.TaskReturn;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.*;
import java.util.concurrent.locks.LockSupport;
import java.util.function.Consumer;

public class Main {
    private static final int TICK_SIZE = 2000;
    private static final int SAVE_INTERVAL = 15;

    public static void main(String[] args) throws Exception {
        int i = 0;
        int j = 0;

        for (String name : Vesta.MARKETS_NAMES) {
            Optional<Path> last = IOdata.getLastSnapshot(name);

            if (last.isPresent()) {
                String json = Files.readString(last.get());
                Market book = Utils.GSON.fromJson(json, Market.class);
                Vesta.MARKETS.put(name, book);
                System.out.println("Loaded " + name);
            }
        }

        while (!Thread.currentThread().isInterrupted()) {
            long start = System.nanoTime();
            i++;


            try {
                runTick();
            }catch (Exception e) {
                e.printStackTrace();
            }
            if ((i % SAVE_INTERVAL) == 0){
                IOdata.saveData();
            }
            if ((j % 50) == 0){
                System.out.printf("[%s] Tamaño: %dkb", LocalTime.now().format(DateTimeFormatter.ISO_LOCAL_TIME), (Utils.getFolderSize(Paths.get("data").toFile()) / 1024));
            }
            j++;
            long end = System.nanoTime();
            long deltaMilis = TimeUnit.NANOSECONDS.toMillis(end - start);
            LockSupport.parkNanos(TimeUnit.MICROSECONDS.toNanos(TICK_SIZE - deltaMilis));
            //System.out.println("duración=" + ((int)deltaMilis)+ "/" + TICK_SIZE + "ms StopTime=" + (TICK_SIZE - deltaMilis));
        }
    }

    @SuppressWarnings("unchecked")
    public static void runTick() throws InterruptedException {
        for (String symbol : Vesta.MARKETS_NAMES) {
            multiThreadBlocking(result -> {
                try {
                    TickMarket tickMarket = new TickMarket((Volumen) result.get(0), (Depth) result.get(1));
                    Market market = Vesta.MARKETS.computeIfAbsent(symbol, Market::new);

                    market.add(tickMarket);
                    market.add((Deque<Trade>) result.get(2));
                }catch (Exception e) {
                    e.printStackTrace();
                    System.out.println("recopilador funcionado...");
                }
            }, () -> BinanceClient.getVolumen(symbol),() -> BinanceClient.getDepth(symbol),() -> BinanceClient.getTrades(symbol));
        }
    }

    private static final ExecutorService EXECUTOR = Executors.newFixedThreadPool(4);

    public static void multiThreadBlocking(
            Consumer<List<?>> runnable,
            TaskReturn<?>... functions
    ) throws InterruptedException {

        List<Future<?>> futures = new ArrayList<>();

        for (TaskReturn<?> task : functions) {
            futures.add(EXECUTOR.submit(task::run));
        }

        List<Object> results = new ArrayList<>();

        for (Future<?> future : futures) {
            try {
                results.add(future.get());
            } catch (ExecutionException e) {
                throw new RuntimeException(e.getCause());
            }
        }

        runnable.accept(results);
    }


    public static void clearData(){
        Vesta.MARKETS.clear();
    }
}