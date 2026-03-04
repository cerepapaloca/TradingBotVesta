package xyz.cereshost.trading;

import lombok.Getter;
import lombok.Setter;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import xyz.cereshost.DataSource;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.engine.PredictionEngine;
import xyz.cereshost.engine.VestaEngine;
import xyz.cereshost.exception.BinanceCodeWeakException;
import xyz.cereshost.io.IOMarket;
import xyz.cereshost.message.MediaNotification;
import xyz.cereshost.message.Notifiable;
import xyz.cereshost.strategy.TradingStrategy;
import xyz.cereshost.utils.BuilderData;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.HttpURLConnection;
import java.net.URI;
import java.net.URL;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.LockSupport;

public final class TradingTickLoop implements Notifiable {

    private static final long CANDLE_MS = 60_000;
    private static final long OFFSET = 1_500;
    private static final int LOCAL_ZIP_WARMUP_DAYS = 5;
    private static final int HISTORY_WINDOW_CANDLES = 5_500;
    private final String symbol;
    @Getter
    private final Executor executor = Executors.newFixedThreadPool(6);
    @NotNull
    private final TradingBinance trading;
    @Nullable
    private final PredictionEngine engine;
    @Getter
    private final TradingStrategy strategy;
    @NotNull @Getter @Setter
    private MediaNotification mediaNotification;
    @Nullable
    private final Market localWarmupMarket;

    private static final ScheduledExecutorService WORKERS = Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r);
                t.setName("Candle-Worker");
                return t;
            });

    public TradingTickLoop(@NotNull String symbol,
                           @Nullable PredictionEngine engine,
                           @NotNull TradingStrategy tradingStrategy,
                           @NotNull BinanceApi binanceApi,
                           @Nullable MediaNotification mediaNotification
    ) {
        this.symbol = symbol;
        this.engine = engine;
        this.strategy = tradingStrategy;
        this.mediaNotification = Objects.requireNonNullElse(mediaNotification, MediaNotification.empty());
        try {
            binanceApi.setExceptionHandler(this::stop);
            binanceApi.setMediaNotification(this.mediaNotification);

            Market warmupMarket = IOMarket.loadMarketsRecentDays(symbol, LOCAL_ZIP_WARMUP_DAYS, true);
            if (warmupMarket.getCandleSimples().isEmpty()) {
                this.localWarmupMarket = null;
                Vesta.warning("No se encontró histórico LOCAL_ZIP para warmup, usando solo BINANCE en vivo.");
            } else {
                this.localWarmupMarket = warmupMarket;
                Vesta.info("Warmup cargado desde LOCAL_ZIP: %d días (%d velas).",
                        LOCAL_ZIP_WARMUP_DAYS, warmupMarket.getCandleSimples().size()
                );
            }

            Market bootMarket = loadMarkt();
            trading = new TradingBinance(binanceApi, mediaNotification, bootMarket);
            trading.setTradingTickLoop(this);
        } catch (InterruptedException | IOException e) {
            throw new RuntimeException(e);
        }
    }

    public boolean isClose = false;

    public void startCandleLoop() {
        trading.updateOrdensSimple();
        if (localWarmupMarket != null) {
            WORKERS.scheduleAtFixedRate(() -> {
                try {
                    localWarmupMarket.clear();
                    localWarmupMarket.concat(IOMarket.loadMarketsRecentDays(symbol, LOCAL_ZIP_WARMUP_DAYS, false));
                } catch (InterruptedException | IOException e) {
                    throw new RuntimeException(e);
                }
            }, 12, 12, TimeUnit.HOURS);
        }
        WORKERS.submit(() -> {
            while (!Thread.currentThread().isInterrupted() && !isClose) {
                try {
                    long serverTime = getBinanceServerTime();
                    long nextCandle = ((serverTime / CANDLE_MS) + 1) * CANDLE_MS;
                    long targetTime = nextCandle + OFFSET;
                    long sleep = targetTime - serverTime;

                    Vesta.info("💤 Tiempo de espera: %.2fs", (float) sleep/1000);
                    if (sleep > 0) LockSupport.parkNanos(TimeUnit.MILLISECONDS.toNanos(sleep));
                    long time = System.currentTimeMillis();

                    performTick();
                    updateStatus();

                    Vesta.info("🕑 Tiempo de procesamiento: %.2fss", (float) (System.currentTimeMillis() - time)/1000f);
                } catch (Exception e) {
                    Vesta.sendErrorException("Error en el loop", e);
                    LockSupport.parkNanos(TimeUnit.SECONDS.toNanos(1));
                }
            }
        });
    }

    public static long getBinanceServerTime() throws Exception {
        URL url = URI.create("https://api.binance.com/api/v3/time").toURL();
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

    private void performTick() throws InterruptedException {
        AtomicReference<Market> market = new AtomicReference<>(null);
        CountDownLatch latch = new CountDownLatch(2);

        executor.execute(() -> {
            try {
                market.set(loadMarkt());
            } catch (InterruptedException | IOException e) {
                stop(e);
            } finally {
                latch.countDown();
            }
        });

        executor.execute(() -> {
            trading.updateOrdens(symbol);
            latch.countDown();
        });

        latch.await();
        Market tickMarket = market.get();
        if (tickMarket == null) {
            Vesta.warning("No se pudo cargar mercado para este tick.");
            return;
        }
        tickMarket.getCandleSimples().removeIf(cs -> (cs.openTime() % (1000 * 60 * 60 * 24)) == 0);
        List<Candle> allCandles = BuilderData.to1mCandles(tickMarket);
        if (allCandles.size() <= VestaEngine.LOOK_BACK + 1) {
            Vesta.warning("Histórico insuficiente para tick: %d velas", allCandles.size());
            return;
        }

        int endExclusive = allCandles.size() - 1;
        int startInclusive = Math.max(VestaEngine.LOOK_BACK, endExclusive - HISTORY_WINDOW_CANDLES);
        if (startInclusive >= endExclusive) {
            return;
        }
        List<Candle> visible = allCandles.subList(startInclusive, endExclusive);
        PredictionEngine.PredictionResult result;
        if (engine != null) {
            result = engine.predictNextPriceDetail(visible, symbol);
        }else {
            result = null;
        }
        trading.getOpens().forEach(Trading.OpenOperation::next);
        strategy.executeStrategy(result, visible, trading);
    }

    public void updateStatus(){
        if (isClose) {
            updateStatusType(StatusType.STOPPED);
            updateStatus("Loop detenido");
            return;
        }

        if (trading.hasOpenOperation()){
            updateStatusType(StatusType.TRADING);
            int longs = 0;
            int shorts = 0;
            for (Trading.OpenOperation op : trading.getOpens()) {
                if (op.getDireccion().equals(Trading.DireccionOperation.LONG)) {
                    longs++;
                }else {
                    shorts++;
                }
            }
            if (longs == 1 && shorts == 0) {
                updateStatus("Operando Long en %s", symbol);
                return;
            }
            if (longs > 1 && shorts == 0) {
                updateStatus("Operando %d Longs en %s", longs, symbol);
                return;
            }
            if (longs == 0 && shorts == 1) {
                updateStatus("Operando Short en %s", symbol);
                return;
            }
            if (longs == 0 && shorts > 1) {
                updateStatus("Operando %d Shorts en %s", longs, symbol);
                return;
            }
            if (longs == 1 && shorts == 1) {
                updateStatus("Operando Long y Short en %s", symbol);
                return;
            }
            updateStatus("Operando: %d L y %d S en ", longs, shorts, shorts);
        }else {
            updateStatusType(StatusType.WAITING);
            updateStatus("Esperando el momento...");
        }

    }

    public void stop(Exception e){
        if (!(e instanceof BinanceCodeWeakException)) {
            Vesta.sendErrorException("Deteniendo Loop por: ", e);
            mediaNotification.critical(String.format("**Deteniendo Loop** por: %s. Revisar Consola para más información", e.getMessage()));
            updateStatus();
            isClose = true;
        }
    }

    @Contract(pure = true, value = " -> new")
    public Market loadMarkt() throws IOException, InterruptedException {
        Market liveMarket = IOMarket.loadMarkets(DataSource.BINANCE, symbol);
        if (localWarmupMarket == null || localWarmupMarket.getCandleSimples().isEmpty()) {
            return liveMarket;
        }

        Market merged = new Market(symbol);
        merged.concat(localWarmupMarket);
        merged.concat(liveMarket);
        merged.sortd();
        return merged;
    };
}
