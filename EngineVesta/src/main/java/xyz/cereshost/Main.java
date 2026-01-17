package xyz.cereshost;

import ai.djl.translate.TranslateException;
import lombok.Getter;
import xyz.cereshost.builder.BuilderData;
import xyz.cereshost.builder.MultiSymbolNormalizer;
import xyz.cereshost.builder.RobustNormalizer;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Candle;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.packet.PacketHandler;

import java.io.IOException;
import java.util.List;

import static xyz.cereshost.EngineUtils.checkEngines;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, TranslateException {
        instance = new Main();
        if (args.length > 0 && args[0].equals("tr")) {
            IOdata.loadAll();
            checkEngines();
            VestaEngine.trainingModel(List.of("BTCUSDT"));
        }else {
            new PacketHandler();
            PacketHandler.sendPacket(new RequestMarketClient("BTCUSDT"), MarketDataServer.class).thenAccept(market -> {
                try {

                    // Cargar normalizadores
                    RobustNormalizer xNormalizer = IOdata.loadXNormalizer();
                    MultiSymbolNormalizer yNormalizer = IOdata.loadYNormalizer();

                    // Crear motor de predicción
                    PredictionEngine engine = new PredictionEngine(xNormalizer, yNormalizer, VestaEngine.LOOK_BACK, 7);

                    // Opción 2: Cargar todo de una vez
                    PredictionEngine.PredictionResult fullSystem =
                            PredictionEngine.loadFullModel("VestaIA");

                    try {
                        float prediction = engine.predictNextPrice(market.getMarket());
                        List<Candle> candles = BuilderData.to1mCandles(market.getMarket());
                        Vesta.info("Predicción: $" + prediction + " " + (((prediction - candles.get(0).close()) / candles.get(0).close()) * 100) + "  " + candles.get(0).close());

                    } finally {
                        // Cerrar recursos
                        fullSystem.close();
                    }

                } catch (Exception e) {
                    e.printStackTrace();
                }
            });
        }
    }
}