package xyz.cereshost;

import ai.djl.translate.TranslateException;
import lombok.Getter;
import xyz.cereshost.packet.PacketHandler;

import java.io.IOException;
import java.util.List;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    public static final List<String> SYMBOLS_TRAINING = List.of("SOLUSDT");

    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, TranslateException, InterruptedException {
        instance = new Main();
        new PacketHandler();
        if (args.length > 0 && args[0].equals("tr")) {
            //List.of("BTCUSDT");// Vesta.MARKETS_NAMES;
            //checkEngines();
            VestaEngine.trainingModel(SYMBOLS_TRAINING);
        }else {
            String symbol = "SOLUSDT";
            PredictionEngine.makePrediction(symbol);
        }
    }
}