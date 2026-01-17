package xyz.cereshost;

import ai.djl.translate.TranslateException;
import lombok.Getter;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.file.IOdata;
import xyz.cereshost.packet.PacketHandler;

import java.io.IOException;
import java.util.List;

import static xyz.cereshost.EngineUtils.checkEngines;

public class Main {

    public static final String NAME_MODEL = "VestaIA";

    @Getter
    private static Main instance;

    public static void main(String[] args) throws IOException, TranslateException, InterruptedException {
        instance = new Main();
        System.setProperty("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:512");
        new PacketHandler();
        if (args.length > 0 && args[0].equals("tr")) {
            List<String> symbols = Vesta.MARKETS_NAMES;
            IOdata.loadMarkets(symbols);
            //checkEngines();
            VestaEngine.trainingModel(symbols);
        }else {
            String symbol = "BTCUSDT";
            IOdata.loadMarkets(symbol);
            PredictionEngine.makePrediction(symbol);
        }
    }
}