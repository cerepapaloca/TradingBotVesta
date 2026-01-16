package xyz.cereshost;

import ai.djl.translate.TranslateException;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.file.IOdata;

import java.io.IOException;

import static xyz.cereshost.EngineUtils.checkEngines;

public class Main {

    public static void main(String[] args) throws IOException, TranslateException {
        IOdata.loadAll();
        checkEngines();

        VestaEngine.trainingModel(Vesta.MARKETS_NAMES);
    }
}