package xyz.cereshost;

import ai.djl.translate.TranslateException;
import xyz.cereshost.file.IOdata;

import java.io.IOException;
import java.util.List;

import static xyz.cereshost.EngineUtils.checkEngines;

public class Main {

    public static void main(String[] args) throws IOException, TranslateException {
        IOdata.loadAll();
        checkEngines();

        VestaEngine.trainingModel(List.of("XRPUSDT"));
    }
}