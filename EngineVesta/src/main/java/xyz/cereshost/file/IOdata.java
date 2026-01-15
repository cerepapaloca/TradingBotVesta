package xyz.cereshost.file;

import xyz.cereshost.Utils;
import xyz.cereshost.market.Market;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

public class IOdata {


    public static void loadAll() throws IOException {
        try (Stream<Path> symbols = Files.list(Path.of("data"))) {
            for (Path symbolPath : symbols.toList()) {

                List<Path> jsons = Files.walk(symbolPath)
                        .filter(p -> p.toString().endsWith(".json"))
                        .sorted()
                        .toList();

                Market merged = null;

                for (Path json : jsons) {
                    Market m = Utils.GSON.fromJson(Files.readString(json), Market.class);

                    if (merged == null) {
                        merged = m;
                    } else {
                        merged.concat(m);
                    }
                }

                if (merged != null) {
                    Utils.MARKETS.put(symbolPath.getFileName().toString(), merged);
                }
            }
        }
    }

}
