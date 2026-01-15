package xyz.cereshost.file;

import xyz.cereshost.Main;
import xyz.cereshost.Utils;
import xyz.cereshost.market.Market;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.time.LocalTime;
import java.util.Comparator;
import java.util.Optional;

public class IOdata {


    private static void saveFile(String symbol, String json) throws Exception {
        LocalDate date = LocalDate.now();
        int hour = LocalTime.now().getHour();

        Path dir = Paths.get("data", symbol, date.toString());
        Files.createDirectories(dir);

        Path file = dir.resolve(hour + ".json");

        if (!Files.exists(file)) {
            Main.clearData();
        }

        Files.writeString(
                file,
                json,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
        //System.out.println("Saved " + symbol + " to " + dir);
    }

    public static void saveData() throws Exception {
        for (Market market : Utils.MARKETS.values()){
            String symbol = market.getSymbol();
            saveFile(symbol, Utils.GSON.toJson(market));
        }
    }

    public static Optional<Path> getLastSnapshot(String symbol) throws Exception {

        LocalDate today = LocalDate.now();
        Path dir = Paths.get("data", symbol, today.toString());

        if (!Files.exists(dir)) {
            return Optional.empty();
        }

        return Files.list(dir)
                .filter(p -> p.toString().endsWith(".json"))
                .max(Comparator.comparingInt(IOdata::hourFromFile));
    }

    private static int hourFromFile(Path path) {
        String name = path.getFileName().toString();
        return Integer.parseInt(name.replace(".json", ""));
    }
}
