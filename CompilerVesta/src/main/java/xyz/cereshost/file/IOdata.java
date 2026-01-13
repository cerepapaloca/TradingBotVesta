package xyz.cereshost.file;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.time.LocalDate;
import java.time.LocalTime;

public class IOdata {


    private static void saveFile(String symbol, String json) throws Exception {
        LocalDate date = LocalDate.now();
        int hour = LocalTime.now().getHour();

        Path dir = Paths.get("data", symbol, date.toString());
        Files.createDirectories(dir);

        Path file = dir.resolve(hour + ".json");

        Files.writeString(
                file,
                json,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
        );
    }

    public static void saveData() throws Exception {}
}
