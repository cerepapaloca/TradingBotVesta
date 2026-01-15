package xyz.cereshost.common;

import com.google.gson.Gson;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.market.Market;

import java.io.File;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

@UtilityClass
public class Utils {

    public static final Gson GSON = new Gson();
    public static final ConcurrentHashMap<String, Market> MARKETS = new ConcurrentHashMap<>();
    public static final List<String> MARKETS_NAMES = List.of("BTCUSDT", "ETHUSDT", "XRPUSDT");

    public static long getFolderSize(@NotNull File folder) {
        long totalSize = 0;

        // Verificar que la ruta es un directorio
        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles(); // Obtener lista de archivos

            if (files != null) {
                for (File file : files) {
                    if (file.isFile()) {
                        totalSize += file.length(); // Sumar tamaño del archivo
                    } else if (file.isDirectory()) {
                        totalSize += getFolderSize(file); // Recursividad para subdirectorios
                    }
                }
            }
        }

        return totalSize; // Retornar el tamaño total en bytes
    }
}
