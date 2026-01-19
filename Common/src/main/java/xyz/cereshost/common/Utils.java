package xyz.cereshost.common;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import lombok.Getter;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

@UtilityClass
public class Utils {

    public static final String BASE_URL_API = "https://api.binance.com/api/v3/";
    public static final Gson GSON = new GsonBuilder().serializeSpecialFloatingPointValues().create();

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

    public static String getRequest(String url) {
        try {
            return HttpClient.newHttpClient()
                    .send(
                            HttpRequest.newBuilder()
                                    .uri(URI.create(url))
                                    .GET()
                                    .build(),
                            HttpResponse.BodyHandlers.ofString()
                    ).body();
        } catch (IOException | InterruptedException e) {
            throw new RuntimeException(e);
        }
    }
}
