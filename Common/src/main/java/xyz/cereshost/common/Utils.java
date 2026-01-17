package xyz.cereshost.common;

import com.google.gson.Gson;
import lombok.Getter;
import lombok.experimental.UtilityClass;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.io.File;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

@UtilityClass
public class Utils {

    public static final Gson GSON = new Gson();

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
