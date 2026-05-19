package xyz.cereshost.vesta.common;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

public class Vesta {

    private static final Logger LOGGER = LogManager.getLogger(Vesta.class);

    public static synchronized void info(String message, Object... o) {
        LOGGER.info(String.format(message, o));
    }
    public static synchronized void info(String message) {
        LOGGER.info(message);
    }

    public static synchronized void warning(String message, Object... o) {
        LOGGER.warn(String.format(message, o));
    }


    public static synchronized void warning(String message) {
        LOGGER.warn(message);
    }

    public static synchronized void error(String message, Object... o) {
        LOGGER.error(String.format(message, o));
    }

    public static synchronized void error(String message) {
        LOGGER.error(message);
    }

    public static synchronized void clearLine(){
        System.out.print("\r " + " ".repeat(150) + "\r");
    }
    public static synchronized void sendErrorException(String message, Exception exception) {
//        LOGGER.error(setFormatException(message, exception));
        exception.printStackTrace();
    }

    public static synchronized void sendWaringException(String message, Exception exception) {
        //LOGGER.warn(setFormatException(message, exception));
        exception.printStackTrace();
    }

    private static String setFormatException(String message, Exception exception) {
        StringBuilder builder = new StringBuilder();
        for (StackTraceElement element : exception.getStackTrace()) {
            builder.append(element.toString()).append("\n\t");
        }
        for (Throwable throwable : exception.getSuppressed()) {
            builder.append("[").append(throwable.getCause()).append("=").append(throwable.getMessage()).append("]").append("\n\t");
            for (StackTraceElement element : throwable.getStackTrace()) {
                builder.append(element.toString()).append("\n\t");
            }
        }

        return String.format("%s [%s=%s] \n\t%s", message, exception.getClass().getSimpleName(), exception.getMessage(), builder);
    }
}
