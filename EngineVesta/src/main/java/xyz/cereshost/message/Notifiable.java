package xyz.cereshost.message;

import org.jetbrains.annotations.NotNull;

public interface Notifiable extends MediaNotification {
    @NotNull MediaNotification getMediaNotification();

    void setMediaNotification(@NotNull MediaNotification mediaNotification);

    @Override
    default void critical(String message, Object... param){
        getMediaNotification().critical(message, param);
    }

    @Override
    default void error(String message, Object... param) {
        getMediaNotification().error(message, param);
    }

    @Override
    default void waring(String message, Object... param) {
        getMediaNotification().waring(message, param);
    }

    @Override
    default void info(String message, Object... param) {
        getMediaNotification().info(message, param);
    }
}
