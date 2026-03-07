package xyz.cereshost.vesta.core.message;

import org.jetbrains.annotations.Nullable;

public interface MediaNotification {
    void critical(String message, Object... param);
    void error(String message, Object... param);
    void waring(String message, Object... param);
    void info(String message, Object... param);
    void updateStatus(String message, Object... param);
    void updateStatusType(@Nullable StatusType type);


    static MediaNotification empty(){
        return new MediaNotification(){
            @Override
            public void critical(String message, Object... param) {}
            @Override
            public void error(String message, Object... param) {}
            @Override
            public void waring(String message, Object... param) {}
            @Override
            public void info(String message, Object... param) {}
            @Override
            public void updateStatus(String message, Object... param) {}
            @Override
            public void updateStatusType(StatusType type) {}
        };
    }

    enum StatusType {
        STOPPED,
        TRADING,
        WAITING
    }
}
