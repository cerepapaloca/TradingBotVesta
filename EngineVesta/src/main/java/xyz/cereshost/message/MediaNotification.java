package xyz.cereshost.message;

public interface MediaNotification {
    void critical(String message, Object... param);
    void error(String message, Object... param);
    void waring(String message, Object... param);
    void info(String message, Object... param);

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
        };
    }
}
