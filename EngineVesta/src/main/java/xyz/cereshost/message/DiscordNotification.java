package xyz.cereshost.message;

import net.dv8tion.jda.api.JDA;
import net.dv8tion.jda.api.JDABuilder;
import net.dv8tion.jda.api.entities.User;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.io.IOdata;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executor;
import java.util.concurrent.Executors;

public class DiscordNotification implements MediaNotification {

    private static final Executor EXECUTOR = Executors.newSingleThreadExecutor();
    private final JDA jda;
    private final String token;
    private final List<String> users = new ArrayList<>();

    public DiscordNotification(@NotNull String token) {
        this.token = token;
        jda = JDABuilder.createDefault(token).build();
    }

    public DiscordNotification() throws IOException {
        IOdata.DiscordConfig config = IOdata.loadDiscordConfig();
        this.token = config.token();
        this.users.addAll(config.users());
        jda = JDABuilder.createDefault(token).build();
    }

    @Override
    public void critical(String message, Object... param) {
        sendMessage(String.format("⛔ " + message, param));
    }

    @Override
    public void error(String message, Object... param) {
        sendMessage(String.format("🟥 " + message, param));
    }

    @Override
    public void waring(String message, Object... param) {
        sendMessage(String.format("🟨 " + message, param));
    }

    @Override
    public void info(String message, Object... param) {
        sendMessage(String.format("🟦 " + message, param));
    }

    public void sendMessage(String message){
        EXECUTOR.execute(() -> {
            for (String s: users){
                User user = jda.retrieveUserById(s).complete();
                if (user == null) continue;
                user.openPrivateChannel().flatMap(channel -> channel.sendMessage(message)).queue();
            }
        });
    }
}
