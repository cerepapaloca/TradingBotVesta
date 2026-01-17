package xyz.cereshost.packet;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.packet.*;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.util.HashSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.LockSupport;

public class PacketHandler extends BasePacketHandler {

    private final ExecutorService executor = Executors.newCachedThreadPool();
    @Getter
    private static final HashSet<SocketProperties> sockets = new HashSet<>();

    private final int PORT = 2545;

    public void upServer() {
        executor.submit(() -> {
            try (ServerSocket serverSocket = new ServerSocket()) {
                serverSocket.setReuseAddress(true);
                serverSocket.bind(new InetSocketAddress("0.0.0.0", PORT));
                Vesta.info("Escuchando en 0.0.0.0:%d", PORT);

                while (!Thread.currentThread().isInterrupted()) {
                    Socket socket = serverSocket.accept();

                    socket.setTcpNoDelay(true);
                    socket.setKeepAlive(true);

                    DataInputStream in = new DataInputStream(socket.getInputStream());
                    DataOutputStream out = new DataOutputStream(socket.getOutputStream());

                    String code = generateIdServer(socket);
                    Vesta.info("Servidor conectado %s", code);

                    SocketProperties sp = new SocketProperties(socket, out, in);
                    sockets.add(sp);

                    executor.submit(() -> startListening(sp));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });

    }

    private void startListening(@NotNull SocketProperties socket) {
        DataInputStream in = socket.input();
        while (!socket.isClosed() && in != null) {
            try {
                int length = in.readInt();
                byte[] messageBytes = new byte[length];
                in.readFully(messageBytes);
                processMessage(messageBytes);
            } catch (IOException e) {
                try {
                    socket.close();
                } catch (IOException ex) {
                    ex.printStackTrace();
                }
                //AviaTerraProxy.getLogger().warn("Error en la conexión con el servidor: " + e.getMessage() + " con " + socket.socket().getInetAddress().getHostAddress());
            }catch (Exception e) {
                e.printStackTrace();
            }
        }
    }

    public static void processMessage(byte[] message) {
        Class<?> clazz = PacketManager.getPacketClass(message);
        PacketListener<? extends Packet> packetListener = listeners.get(clazz);
        Packet p;
        if (packetListener == null){
            p = PacketManager.decodePacket(message);
        }else {
            p = packetListener.decodePacketAndReceive(message);
        }
        BasePacketHandler.replyFuture(p);
    }

    public static void sendPacket(@NotNull SocketProperties socket, @NotNull Packet packet){
        byte[] data = PacketManager.encodePacket(packet);

        for (SocketProperties socketProperties : PacketHandler.getSockets()){
            DataOutputStream out = socket.output();
            try {
                synchronized (out) { // Sincronizar
                    if (socket.isClosed()) {
                        return;
                    }
                    out.writeInt(data.length);
                    out.write(data);
                    out.flush();
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }
    public static void sendPacketReply(@NotNull PacketClient paketOld, @NotNull Packet packetSend) {
        packetSend.setUuidPacket(paketOld.getUuidPacket());
        for  (SocketProperties socket : PacketHandler.getSockets()){
            sendPacket(socket, packetSend);
        }
    }

    public static <T extends Packet> @NotNull CompletableFuture<T> sendPacket(SocketProperties socket, @NotNull Packet packet, Class<T> packetRepose) {
        sendPacket(socket, packet);
        return BasePacketHandler.packetPendingOnResponse(packet, packetRepose);
    }
}
