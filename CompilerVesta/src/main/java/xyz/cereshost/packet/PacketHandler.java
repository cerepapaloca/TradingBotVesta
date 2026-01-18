package xyz.cereshost.packet;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.packet.*;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.HashSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.LockSupport;

public class PacketHandler extends BasePacketHandler {

    private final ExecutorService executor = Executors.newCachedThreadPool();
    @Getter
    private static  SocketProperties socketLast = null;

    private final int PORT = 2545;

    public void upServer() {
        executor.submit(() -> {
            try {
                ServerSocket serverSocket = ServerSocketChannel.open().socket();
                serverSocket.setReuseAddress(true);
                serverSocket.bind(new InetSocketAddress("0.0.0.0", PORT));
                Vesta.info("🚀 Servidor escuchando en 0.0.0.0:%d", PORT);

                while (!Thread.currentThread().isInterrupted()) {
                    Socket socket = serverSocket.accept();

                    socket.setTcpNoDelay(true);
                    socket.setKeepAlive(true);
                    socket.setSendBufferSize(4 * 1024 * 1024);
                    socket.setReceiveBufferSize(4 * 1024 * 1024);

                    BufferedInputStream in =
                            new BufferedInputStream(socket.getInputStream(), 4 * 1024 * 1024);
                    BufferedOutputStream out =
                            new BufferedOutputStream(socket.getOutputStream(), 4 * 1024 * 1024);

                    String code = generateIdServer(socket);
                    Vesta.info("🔗 Cliente conectado: %s", code);

                    SocketProperties sp = new SocketProperties(socket, out, in);
                    socketLast = sp;

                    executor.submit(() -> startListening(sp));
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    private void startListening(@NotNull SocketProperties sp) {
        SocketChannel channel = sp.socket().getChannel();

        ByteBuffer header = ByteBuffer.allocateDirect(4);
        ByteBuffer body = null;

        try {
            while (!sp.isClosed()) {

                header.clear();
                while (header.hasRemaining()) {
                    if (channel.read(header) == -1) {
                        throw new EOFException();
                    }
                }

                header.flip();
                int length = header.getInt();

                body = ByteBuffer.allocateDirect(length);
                while (body.hasRemaining()) {
                    if (channel.read(body) == -1) {
                        throw new EOFException();
                    }
                }

                body.flip();
                byte[] message = new byte[length];
                body.get(message);

                processMessage(message);
            }
        } catch (IOException e) {

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

    public static void sendPacket(@NotNull Packet packet){
        byte[] payload = PacketManager.encodePacket(packet);
        SocketProperties sp = socketLast;
        if (sp == null || sp.isClosed()) {
            return;
        }
        try {
            SocketChannel channel = sp.socket().getChannel();
            // Buffer reutilizable (idealmente ThreadLocal o dentro de SocketProperties)
            ByteBuffer buffer = ensureCapacity(sp.writeBuffer(), Integer.BYTES + payload.length);
            buffer.clear();
            buffer.putInt(payload.length);

            buffer.put(payload);
            buffer.flip();
            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }

        } catch (IOException e) {
            try {
                sp.close();
            } catch (IOException ignored) {}
        }
    }

    private static ByteBuffer ensureCapacity(ByteBuffer buffer, int required) {
        if (buffer.capacity() >= required) {
            buffer.clear();
            return buffer;
        }

        return ByteBuffer.allocateDirect(required);
    }

    public static void sendPacketReply(@NotNull PacketClient paketOld, @NotNull Packet packetSend) {
        packetSend.setUuidPacket(paketOld.getUuidPacket());
        sendPacket(packetSend);
    }

    public static <T extends Packet> @NotNull CompletableFuture<T> sendPacket(SocketProperties socket, @NotNull Packet packet, Class<T> packetRepose) {
        sendPacket(packet);
        return BasePacketHandler.packetPendingOnResponse(packet, packetRepose);
    }
}
