package xyz.cereshost.packet;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.packet.*;
import xyz.cereshost.common.packet.client.HelloClient;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.nio.channels.SocketChannel;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.UUID;
import java.util.concurrent.*;
import java.util.concurrent.locks.LockSupport;

public class PacketHandler extends BasePacketHandler {

    private static final Queue<PacketClient> packetQueue = new ConcurrentLinkedQueue<>();
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private static SocketProperties socketProperties;
    public static final UUID idClient = UUID.randomUUID();
    private static final String HOST = "192.168.1.55";// localhost
    private static final CountDownLatch latch = new CountDownLatch(1);

    private static final int PORT = 2545;

    public PacketHandler() {
        executor.submit(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Socket socket = SocketChannel.open().socket();

                    socket.setTcpNoDelay(true);
                    socket.setKeepAlive(true);

                    // Buffers grandes (MUY importante)
                    socket.setSendBufferSize(4 * 1024 * 1024);   // 4MB
                    socket.setReceiveBufferSize(4 * 1024 * 1024);

                    socket.connect(new InetSocketAddress(HOST, PORT), 5_000);
                    Vesta.info("✅ Cliente conectado a %s", HOST);

                    BufferedOutputStream out =
                            new BufferedOutputStream(socket.getOutputStream(), 4 * 1024 * 1024);
                    BufferedInputStream in =
                            new BufferedInputStream(socket.getInputStream(), 4 * 1024 * 1024);

                    socketProperties = new SocketProperties(socket, out, in);

                    sendAllPacket();          // Enviar en bloques
                    sendPacket(new HelloClient());
                    Vesta.info("✅ UUID del cliente: %s", idClient);
                    latch.countDown();
                    handleClientConnection(); // Leer en bloques
                } catch (IOException e) {
                    Vesta.warning("Reconectando en 5s...");
                    LockSupport.parkNanos(5_000_000_000L);
                }
            }
        });

    }

    private void handleClientConnection() {
        SocketProperties sp = socketProperties;
        SocketChannel channel = sp.socket().getChannel();

        ByteBuffer header = ByteBuffer.allocateDirect(4);
        ByteBuffer body;

        try {
            while (!sp.isClosed()) {

                // Leer tamaño
                header.clear();
                while (header.hasRemaining()) {
                    if (channel.read(header) == -1) {
                        throw new EOFException();
                    }
                }
                header.flip();
                int length = header.getInt();

                // Leer cuerpo
                body = ByteBuffer.allocateDirect(length);
                while (body.hasRemaining()) {
                    if (channel.read(body) == -1) {
                        throw new EOFException();
                    }
                }

                body.flip();
                byte[] data = new byte[length];
                body.get(data);

                processPacket(data);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void processPacket(byte[] message) {
        Class<?> clazz = PacketManager.getPacketClass(message);
        PacketListener<? extends Packet> packetListener = BasePacketHandler.listeners.get(clazz);
        Packet p = PacketManager.decodePacket(message);;

        if (packetListener != null){
            packetListener.receivePacket(p);
        }
        BasePacketHandler.replyFuture(p);
    }

    public static void sendPacket(@NotNull PacketClient packet) {
        packet.setFrom(idClient);
        byte[] payload = PacketManager.encodePacket(packet);

        SocketProperties sp = socketProperties;
        if (sp == null || sp.isClosed()) {
            packetQueue.add(packet);
            return;
        }
        try {
            SocketChannel channel = sp.socket().getChannel();

            ByteBuffer buffer = sp.writeBuffer();
            buffer.clear();

            buffer.putInt(payload.length);
            buffer.put(payload);
            buffer.flip();

            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }

        } catch (IOException e) {
            packetQueue.add(packet);
            try {
                sp.close();
            } catch (IOException ignored) {}
        }
    }

    public static void sendAllPacket() {
        List<PacketClient> packets = new ArrayList<>(packetQueue);
        packetQueue.clear();
        for (PacketClient packet : packets) {
            sendPacket(packet);
        }
    }

    public static <T extends Packet> @NotNull CompletableFuture<T> sendPacket(@NotNull PacketClient packet, Class<T> packetRepose) {
        try {
            latch.await();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        }
        sendPacket(packet);
        return packetPendingOnResponse(packet, packetRepose);
    }


    public static void sendPacketReplay(@NotNull Packet packetOld, @NotNull PacketClient packet) {
        packet.setUuidPacket(packetOld.getUuidPacket());
        sendPacket(packet);
    }
}
