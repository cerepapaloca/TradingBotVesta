package xyz.cereshost.packet;

import org.jetbrains.annotations.NotNull;
import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.packet.*;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.locks.LockSupport;

public class PacketHandler extends BasePacketHandler {

    private static final Queue<PacketClient> packetQueue = new ConcurrentLinkedQueue<>();
    private final ExecutorService executor = Executors.newCachedThreadPool();
    private static SocketProperties socketProperties;
    private static String id;
    private static final String HOST = "192.168.1.55";// 192.168.1.55
    private static final int PORT = 2545;

    public PacketHandler() {
        executor.submit(() -> {
            while (!Thread.currentThread().isInterrupted()) {
                try {
                    Socket socket = new Socket();
                    socket.setTcpNoDelay(true);
                    socket.setKeepAlive(true);
                    //socket.setReuseAddress(true);
                    socket.connect(new InetSocketAddress(HOST, PORT), 5_000);
                    Vesta.info("✅ Cliente Conectado a %s", HOST);
                    DataOutputStream out = new DataOutputStream(socket.getOutputStream());
                    DataInputStream in = new DataInputStream(socket.getInputStream());
                    socketProperties = new SocketProperties(socket, out, in);
                    sendAllPacket();
                    handleClientConnection();
                } catch (IOException e) {
                    LockSupport.parkNanos((long) 5.0E+9);// 5s
                    e.printStackTrace();
                }
            }
        });
    }

    private void handleClientConnection() {
        try {
            DataInputStream input = socketProperties.input();
            while (!socketProperties.isClosed() && input != null) {
                int dataLength = input.readInt();
                byte[] receivedData = new byte[dataLength];
                input.readFully(receivedData);
                processPacket(receivedData);
            }
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                socketProperties.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void processPacket(byte[] message) {
        Class<?> clazz = PacketManager.getPacketClass(message);
        PacketListener<? extends Packet> packetListener = BasePacketHandler.listeners.get(clazz);
        Packet p;

        if (packetListener == null){
            p = PacketManager.decodePacket(message);
        }else {
            p = packetListener.decodePacketAndReceive(message);
        }
        BasePacketHandler.replyFuture(p);
    }

    public static void sendPacket(@NotNull PacketClient packet) {
        packet.setFrom(Main.getInstance().getClass().getName());
        byte[] data = PacketManager.encodePacket(packet);
        if (socketProperties != null && socketProperties.socket() != null && !socketProperties.isClosed()) {
            DataOutputStream out = socketProperties.output();
            try {
                synchronized (out) {
                    if (socketProperties.isClosed()) {
                        packetQueue.add(packet);
                        return;
                    }
                    out.writeInt(data.length);
                    out.write(data);
                    out.flush();
                }
            } catch (IOException e) {
                packetQueue.add(packet);
                e.printStackTrace();
            }
        }else {
            packetQueue.add(packet);
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
        sendPacket(packet);
        return packetPendingOnResponse(packet, packetRepose);
    }


    public static void sendPacketReplay(@NotNull Packet packetOld, @NotNull PacketClient packet) {
        packet.setUuidPacket(packetOld.getUuidPacket());
        sendPacket(packet);
    }
}
