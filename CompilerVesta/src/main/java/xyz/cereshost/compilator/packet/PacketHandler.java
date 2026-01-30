package xyz.cereshost.compilator.packet;

import lombok.Getter;
import org.jetbrains.annotations.NotNull;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.packet.*;
import xyz.cereshost.common.packet.client.HelloClient;

import java.io.*;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.SocketException;
import java.nio.ByteBuffer;
import java.nio.channels.ServerSocketChannel;
import java.nio.channels.SocketChannel;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class PacketHandler extends BasePacketHandler {

    private final ExecutorService executor = Executors.newFixedThreadPool(4);
    @Getter
    private static final HashMap<UUID, SocketProperties> sockets = new HashMap<>();
    @Getter
    private static final HashMap<UUID, UUID> mapUUIDS = new HashMap<>();

    private final int PORT = 2545;

    public void upServer() {
        executor.submit(() -> {
            try (ServerSocketChannel serverChannel = ServerSocketChannel.open()) {
                serverChannel.bind(new InetSocketAddress("0.0.0.0", PORT));
                Vesta.info("🚀 Servidor escuchando en 0.0.0.0:%d", PORT);

                while (!Thread.currentThread().isInterrupted()) {
                    SocketChannel clientChannel = serverChannel.accept();

                    // Configuración del Socket
                    Socket socket = clientChannel.socket();
                    socket.setTcpNoDelay(true);
                    socket.setKeepAlive(true);
                    socket.setSendBufferSize(4 * 1024 * 1024);
                    socket.setReceiveBufferSize(4 * 1024 * 1024);

                    String code = socket.getRemoteSocketAddress().toString();

                    // Nota: SocketProperties debe estar preparado para manejar SocketChannel
                    SocketProperties sp = new SocketProperties(socket, null, null);
                    UUID uuid = UUID.randomUUID();
                    sockets.put(uuid, sp);
                    mapUUIDS.put(uuid, null);

                    executor.submit(() -> {
                        Vesta.info("🔗 Cliente conectado: %s", code);
                        startListening(sp, clientChannel);
                    });
                }
            } catch (IOException e) {
                Vesta.info("❌ Error en el servidor: %s", e.getMessage());
            }
        });
    }

    private void startListening(@NotNull SocketProperties sp, SocketChannel channel) {
        ByteBuffer header = ByteBuffer.allocateDirect(4);

        try {
            while (channel.isOpen() && !Thread.currentThread().isInterrupted()) {
                header.clear();

                // Leer el tamaño del paquete (4 bytes)
                while (header.hasRemaining()) {
                    if (channel.read(header) == -1) throw new EOFException("Cliente desconectado");
                }

                header.flip();
                int length = header.getInt();

                if (length <= 0 || length > 10 * 1024 * 1024) { // Protección básica contra paquetes corruptos
                    break;
                }

                ByteBuffer body = ByteBuffer.allocateDirect(length);
                while (body.hasRemaining()) {
                    if (channel.read(body) == -1) throw new EOFException("Conexión perdida al leer cuerpo");
                }

                body.flip();
                byte[] message = new byte[length];
                body.get(message);

                processMessage(message);
            }
        } catch (EOFException e) {
            Vesta.info("🔌 Cliente desconectado formalmente.");
        } catch (SocketException ignored){
        } catch (Exception e) {
            Vesta.info("⚠️ Error en lectura de paquete: %s", e.getMessage());
        } finally {
            cleanup(sp, channel);
        }
    }

    private void cleanup(SocketProperties sp, SocketChannel channel) {
        try {
            if (channel != null) channel.close();
            if (sp.socket() != null) sp.socket().close();
        } catch (IOException ignored) {}

        sockets.entrySet().removeIf(entry -> entry.getValue().equals(sp));
    }

    public static void processMessage(byte[] message) {
        try {
            Class<?> clazz = PacketManager.getPacketClass(message);
            PacketListener<? extends Packet> packetListener = listeners.get(clazz);
            Packet p = PacketManager.decodePacket(message);

            if (p instanceof HelloClient hello) {
                for (Map.Entry<UUID, UUID> entry : mapUUIDS.entrySet()) {
                    if (entry.getValue() == null) {
                        mapUUIDS.put(entry.getKey(), hello.getFrom());
                        Vesta.info("✅ Vinculado: UUID Cliente %s -> Server ID %s", hello.getFrom(), entry.getKey());
                        break;
                    }
                }
            }

            if (packetListener != null) {
                packetListener.receivePacket(p);
            }
            BasePacketHandler.replyFuture(p);
        } catch (Exception e) {
            Vesta.info("❌ Error procesando mensaje: %s", e.getMessage());
        }
    }

    public static void sendPacket(@NotNull Packet packet, UUID to) {
        byte[] payload = PacketManager.encodePacket(packet);
        UUID internalId = null;

        for (Map.Entry<UUID, UUID> entry : mapUUIDS.entrySet()) {
            if (to.equals(entry.getValue())) {
                internalId = entry.getKey();
                break;
            }
        }

        if (internalId == null) return;
        SocketProperties sp = sockets.get(internalId);

        if (sp == null || sp.isClosed()) return;

        try {
            SocketChannel channel = sp.socket().getChannel();
            if (channel == null) return;

            ByteBuffer buffer = ByteBuffer.allocateDirect(Integer.BYTES + payload.length);
            buffer.putInt(payload.length);
            buffer.put(payload);
            buffer.flip();

            while (buffer.hasRemaining()) {
                channel.write(buffer);
            }
        } catch (IOException e) {
            Vesta.info("❌ Error enviando paquete: %s", e.getMessage());
        }
    }

    public static void sendPacketReply(@NotNull PacketClient paketOld, @NotNull Packet packetSend) {
        packetSend.setUuidPacket(paketOld.getUuidPacket());
        sendPacket(packetSend, paketOld.getFrom());
    }
}