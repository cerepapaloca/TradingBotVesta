package xyz.cereshost.common.packet;

import org.jetbrains.annotations.NotNull;

import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

public abstract class BasePacketHandler {

    /**
     * Los paquetes que espera una respuesta rápida
     */
    protected static final HashMap<UUID, PacketPending> packetPending = new HashMap<>();
    protected static final HashMap<Class<? extends Packet>, PacketListener<? extends Packet>> listeners = new HashMap<>();

    public static <T extends Packet> @NotNull CompletableFuture<T> packetPendingOnResponse(@NotNull Packet packet, Class<T> packetClass) {
        CompletableFuture<T> future = new CompletableFuture<>();
        UUID uuid = genereUUIDFromClass(packet, packetClass);
        packetPending.put(uuid, new PacketPending(packetClass, future));
        return future;
    }

    /**
     * Cumples el future de la repuesta rápida, Para que funcione el
     * paquete tiene que tener la misma UUID que el solicito paquete 
     * @param packet Paquete de respuesta (Debe tener la misma UUID que el paquete solicitado)
     */

    @SuppressWarnings("unchecked")
    public static void replyFuture(@NotNull Packet packet) {
        // Obtiene el future que tiene que responder
        UUID uuid = genereUUIDFromClass(packet, packet.getClass());
        PacketPending pending = packetPending.remove(uuid);
        //System.out.println("recibe " + packet.getClass().getSimpleName() + " | " + packet.getUuidPacket() + " | " + uuid);
        if (pending == null) return;
        CompletableFuture<? extends Packet> future = pending.future();
        if (!pending.clazz().isAssignableFrom(packet.getClass())) {
            packetPending.put(packet.getUuidPacket(), pending);
            future.completeExceptionally(new IllegalArgumentException("Tipo de paquete no esperado: " + packet.getClass() + ". Se esperaba: " + pending.clazz()));
            return;
        }
        try {
            ((CompletableFuture<Packet>) future).complete(packet);
        } catch (ClassCastException e) {
            packetPending.put(packet.getUuidPacket(), pending);
            future.completeExceptionally(new IllegalArgumentException("No se pudo hacer un cast: " + packet.getClass()));
        }
    }

    public static void addListener(PacketListener<? extends Packet> listener) {
        listeners.put(listener.getClazz(), listener);
    }

    protected record PacketPending(Class<? extends Packet> clazz, @NotNull CompletableFuture<? extends Packet> future) {}

    public static @NotNull String generateIdClient(@NotNull Socket socket) {
        return socket.getInetAddress().getHostAddress() + ":" +
                socket.getPort() + "/" +
                socket.getLocalAddress().getHostAddress() + ":" +
                socket.getLocalPort();
    }

    public static @NotNull String generateIdServer(@NotNull Socket socket) {
        return socket.getLocalAddress().getHostAddress() + ":" +
                socket.getLocalPort() + "/" +
                socket.getInetAddress().getHostAddress() + ":" +
                socket.getPort();
    }

    public static @NotNull UUID genereUUIDFromClass(@NotNull Packet packet, @NotNull Class<? extends Packet> clazz) {
        String name = clazz.getName() + ":" + packet.getUuidPacket().toString();
        return UUID.nameUUIDFromBytes(name.getBytes(StandardCharsets.UTF_8));
    }

}
