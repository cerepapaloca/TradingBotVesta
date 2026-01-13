package xyz.cereshost.packet;

import lombok.Getter;
import lombok.Setter;

import java.util.UUID;

import static xyz.cereshost.packet.PacketManager.gsonPacket;

@Getter
@Setter
public abstract class Packet {

    /**
     * El ID que identifica su clase de manera única
     */
    private final int id;

    /**
     * La UUID que identifica el paquete de manera única
     */
    private UUID uuidPacket;

    public Packet(){
        id = PacketManager.whatIsMyId(this.getClass());
        uuidPacket = UUID.randomUUID();
    }

    @Override
    public String toString() {
        return gsonPacket.toJson(this);
    }
}
