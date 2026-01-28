package xyz.cereshost.common.packet;

import lombok.Getter;
import lombok.Setter;

import java.util.UUID;

@Getter
@Setter
public abstract class PacketClient extends Packet {

    private UUID from;
}
