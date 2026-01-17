package xyz.cereshost.common.packet;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public abstract class PacketClient extends Packet {

    private String from;
}
