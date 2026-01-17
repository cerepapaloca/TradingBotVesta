package xyz.cereshost.common.packet.client;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import xyz.cereshost.common.packet.PacketClient;

@Getter
@RequiredArgsConstructor
public class RequestMarketClient extends PacketClient {

    private final String symbol;
}
