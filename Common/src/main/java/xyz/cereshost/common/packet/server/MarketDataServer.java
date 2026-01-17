package xyz.cereshost.common.packet.server;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.packet.Packet;

@Getter
@RequiredArgsConstructor
public class MarketDataServer extends Packet {

    private final Market market;
}
