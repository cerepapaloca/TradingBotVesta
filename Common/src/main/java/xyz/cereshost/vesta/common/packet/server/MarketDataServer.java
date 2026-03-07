package xyz.cereshost.vesta.common.packet.server;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import xyz.cereshost.vesta.common.market.Market;
import xyz.cereshost.vesta.common.packet.Packet;

@Getter
@RequiredArgsConstructor
public class MarketDataServer extends Packet {

    private final Market market;
    private final long lastUpdate;
}
