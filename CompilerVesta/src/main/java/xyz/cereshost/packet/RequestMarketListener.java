package xyz.cereshost.packet;

import xyz.cereshost.common.packet.PacketListener;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.file.IOdata;

public class RequestMarketListener extends PacketListener<RequestMarketClient> {
    @Override
    public void onReceive(RequestMarketClient packet) {
        PacketHandler.sendPacketReply(packet, new MarketDataServer(IOdata.loadForIa(packet.getSymbol())));
    }
}
