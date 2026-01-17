package xyz.cereshost.packet;

import xyz.cereshost.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.packet.PacketListener;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.file.IOdata;

public class RequestMarketListener extends PacketListener<RequestMarketClient> {
    @Override
    public void onReceive(RequestMarketClient packet) {

        Market marketLoaded = IOdata.loadMarket(packet.getSymbol());
        try {
            Main.updateData(packet.getSymbol());
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
        marketLoaded.concat(Vesta.MARKETS.get(packet.getSymbol()));
        marketLoaded.sortd();
        PacketHandler.sendPacketReply(packet, new MarketDataServer(marketLoaded));
    }
}
