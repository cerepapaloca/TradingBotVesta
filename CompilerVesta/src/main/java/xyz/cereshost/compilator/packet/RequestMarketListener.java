package xyz.cereshost.compilator.packet;

import xyz.cereshost.compilator.Main;
import xyz.cereshost.common.Vesta;
import xyz.cereshost.common.market.Market;
import xyz.cereshost.common.packet.PacketListener;
import xyz.cereshost.common.packet.client.RequestMarketClient;
import xyz.cereshost.common.packet.server.MarketDataServer;
import xyz.cereshost.compilator.file.IOdata;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RequestMarketListener extends PacketListener<RequestMarketClient> {

    private static final ExecutorService EXECUTOR_NETWORK = Executors.newScheduledThreadPool(6);

    @Override
    public void onReceive(RequestMarketClient packet) {
        EXECUTOR_NETWORK.submit(() -> {
            long systemTime = System.currentTimeMillis();
            Vesta.info("📂 Preparando datos para: " + packet.getSymbol());
            Market marketLoaded;
            if (packet.isAllMarket()){
                marketLoaded = IOdata.loadMarket(packet.getSymbol(), Integer.MAX_VALUE);
            }else {
                marketLoaded = IOdata.loadMarket(packet.getSymbol(), 2);
            }
            try {
                Main.updateData(packet.getSymbol());
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
            marketLoaded.concat(Vesta.MARKETS.get(packet.getSymbol()));
            marketLoaded.sortd();
            Vesta.info("📡 Datos recopilados de: " + packet.getSymbol() + " (" + ( System.currentTimeMillis() - systemTime) + "ms) Enviando..." );
            PacketHandler.sendPacketReply(packet, new MarketDataServer(marketLoaded, System.currentTimeMillis()));
        });

    }
}
