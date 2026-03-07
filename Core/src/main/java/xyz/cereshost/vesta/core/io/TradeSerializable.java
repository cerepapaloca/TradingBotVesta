package xyz.cereshost.vesta.core.io;

import xyz.cereshost.vesta.core.engine.VestaEngine;
import xyz.cereshost.vesta.common.market.Trade;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.FutureTask;

import static xyz.cereshost.vesta.core.io.IOMarket.*;

public class TradeSerializable implements SerializableBin<Trade>, SerializableCSV<Trade> {

    @Override
    public void submitBatch(Deque<FutureTask<List<Trade>>> tasks, List<String> batch) {
        List<String> batchCopy = new ArrayList<>(batch);
        FutureTask<List<Trade>> task = new FutureTask<>(() -> parseTradeBatch(batchCopy));
        tasks.addLast(task);
        VestaEngine.EXECUTOR_AUXILIAR_BUILD.execute(task);
    }

    @Override
    public void writeBin(File zipFile, Deque<Trade> trades) throws IOException {
        String entryName = binEntryName(zipFile);
        rewriteZipWithBinEntry(zipFile, entryName, out -> {
            out.writeInt(TRADE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            for (Trade trade : trades) {
                out.writeLong(trade.time());
                out.writeDouble(trade.price());
                out.writeDouble(trade.qty());
                out.writeBoolean(trade.isBuyerMaker());
            }
        });
    }

    @Override
    public Deque<Trade> readBin(DataInputStream in) throws IOException {
        int magic = in.readInt();
        if (magic != TRADE_BIN_MAGIC) {
            return null;
        }
        int version = in.readInt();
        if (version != BIN_VERSION) {
            return null;
        }
        Deque<Trade> list = new ArrayDeque<>(250_000);
        while (true) {
            try {
                long time = in.readLong();
                double price = in.readDouble();
                double qty = in.readDouble();
                boolean isBuyerMaker = in.readBoolean();
                list.add(new Trade(time, (float) price, (float) qty, isBuyerMaker));
            } catch (EOFException eof) {
                break;
            }
        }
        return list;
    }

    private static List<Trade> parseTradeBatch(List<String> lines) {
        List<Trade> trades = new ArrayList<>(lines.size());
        for (String line : lines) {
            Trade trade = parseTradeLine(line);
            if (trade != null) {
                trades.add(trade);
            }
        }
        return trades;
    }

    public static Trade parseTradeLine(String line) {
        int p0 = line.indexOf(',');
        int p1 = line.indexOf(',', p0 + 1);
        int p2 = line.indexOf(',', p1 + 1);
        int p3 = line.indexOf(',', p2 + 1);
        int p4 = line.indexOf(',', p3 + 1);
        int p5 = line.indexOf(',', p4 + 1);
        if (p0 <= 0 || p1 <= 0 || p2 <= 0 || p3 <= 0 || p4 <= 0) {
            return null;
        }
        if (p5 == -1) p5 = line.length(); // Por si es la ultima columna

        return new Trade(
                Long.parseLong(line.substring(p3 + 1, p4)),        // time (col 4)
                Float.parseFloat(line.substring(p0 + 1, p1)),     // price (col 1)
                Float.parseFloat(line.substring(p1 + 1, p2)),     // qty (col 2)
                "true".equals(line.substring(p4 + 1, p5)) // isBuyerMaker (col 5)
        );
    }
}
