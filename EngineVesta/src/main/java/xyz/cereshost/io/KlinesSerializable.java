package xyz.cereshost.io;

import xyz.cereshost.common.market.CandleSimple;
import xyz.cereshost.common.market.Volumen;
import xyz.cereshost.engine.VestaEngine;

import java.io.DataInputStream;
import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.concurrent.FutureTask;

import static xyz.cereshost.io.IOMarket.*;

public class KlinesSerializable implements SerializableBin<CandleSimple>, SerializableCSV<CandleSimple> {

    @Override
    public void submitBatch(Deque<FutureTask<List<CandleSimple>>> tasks, List<String> batch) {
        List<String> batchCopy = new ArrayList<>(batch);
        FutureTask<List<CandleSimple>> task = new FutureTask<>(() -> parseKlineBatch(batchCopy));
        tasks.addLast(task);
        VestaEngine.EXECUTOR_AUXILIAR_BUILD.execute(task);
    }

    @Override
    public void writeBin(File zipFile, Deque<CandleSimple> candles) throws IOException {
        String entryName = binEntryName(zipFile);
        rewriteZipWithBinEntry(zipFile, entryName, out -> {
            out.writeInt(KLINE_BIN_MAGIC);
            out.writeInt(BIN_VERSION);
            for (CandleSimple candle : candles) {
                Volumen vol = candle.volumen();
                out.writeLong(candle.openTime());
                out.writeDouble(candle.open());
                out.writeDouble(candle.high());
                out.writeDouble(candle.low());
                out.writeDouble(candle.close());
                out.writeDouble(vol.quoteVolume());
                out.writeDouble(vol.baseVolume());
                out.writeDouble(vol.takerBuyQuoteVolume());
                out.writeDouble(vol.sellQuoteVolume());
                out.writeDouble(vol.deltaUSDT());
                out.writeDouble(vol.buyRatio());
            }
        });
    }

    @Override
    public Deque<CandleSimple> readBin(DataInputStream in) throws IOException {
        int magic = in.readInt();
        if (magic != KLINE_BIN_MAGIC) {
            return null;
        }
        int version = in.readInt();
        if (version != BIN_VERSION) {
            return null;
        }
        Deque<CandleSimple> list = new ArrayDeque<>(44_000);
        while (true) {
            try {
                long openTime = in.readLong();
                double open = in.readDouble();
                double high = in.readDouble();
                double low = in.readDouble();
                double close = in.readDouble();
                double quoteVolume = in.readDouble();
                double baseVolume = in.readDouble();
                double takerBuyQuoteVolume = in.readDouble();
                double sellQuoteVolume = in.readDouble();
                double deltaUSDT = in.readDouble();
                double buyRatio = in.readDouble();
                list.add(new CandleSimple(
                        openTime,
                        open,
                        high,
                        low,
                        close,
                        new Volumen(quoteVolume, baseVolume, takerBuyQuoteVolume, sellQuoteVolume, deltaUSDT, buyRatio)
                ));
            } catch (EOFException eof) {
                break;
            }
        }
        return list;
    }

    private static List<CandleSimple> parseKlineBatch(List<String> lines) {
        List<CandleSimple> candles = new ArrayList<>(lines.size());
        for (String line : lines) {
            CandleSimple candle = parseKlineLine(line);
            candles.add(candle);
        }
        return candles;
    }

    public static CandleSimple parseKlineLine(String line) {
        String[] p = line.split(",");
        double quoteVolume = Double.parseDouble(p[7]);
        double takerBuyQuoteVolume = Double.parseDouble(p[10]);

        return new CandleSimple(
                Long.parseLong(p[0]), // Open time
                Double.parseDouble(p[1]), // Open
                Double.parseDouble(p[2]), // High
                Double.parseDouble(p[3]), // Low
                Double.parseDouble(p[4]), // Close
                new Volumen(quoteVolume, Double.parseDouble(p[5]), takerBuyQuoteVolume,
                        quoteVolume - takerBuyQuoteVolume,
                        takerBuyQuoteVolume - (quoteVolume - takerBuyQuoteVolume),
                        (quoteVolume == 0) ? 0 : takerBuyQuoteVolume / quoteVolume)
        );
    }
}
