# FileRead.py  (corregido)
from pathlib import Path
import json
import pandas as pd
import numpy as np
from typing import List

def find_data_dir(max_up: int = 6) -> Path:
    """
    Busca una carpeta llamada 'data' subiendo desde el directorio del script.
    Lanza FileNotFoundError si no la encuentra dentro de max_up niveles.
    """
    p = Path(__file__).resolve().parent
    for _ in range(max_up):
        candidate = p / "data"
        if candidate.exists() and candidate.is_dir():
            return candidate
        p = p.parent
    raise FileNotFoundError(f"No se encontró la carpeta 'data' subiendo {max_up} niveles desde {Path(__file__).resolve().parent}")

# -----------------------
# Helpers: file loading
# -----------------------
def load_market_files(symbol: str, data_dir: Path = None) -> List[dict]:
    """
    Busca archivos JSON bajo data_dir/symbol/YYYY-MM-DD/*.json
    y devuelve lista de objetos JSON ordenados cronológicamente.
    Si data_dir es None, intenta buscar 'data' subiendo niveles.
    """
    if data_dir is None:
        DATA_DIR = find_data_dir()
    else:
        DATA_DIR = Path(data_dir)

    symbol_path = DATA_DIR / symbol
    print("DATA DIR:", DATA_DIR.resolve())
    if not symbol_path.exists():
        raise FileNotFoundError(symbol_path)

    objs = []
    for day_dir in sorted([p for p in symbol_path.iterdir() if p.is_dir()]):
        for hour_file in sorted(day_dir.glob("*.json")):
            with open(hour_file, "r", encoding="utf8") as f:
                try:
                    obj = json.load(f)
                except Exception as e:
                    print("Error leyendo", hour_file, e)
                    continue
                objs.append(obj)
    return objs

# -----------------------
# Extract trades
# -----------------------
def extract_trades_from_markets(markets: List[dict]) -> pd.DataFrame:
    """
    Devuelve DataFrame con columnas:
      time(ms), price, qty, quoteQty, isBuyerMaker, id
    """
    records = []
    for m in markets:
        trades = None
        if isinstance(m, dict):
            trades = m.get("trades") or m.get("trade") or m.get("Trades") or m.get("aggTrades") or m.get("aggrTrades")
        elif isinstance(m, list):
            # el archivo contiene directamente la lista de trades
            trades = m

        if not trades:
            continue

        for t in trades:
            try:
                # si t no es dict (por alguna razón), intentar convertir
                if not isinstance(t, dict):
                    continue
                trade_id = t.get("id") or t.get("tradeId") or None
                price = float(t.get("price"))
                qty = float(t.get("qty"))
                quoteQty = float(t.get("quoteQty")) if t.get("quoteQty") is not None else price * qty
                ts = int(t.get("time") or t.get("timestamp") or t.get("ts") or 0)
                isBuyerMaker = bool(t.get("isBuyerMaker"))
            except Exception:
                # salta trades malformados
                continue

            records.append({
                "time": ts,
                "price": price,
                "qty": qty,
                "quoteQty": quoteQty,
                "isBuyerMaker": isBuyerMaker,
                "id": trade_id
            })

    df = pd.DataFrame(records)
    if df.empty:
        return df
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# -----------------------
# Trades -> Candles 1m
# -----------------------
def trades_to_1m_candles(trades_df: pd.DataFrame, tz=None) -> pd.DataFrame:
    """
    Convierte trades DataFrame a velas de 1 minuto con features:
     open, high, low, close, volume_base (qty), quote_volume (USDT),
     buy_quote_volume, sell_quote_volume, delta_usdt, trade_count, vwap, imbalance
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame()

    ts_min = trades_df['time'].min()
    ts_max = trades_df['time'].max()
    if ts_max > 1e12:
        unit = 'ms'
    elif ts_max > 1e9:
        unit = 's'
    else:
        raise ValueError(f"timestamps inválidos: min={ts_min}, max={ts_max}")

    trades_df = trades_df.copy()
    trades_df['dt'] = pd.to_datetime(trades_df['time'], unit=unit)
    if tz:
        trades_df['dt'] = trades_df['dt'].dt.tz_localize('UTC').dt.tz_convert(tz)

    trades_df = trades_df.set_index('dt')

    # resample con '1min'
    ohlc = trades_df['price'].resample('1min').ohlc()
    base_vol = trades_df['qty'].resample('1min').sum()
    quote_vol = trades_df['quoteQty'].resample('1min').sum()
    trade_count = trades_df['price'].resample('1min').count()

    buy_quote = trades_df.loc[~trades_df['isBuyerMaker'], 'quoteQty'].resample('1min').sum().fillna(0.0)
    sell_quote = trades_df.loc[trades_df['isBuyerMaker'], 'quoteQty'].resample('1min').sum().fillna(0.0)

    vwap_num = (trades_df['price'] * trades_df['quoteQty']).resample('1min').sum()
    vwap = vwap_num / (quote_vol.replace(0, np.nan))
    vwap = vwap.ffill().fillna(ohlc['close'])

    candles = pd.concat([
        ohlc,
        base_vol.rename('volume_base'),
        quote_vol.rename('quote_volume'),
        buy_quote.rename('buy_quote_volume'),
        sell_quote.rename('sell_quote_volume'),
        trade_count.rename('trade_count'),
        vwap.rename('vwap')
    ], axis=1)

    # fillna sin chained assignment
    candles['close'] = candles['close'].ffill()
    candles[['open','high','low','vwap']] = candles[['open','high','low','vwap']].fillna(candles['close'])
    candles[['volume_base','quote_volume','buy_quote_volume','sell_quote_volume']] = \
        candles[['volume_base','quote_volume','buy_quote_volume','sell_quote_volume']].fillna(0.0)
    candles['trade_count'] = candles['trade_count'].fillna(0).astype(int)

    candles['delta_usdt'] = candles['buy_quote_volume'] - candles['sell_quote_volume']
    candles['imbalance'] = candles['buy_quote_volume'] / (candles['buy_quote_volume'] + candles['sell_quote_volume'] + 1e-12)

    candles = candles.sort_index()
    return candles

# alias por compatibilidad
def trades_to_candles_safe(trades_df: pd.DataFrame, tz=None) -> pd.DataFrame:
    return trades_to_1m_candles(trades_df, tz=tz)

# -----------------------
# Build future range labels (next N minutes)
# -----------------------
def build_future_ranges(candles: pd.DataFrame, n_future: int = 5):
    if candles is None or candles.empty:
        return candles
    df = candles.copy()
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    N = len(df)
    for k in range(1, n_future + 1):
        up = np.full(N, np.nan, dtype=float)
        down = np.full(N, np.nan, dtype=float)
        for i in range(N):
            end = min(N, i + k + 1)
            if i + 1 >= N:
                up[i] = np.nan
                down[i] = np.nan
            else:
                max_high = highs[i+1:end].max()
                min_low = lows[i+1:end].min()
                up[i] = (max_high - closes[i]) / closes[i]
                down[i] = (closes[i] - min_low) / closes[i]
        df[f'future_up_{k}'] = up
        df[f'future_down_{k}'] = down
    return df

# -----------------------
# Dataset builder (sliding windows)
# -----------------------
def build_dataset_from_candles(candles: pd.DataFrame, lookback: int = 10, n_future: int = 5,
                               features = None):
    if features is None:
        features = ['close', 'quote_volume', 'delta_usdt', 'imbalance', 'vwap']

    if candles is None or candles.empty:
        return np.empty((0,)), np.empty((0,))

    df = candles.copy().dropna(subset=['close'])
    df = df.reset_index(drop=False)
    n = len(df)
    X_list = []
    y_list = []

    for i in range(lookback, n - n_future):
        window = df.iloc[i - lookback:i]
        X = window[features].values.astype(float)

        future_cols = [f'future_up_{k}' for k in range(1, n_future+1)] + \
                      [f'future_down_{k}' for k in range(1, n_future+1)]

        y_vals = df.loc[i, future_cols].values.astype(float)
        if np.isnan(y_vals).any():
            continue

        X_list.append(X)
        y_list.append(y_vals)

    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# Example quick run guard (kept similar to tu versión original)
if __name__ == "__main__":
    SYMBOL = "XRPUSDT"
    market_objs = load_market_files(SYMBOL)
    print("Loaded JSON files:", len(market_objs))
    trades_df = extract_trades_from_markets(market_objs)
    print("Total trades:", len(trades_df))
    candles = trades_to_1m_candles(trades_df)
    print("Candles:", candles.shape)
    print(candles.head())
    candles_with_targets = build_future_ranges(candles, n_future=5)
    X, y = build_dataset_from_candles(candles_with_targets, lookback=10, n_future=5)
    print("X shape:", X.shape, "y shape:", y.shape)
