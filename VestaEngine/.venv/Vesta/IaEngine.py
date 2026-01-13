# IaEngine.py (train / predict separado, agrega depth option A)
import sys
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd
import json
import matplotlib.pyplot as plt

from FileRead import load_market_files, extract_trades_from_markets, trades_to_1m_candles, build_future_ranges

MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_EXT = ".keras"

# -----------------------
# Depth merge helper (Option A)
# -----------------------
def merge_depth_features_from_marketobjs(candles: pd.DataFrame, market_objs: list) -> pd.DataFrame:
    """
    Intenta extraer fields de profundidad desde los market_objs y unirlos por minute.
    Busca bajo cada market object una clave 'candles' o 'tickMarkers' que contenga
    items con openTime/timestamp y bidLiquidity/askLiquidity/orderbookImbalance/spread/mid_price.
    Si no encuentra nada, rellena columnas con 0.
    """
    if candles is None or candles.empty:
        return candles

    # prepare empty depth df indexed by candles.index
    depth_df = pd.DataFrame(index=candles.index)
    # default columns
    depth_df['bid_liq'] = 0.0
    depth_df['ask_liq'] = 0.0
    depth_df['depth_imbalance'] = 0.0
    depth_df['spread'] = 0.0
    depth_df['mid_price'] = 0.0

    # try to parse market objects: many possible shapes -> be defensive
    records = []
    for m in market_objs:
        # if m is dict and contains 'candles' array with depth info (your earlier example)
        if isinstance(m, dict):
            # case 1: m['candles'] present as array of per-minute dicts (openTime + bidLiquidity etc)
            if 'candles' in m and isinstance(m['candles'], list):
                for c in m['candles']:
                    # try to read openTime or timestamp
                    try:
                        ts = c.get('openTime') or c.get('timestamp') or c.get('time') or c.get('ts')
                        if ts is None:
                            continue
                        ts = int(ts)
                        # normalize to minute start
                        dt = pd.to_datetime(ts, unit='ms').floor('T')
                        rec = {
                            'dt': dt,
                            'bid_liq': float(c.get('bidLiquidity', 0.0)),
                            'ask_liq': float(c.get('askLiquidity', 0.0)),
                            'depth_imbalance': float(c.get('orderbookImbalance', 0.0)),
                            'spread': float(c.get('spread', 0.0)) if c.get('spread') is not None else 0.0,
                            'mid_price': float(c.get('mid_price', 0.0)) if c.get('mid_price') is not None else 0.0
                        }
                        records.append(rec)
                    except Exception:
                        continue

            # case 2: m contains 'tickMarkers' which may contain depth snapshots
            elif 'tickMarkers' in m and isinstance(m['tickMarkers'], list):
                for t in m['tickMarkers']:
                    # t may contain 'depth' or 'orderbook' or 'levels'
                    depth = None
                    if isinstance(t, dict):
                        if 'depth' in t:
                            depth = t.get('depth')
                        elif 'orderbook' in t:
                            depth = t.get('orderbook')
                        # some variants have bids/asks directly
                        bids = t.get('bids')
                        asks = t.get('asks')
                        ts = t.get('timestamp') or t.get('time') or t.get('ts')
                        if ts is None:
                            # try nested
                            ts = t.get('openTime')
                        if ts is None:
                            continue
                        try:
                            dt = pd.to_datetime(int(ts), unit='ms').floor('T')
                        except Exception:
                            continue

                        # compute simple aggregates if bids/asks present
                        try:
                            bid_list = bids or (depth.get('bids') if depth else None)
                            ask_list = asks or (depth.get('asks') if depth else None)
                            if bid_list and ask_list:
                                # assume arrays of [price, qty]
                                bid_qty = sum(float(x[1]) for x in bid_list[:10])
                                ask_qty = sum(float(x[1]) for x in ask_list[:10])
                                best_bid = float(bid_list[0][0])
                                best_ask = float(ask_list[0][0])
                                spread = best_ask - best_bid
                                mid = (best_ask + best_bid) / 2.0
                                rec = {
                                    'dt': dt,
                                    'bid_liq': bid_qty,
                                    'ask_liq': ask_qty,
                                    'depth_imbalance': (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-12),
                                    'spread': spread,
                                    'mid_price': mid
                                }
                                records.append(rec)
                        except Exception:
                            continue
            else:
                # fallback: search for any list value containing dicts with bids/asks
                for k, v in m.items():
                    if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
                        # examine first element for bids/asks key
                        maybe = v[0]
                        if 'bids' in maybe and 'asks' in maybe:
                            # iterate v elements
                            for item in v:
                                ts = item.get('timestamp') or item.get('time') or item.get('openTime') or item.get('ts')
                                if ts is None:
                                    continue
                                try:
                                    dt = pd.to_datetime(int(ts), unit='ms').floor('T')
                                except Exception:
                                    continue
                                bids = item.get('bids')
                                asks = item.get('asks')
                                try:
                                    bid_qty = sum(float(x[1]) for x in bids[:10])
                                    ask_qty = sum(float(x[1]) for x in asks[:10])
                                    best_bid = float(bids[0][0])
                                    best_ask = float(asks[0][0])
                                    rec = {
                                        'dt': dt,
                                        'bid_liq': bid_qty,
                                        'ask_liq': ask_qty,
                                        'depth_imbalance': (bid_qty - ask_qty) / (bid_qty + ask_qty + 1e-12),
                                        'spread': best_ask - best_bid,
                                        'mid_price': (best_ask + best_bid) / 2.0
                                    }
                                    records.append(rec)
                                except Exception:
                                    continue

    if len(records) == 0:
        # nothing found, return candles with zero depth features
        return candles.assign(
            bid_liq=0.0, ask_liq=0.0, depth_imbalance=0.0, spread=0.0, mid_price=0.0
        )

    depth_rec_df = pd.DataFrame(records).drop_duplicates(subset=['dt']).set_index('dt')
    # reindex to candles index and forward/back fill small gaps
    depth_rec_df = depth_rec_df.reindex(candles.index).ffill().fillna(0.0)
    # merge
    merged = pd.concat([candles, depth_rec_df[['bid_liq','ask_liq','depth_imbalance','spread','mid_price']]], axis=1)
    # fill any remaining NaNs
    merged[['bid_liq','ask_liq','depth_imbalance','spread','mid_price']] = merged[['bid_liq','ask_liq','depth_imbalance','spread','mid_price']].fillna(0.0)
    return merged

# -----------------------
# Build sequences (predict open & close next minute)
# -----------------------
def build_sequences_multi(candles, features, lookback=5, predict_cols=['open','close']):
    arr = candles[features].values.astype(np.float32)
    n = len(arr)
    X_list, y_list = [], []
    for i in range(lookback, n):
        X = arr[i-lookback:i]
        y = candles.iloc[i][predict_cols].values.astype(np.float32)
        if np.isnan(y).any():
            continue
        X_list.append(X)
        y_list.append(y)
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# -----------------------
# Train model
# -----------------------
def train_model(symbol: str,
                lookback: int = 5,
                epochs: int = 100,
                batch_size: int = 64):

    print("Training for symbol:", symbol)

    market_objs = load_market_files(symbol)
    trades_df = extract_trades_from_markets(market_objs)
    if trades_df is None or trades_df.empty:
        raise RuntimeError("No trades found to train on")

    candles = trades_to_1m_candles(trades_df)
    if candles is None or candles.empty:
        raise RuntimeError("No candles generated")

    candles = merge_depth_features_from_marketobjs(candles, market_objs)

    features = [
        'close','quote_volume','delta_usdt','imbalance','vwap',
        'bid_liq','ask_liq','depth_imbalance','spread','mid_price'
    ]

    predict_cols = ['open', 'close']

    df = candles.dropna(subset=features + predict_cols)
    if df.empty or len(df) < lookback + 10:
        raise RuntimeError("Not enough rows to train")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    df_scaled = df.copy()
    df_scaled[features] = scaler_X.fit_transform(df[features])
    df_scaled[predict_cols] = scaler_y.fit_transform(df[predict_cols])

    X, y = build_sequences_multi(
        df_scaled,
        features,
        lookback=lookback,
        predict_cols=predict_cols
    )

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, len(features))),
        tf.keras.layers.LSTM(1024),
        tf.keras.layers.Dense(1024, activation='relu'),
        tf.keras.layers.Dense(len(predict_cols))
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    model.summary()

    # ✅ GUARDAMOS EL HISTORIAL
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # ✅ GUARDADO CORRECTO
    model_path = MODEL_DIR / f"{symbol}_model{MODEL_EXT}"
    scaler_X_path = MODEL_DIR / f"{symbol}_scaler_X.pkl"
    scaler_y_path = MODEL_DIR / f"{symbol}_scaler_y.pkl"

    model.save(model_path)
    joblib.dump(scaler_X, scaler_X_path)
    joblib.dump(scaler_y, scaler_y_path)

    # -----------------------
    # 📊 GRAFICA DE CERTEZA (LOSS)
    # -----------------------
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title(f'{symbol} — Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    print("Saved model to:", model_path)
    print("Saved scalers to:", scaler_X_path, scaler_y_path)

# -----------------------
# Predict next candle body (open, close)
# -----------------------
def predict_next_body(symbol: str, lookback: int = 5):
    """
    Carga modelo y scaler, obtiene las últimas velas (recomputando desde trades)
    y devuelve (pred_open, pred_close) en precio real (no escalado).
    """
    model_path = MODEL_DIR / f"{symbol}_model{MODEL_EXT}"
    scaler_X_path = MODEL_DIR / f"{symbol}_scaler_X.pkl"
    scaler_y_path = MODEL_DIR / f"{symbol}_scaler_y.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Train first.")

    model = tf.keras.models.load_model(str(model_path))
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    market_objs = load_market_files(symbol)
    trades_df = extract_trades_from_markets(market_objs)
    if trades_df is None or trades_df.empty:
        raise RuntimeError("No trades to predict from")

    candles = trades_to_1m_candles(trades_df)
    candles = merge_depth_features_from_marketobjs(candles, market_objs)

    features = ['close','quote_volume','delta_usdt','imbalance','vwap',
                'bid_liq','ask_liq','depth_imbalance','spread','mid_price']

    # prepare last lookback window
    df = candles.dropna(subset=features)
    if len(df) < lookback:
        raise RuntimeError("Not enough candles for prediction")

    last_window = df[features].values[-lookback:]
    last_window_scaled = scaler_X.transform(last_window)
    X_input = np.expand_dims(last_window_scaled, axis=0).astype(np.float32)  # shape (1, lookback, n_features)

    pred_scaled = model.predict(X_input)  # shape (1,2)
    pred_real = scaler_y.inverse_transform(pred_scaled)  # back to (open, close) prices

    pred_open, pred_close = float(pred_real[0, 0]), float(pred_real[0, 1])
    return pred_open, pred_close

# -----------------------
# CLI usage
# -----------------------
def print_usage():
    print("Usage:")
    print(" python IaEngine.py train SYMBOL [lookback] [epochs]")
    print(" python IaEngine.py predict SYMBOL [lookback]")

if __name__ == "__main__":
    print("TF:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    cmd = sys.argv[1].lower()
    sym = sys.argv[2].upper()

    if cmd == "train":

        lookback = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        epochs = int(sys.argv[4]) if len(sys.argv) >= 5 else 50
        train_model(sym, lookback=lookback, epochs=epochs, batch_size=64)
    elif cmd == "predict":
        lookback = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        o, c = predict_next_body(sym, lookback=lookback)
        print(f"Predicted next open: {o:.6f}   next close: {c:.6f}")
    else:
        print_usage()

