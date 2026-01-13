import time
import numpy as np
import pandas as pd
import ccxt
import requests


# ---------- Exchange ----------
def make_exchange():
    ex = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })
    return ex


# ---------- Data ----------
def fetch_usdt_markets(exchange):
    markets = exchange.load_markets()
    symbols = []
    for s, m in markets.items():
        if m.get("active") and m.get("spot") and s.endswith("/USDT"):
            base = s.split("/")[0]
            if base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                continue
            symbols.append(s)
    return symbols


def fetch_ohlcv_df(exchange, symbol, timeframe="1m", limit=720):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return None

    if not ohlcv:
        return None

    df = pd.DataFrame(ohlcv, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


# ---------- Indicators ----------
def zscore(series: pd.Series, window: int):
    m = series.rolling(window).mean()
    s = series.rolling(window).std(ddof=0)
    z = (series - m) / (s.replace(0, np.nan))
    return z


def rsi(series: pd.Series, period: int = 14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out


def atr(df: pd.DataFrame, period: int = 14):
    """
    ATR (Average True Range)
    """
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    out = tr.ewm(alpha=1/period, adjust=False).mean()
    return out


# ---------- Features / Scoring ----------
def compute_features(df: pd.DataFrame, timeframe="1m"):
    df = df.copy()

    # returns
    df["ret_1"] = df["close"].pct_change(1)
    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)

    # realized volatility proxy (son 60 bar)
    df["rv_60"] = df["ret_1"].rolling(60).std(ddof=0)

    # Volume anomaly baseline
    vol_window = 120
    if timeframe == "1h":
        vol_window = 72
    df["vol_z"] = zscore(df["volume"], window=vol_window)

    # volume ratio (göreli hacim)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / (df["vol_ma20"].replace(0, np.nan))

    # Trend: EMA fast/slow
    df["ema_20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema_50"] = df["close"].ewm(span=50, adjust=False).mean()
    df["trend"] = np.where(df["ema_20"] > df["ema_50"], 1, -1)  # 1 bullish, -1 bearish

    # RSI
    df["rsi_14"] = rsi(df["close"], period=14)

    # ATR
    df["atr_14"] = atr(df, period=14)
    df["atr_pct"] = df["atr_14"] / (df["close"].replace(0, np.nan))

    # breakout vs last 60 bars high
    df["hh_60"] = df["high"].rolling(60).max()
    df["breakout"] = (df["close"] > df["hh_60"].shift(1)).astype(int)

    # giveback heuristic
    if timeframe == "1h":
        spike_thr = 0.02
        giveback_thr = -0.015
    else:
        spike_thr = 0.01
        giveback_thr = -0.008

    df["spike"] = (df["ret_1"] > spike_thr).astype(int)
    df["giveback"] = ((df["ret_1"] < giveback_thr) & (df["spike"].shift(1) == 1)).astype(int)

    return df


def infer_direction(last_row: pd.Series):
    trend = int(last_row.get("trend", 0))
    r5 = last_row.get("ret_5", np.nan)
    r15 = last_row.get("ret_15", np.nan)

    if pd.notna(r5) and pd.notna(r15):
        if trend == 1 and r5 > 0 and r15 > 0:
            return "LONG"
        if trend == -1 and r5 < 0 and r15 < 0:
            return "SHORT"
    return "NEUTRAL"


def score_latest(df: pd.DataFrame):
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else None

    vol_z = last.get("vol_z", np.nan)
    vol_z_prev = prev.get("vol_z", np.nan) if prev is not None else np.nan
    vol_ratio = last.get("vol_ratio", np.nan)

    r5 = last.get("ret_5", np.nan)
    r15 = last.get("ret_15", np.nan)
    rv_60 = last.get("rv_60", np.nan)

    breakout = int(last.get("breakout", 0))
    giveback = int(last.get("giveback", 0))
    trend = int(last.get("trend", 0))
    rsi14 = last.get("rsi_14", np.nan)

    atr_14 = last.get("atr_14", np.nan)
    atr_pct = last.get("atr_pct", np.nan)

    direction = infer_direction(last)

    vol_component = np.clip(vol_z if pd.notna(vol_z) else 0, -2, 8)
    mom_component = 100 * (0.6 * (r5 if pd.notna(r5) else 0) + 0.4 * (r15 if pd.notna(r15) else 0))
    breakout_component = 1.5 * breakout
    penalty = 2.5 * giveback

    trend_bonus = 0.5 if direction == "LONG" else (-0.5 if direction == "SHORT" else 0.0)
    score = (1.2 * vol_component) + (0.8 * mom_component) + breakout_component - penalty + trend_bonus

    return {
        "score": float(score),
        "vol_z": float(vol_z) if pd.notna(vol_z) else None,
        "vol_z_prev": float(vol_z_prev) if pd.notna(vol_z_prev) else None,
        "vol_ratio": float(vol_ratio) if pd.notna(vol_ratio) else None,
        "ret_5": float(r5) if pd.notna(r5) else None,
        "ret_15": float(r15) if pd.notna(r15) else None,
        "rv_60": float(rv_60) if pd.notna(rv_60) else None,
        "atr_14": float(atr_14) if pd.notna(atr_14) else None,
        "atr_pct": float(atr_pct) if pd.notna(atr_pct) else None,
        "trend": int(trend),
        "rsi_14": float(rsi14) if pd.notna(rsi14) else None,
        "direction": direction,
        "breakout": breakout,
        "giveback": giveback,
        "close": float(last["close"]),
        "ts": str(last["ts"]),
    }


def scan(exchange, symbols, timeframe="1m", limit=720, min_price=1e-7):
    rows = []
    sleep_s = max(exchange.rateLimit / 1000.0, 0.05)

    min_bars = 200
    if timeframe == "1h":
        min_bars = 150

    for sym in symbols:
        df = fetch_ohlcv_df(exchange, sym, timeframe=timeframe, limit=limit)
        if df is None or len(df) < min_bars:
            time.sleep(sleep_s)
            continue

        if float(df["close"].iloc[-1]) <= min_price:
            time.sleep(sleep_s)
            continue

        df = compute_features(df, timeframe=timeframe)
        s = score_latest(df)

        if s["vol_z"] is None:
            time.sleep(sleep_s)
            continue

        rows.append({"symbol": sym, **s})
        time.sleep(sleep_s)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out


# ---------- Universe selection ----------
def get_top_usdt_symbols_by_quote_volume(limit=200, min_quote_usdt=5_000_000):
    url = "https://api.binance.com/api/v3/ticker/24hr"

    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    rows = []
    for it in data:
        sym = it.get("symbol", "")
        if not sym.endswith("USDT"):
            continue
        if sym.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")):
            continue

        try:
            quote_vol = float(it.get("quoteVolume", 0.0))
            last_price = float(it.get("lastPrice", 0.0))
        except Exception:
            continue

        if quote_vol < float(min_quote_usdt):
            continue
        if last_price <= 0:
            continue

        ccxt_symbol = sym[:-4] + "/USDT"
        rows.append((ccxt_symbol, quote_vol))

    rows.sort(key=lambda x: x[1], reverse=True)
    return [r[0] for r in rows[: int(limit)]]
