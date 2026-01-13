import streamlit as st
import pandas as pd
import os
import time
import requests
from datetime import datetime, timezone

from streamlit_autorefresh import st_autorefresh

from scanner import (
    make_exchange,
    scan,
    fetch_ohlcv_df,
    compute_features,
    get_top_usdt_symbols_by_quote_volume
)

st.set_page_config(page_title="Crypto Anomali Tarayıcı (Ücretsiz)", layout="wide")

st.title("🚨 Crypto Anomali / Momentum Erken Uyarı (Ücretsiz)")
st.caption("Bu araç 'pump' garantisi vermez; olağandışı hacim + momentum hareketlerini erken fark etmek için uyarı üretir.")

state = st.session_state
if "watchlist" not in state:
    state["watchlist"] = []
if "last_alert_ts" not in state:
    state["last_alert_ts"] = {}
if "last_tg_ts" not in state:
    state["last_tg_ts"] = {}

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Ayarlar")
    timeframe = st.selectbox("Zaman Dilimi", ["1m", "5m", "1h"], index=0)

    default_limit = 720 if timeframe in ["1m", "5m"] else 500
    limit = st.slider("Geçmiş Bar Sayısı", min_value=200, max_value=1000, value=int(default_limit), step=50)
    topn = st.slider("Gösterilecek İlk N", 5, 50, 20, 5)

    st.divider()
    st.subheader("Likidite filtresi")
    top_symbols = st.slider("Taranacak coin sayısı (en likit)", 50, 400, 200, 25)
    min_quote_usdt = st.number_input("Min 24h Hacim (USDT)", value=5_000_000, step=500_000)

    st.divider()
    st.subheader("Alarm koşulları")
    vol_z_thr = st.number_input("Hacim Z eşiği (vol_z)", value=3.0, step=0.5)
    mom_label = "Momentum eşiği (%) [5 bar]" if timeframe == "1h" else "Momentum eşiği (%) [5 bar]"
    ret5_thr_pct = st.number_input(mom_label, value=0.8, step=0.1)
    cooldown_min = st.number_input("Cooldown (dakika)", value=10, step=1)

    st.divider()
    st.subheader("False Breakout filtresi (Spam azaltır)")
    max_spread_pct = st.number_input("Max spread (%)", value=0.40, step=0.05)
    rsi_extreme_long = st.number_input("LONG için RSI extreme üstü", value=85.0, step=1.0)
    rsi_extreme_short = st.number_input("SHORT için RSI extreme altı", value=15.0, step=1.0)
    vol_fade_check = st.toggle("Breakout'ta hacim sönmesi filtresi", value=True)

    st.divider()
    st.subheader("Otomatik yenileme")
    auto_refresh = st.toggle("Auto-refresh (tarama)", value=False)
    refresh_sec = st.slider("Yenileme aralığı (sn)", 15, 300, 60, 5)

    st.divider()
    st.subheader("📨 Telegram (Sadece Ciddi Bildirim)")
    tg_enabled = st.toggle("Telegram bildirimlerini aç", value=False)

    tg_token_default = st.secrets.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat_default = st.secrets.get("TELEGRAM_CHAT_ID", "")

    tg_token = st.text_input("Bot Token (secrets yoksa gir)", value=tg_token_default, type="password")
    tg_chat_id = st.text_input("Chat ID (secrets yoksa gir)", value=str(tg_chat_default))

    tg_min_quality = st.slider("Telegram için minimum kalite", 0, 100, 75, 1)
    tg_min_conf = st.slider("Telegram için minimum confidence", 0, 100, 75, 1)
    tg_breakout_only = st.toggle("Sadece Breakout (kırılım) gönder", value=True)
    tg_only_long_short = st.toggle("Sadece LONG/SHORT gönder (NEUTRAL hariç)", value=True)
    tg_skip_late = st.toggle("LATE phase gönderme (riskli)", value=True)
    tg_cooldown_min = st.number_input("Telegram cooldown (dakika)", value=20, step=1)
    tg_max_items = st.slider("Telegram mesajında max coin", 1, 10, 6, 1)

    st.divider()
    run_btn = st.button("Taramayı Başlat / Yenile")

if auto_refresh:
    st_autorefresh(interval=int(refresh_sec) * 1000, key="auto_refresh_counter")

# ---------------- Helpers ----------------
@st.cache_data(ttl=300)
def get_symbols_cached(limit_: int, min_quote_: float):
    return get_top_usdt_symbols_by_quote_volume(limit=limit_, min_quote_usdt=min_quote_)

def run_scan():
    symbols = get_symbols_cached(int(top_symbols), float(min_quote_usdt))
    ex = make_exchange()
    df = scan(ex, symbols, timeframe=timeframe, limit=int(limit))
    return df

def get_quote_volume_map_usdt():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    m = {}
    try:
        data = requests.get(url, timeout=15).json()
        for it in data:
            sym = it.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            if sym.endswith(("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")):
                continue
            qv = float(it.get("quoteVolume", 0.0))
            ccxt_symbol = sym[:-4] + "/USDT"
            m[ccxt_symbol] = qv
    except Exception:
        pass
    return m

def get_spread_map(exchange, symbols, max_n=40):
    out = {}
    sleep_s = max(exchange.rateLimit / 1000.0, 0.05)
    for sym in symbols[:max_n]:
        try:
            t = exchange.fetch_ticker(sym)
            bid = t.get("bid", None)
            ask = t.get("ask", None)
            if bid and ask and bid > 0 and ask > 0:
                mid = (bid + ask) / 2
                spread_pct = (ask - bid) / mid * 100
                out[sym] = float(spread_pct)
            else:
                out[sym] = None
            time.sleep(sleep_s)
        except Exception:
            out[sym] = None
    return out

def quality_score_row(row, spread_pct, quote_vol):
    if spread_pct is None or pd.isna(spread_pct):
        spread_score = 0.5
    else:
        spread_score = 1.0 - min(max(float(spread_pct) - 0.20, 0) / 0.40, 1.0)

    if quote_vol is None or pd.isna(quote_vol) or float(quote_vol) <= 0:
        liq_score = 0.3
    else:
        import math
        v = max(float(quote_vol), 1.0)
        lo = math.log10(5_000_000)
        hi = math.log10(100_000_000)
        x = (math.log10(v) - lo) / (hi - lo)
        liq_score = min(max(x, 0.0), 1.0)

    rv = row.get("rv_60", None)
    if rv is None or pd.isna(rv):
        vol_score = 0.5
    else:
        target = 0.008 if timeframe in ["1m", "5m"] else 0.02
        vol_score = 1.0 - min(max(float(rv) / target, 0.0), 1.0)

    breakout_bonus = 1.0 if int(row.get("breakout", 0)) == 1 else 0.0
    giveback_pen = 1.0 if int(row.get("giveback", 0)) == 1 else 0.0

    score = 100.0 * (
        0.35 * spread_score +
        0.35 * liq_score +
        0.20 * vol_score +
        0.10 * breakout_bonus
    ) - (15.0 * giveback_pen)

    return float(min(max(score, 0.0), 100.0))

def add_quality(df, quote_map, spread_map):
    df = df.copy()
    df["quote_vol_usdt"] = df["symbol"].map(quote_map)
    df["spread_pct"] = df["symbol"].map(spread_map)
    df["quality"] = df.apply(lambda r: quality_score_row(r, r["spread_pct"], r["quote_vol_usdt"]), axis=1)
    return df

def add_confidence(df):
    """
    Confidence: 0-100
    - quality (0-100)
    - score_rank (0-100)
    """
    df = df.copy()
    df["score_rank"] = df["score"].rank(pct=True) * 100.0
    df["confidence"] = (0.60 * df["quality"] + 0.40 * df["score_rank"]).clip(0, 100)
    return df

def pump_phase_row(row, vol_thr, mom_thr):
    """
    Phase: EARLY / MID / LATE / NONE
    """
    direction = str(row.get("direction", "NEUTRAL"))
    vol_z = row.get("vol_z", None)
    breakout = int(row.get("breakout", 0))
    giveback = int(row.get("giveback", 0))
    rsi14 = row.get("rsi_14", None)
    ret5 = row.get("ret_5", None)

    if vol_z is None or pd.isna(vol_z) or ret5 is None or pd.isna(ret5):
        return "NONE"

    vol_z = float(vol_z)
    ret5 = float(ret5)
    rsi = float(rsi14) if (rsi14 is not None and pd.notna(rsi14)) else None

    # Late: aşırı RSI veya giveback
    if giveback == 1:
        return "LATE"
    if rsi is not None:
        if direction == "LONG" and rsi >= 80:
            return "LATE"
        if direction == "SHORT" and rsi <= 20:
            return "LATE"

    # Mid: breakout + güçlü hacim + momentum
    if breakout == 1 and vol_z >= vol_thr:
        if direction == "LONG" and ret5 >= mom_thr:
            return "MID"
        if direction == "SHORT" and ret5 <= -mom_thr:
            return "MID"

    # Early: hacim anomali + momentum var ama breakout yok
    if breakout == 0 and vol_z >= vol_thr:
        if direction == "LONG" and ret5 >= (mom_thr * 0.5):
            return "EARLY"
        if direction == "SHORT" and ret5 <= -(mom_thr * 0.5):
            return "EARLY"

    return "NONE"

def false_breakout_filter(row):
    """
    False breakout / risk filtresi: (is_filtered, reason)
    """
    reasons = []

    spread = row.get("spread_pct", None)
    if spread is not None and pd.notna(spread):
        if float(spread) > float(max_spread_pct):
            reasons.append(f"Spread>{max_spread_pct:.2f}%")

    direction = str(row.get("direction", "NEUTRAL"))
    rsi14 = row.get("rsi_14", None)
    if rsi14 is not None and pd.notna(rsi14):
        rsi14 = float(rsi14)
        if direction == "LONG" and rsi14 >= float(rsi_extreme_long):
            reasons.append(f"RSI≥{float(rsi_extreme_long):.0f}")
        if direction == "SHORT" and rsi14 <= float(rsi_extreme_short):
            reasons.append(f"RSI≤{float(rsi_extreme_short):.0f}")

    if vol_fade_check:
        breakout = int(row.get("breakout", 0))
        vz = row.get("vol_z", None)
        vz_prev = row.get("vol_z_prev", None)
        if breakout == 1 and vz is not None and pd.notna(vz):
            vz = float(vz)
            if vz_prev is not None and pd.notna(vz_prev):
                vz_prev = float(vz_prev)
                # breakout oldu ama vol_z düşüyorsa “sönme” riski
                if vz < vz_prev and vz < 2.5:
                    reasons.append("Hacim sönüyor")

    return (len(reasons) > 0, ", ".join(reasons))

def tp_sl_row(row):
    """
    TP/SL yüzdeleri (volatiliteye göre) + fiyat seviyeleri
    - risk = max(rv_60, atr_pct)
    """
    close = row.get("close", None)
    direction = str(row.get("direction", "NEUTRAL"))

    if close is None or pd.isna(close):
        return (None, None, None, None, None, None)

    close = float(close)
    rv = row.get("rv_60", None)
    atrp = row.get("atr_pct", None)

    rv = float(rv) if (rv is not None and pd.notna(rv)) else None
    atrp = float(atrp) if (atrp is not None and pd.notna(atrp)) else None

    base = 0.0
    if rv is not None:
        base = max(base, rv)
    if atrp is not None:
        base = max(base, atrp)

    if base <= 0:
        return (None, None, None, None, None, None)

    # basit plan
    sl_pct = 0.90 * base
    tp1_pct = 1.20 * base
    tp2_pct = 2.00 * base

    if direction == "SHORT":
        sl_price = close * (1 + sl_pct)
        tp1_price = close * (1 - tp1_pct)
        tp2_price = close * (1 - tp2_pct)
    else:  # LONG veya NEUTRAL varsayımı
        sl_price = close * (1 - sl_pct)
        tp1_price = close * (1 + tp1_pct)
        tp2_price = close * (1 + tp2_pct)

    return (sl_pct, tp1_pct, tp2_pct, sl_price, tp1_price, tp2_price)

# ✅ HTML parse_mode
def send_telegram_message(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True
        }
        r = requests.post(url, json=payload, timeout=15)
        return r.status_code == 200
    except Exception:
        return False

def build_fake_signal_message(timeframe: str) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    header = (
        f"🧪 <b>TEST - CİDDİ SİNYAL</b>  <b>LONG/SHORT</b> + <b>BREAKOUT</b> ✅\n"
        f"⏱ <b>TF:</b> {timeframe} | ⭐ <b>Conf≥</b>75 | 🧊 <b>Cooldown:</b>20dk\n"
        f"🕒 <i>{now_utc}</i>\n"
        f"────────────────────"
    )
    block = (
        "\n<b>1) AVAX/USDT</b>  🟢 <b>LONG</b>   🚀\n"
        "<code>"
        "Conf:82  Q:78  P:13.6100  vol_z:4.12  mom5:1.34%\n"
        "Phase:MID  SL:-0.90%  TP1:+1.20%  TP2:+2.00%"
        "</code>\n"
        "────────────────────"
    )
    return header + block

with st.sidebar:
    if st.button("📩 Telegram Test Mesajı Gönder"):
        msg = build_fake_signal_message(timeframe)
        ok = send_telegram_message(tg_token, tg_chat_id, msg)
        if ok:
            st.success("Test mesajı Telegram'a gönderildi ✅")
        else:
            st.error("Test mesajı gönderilemedi. Token/ChatID veya botta /start kontrol et.")

# ---------------- Run ----------------
should_run = run_btn or auto_refresh

if should_run:
    with st.spinner("Piyasayı tarıyorum... (ücretsiz API, biraz sürebilir)"):
        results = run_scan()

    if results is None or results.empty:
        st.warning("Sonuç yok. Biraz sonra tekrar dene veya bar sayısını artır.")
        st.stop()

    quote_map = get_quote_volume_map_usdt()
    ex = make_exchange()

    sym_list = results["symbol"].head(int(topn)).tolist()
    sym_list += state["watchlist"]
    sym_list = list(dict.fromkeys(sym_list))
    spread_map = get_spread_map(ex, sym_list, max_n=40)

    results_q = add_quality(results, quote_map, spread_map)
    results_q = add_confidence(results_q)

    # phase + filter + tp/sl
    mom_thr = float(ret5_thr_pct) / 100.0
    results_q["phase"] = results_q.apply(lambda r: pump_phase_row(r, float(vol_z_thr), mom_thr), axis=1)
    filt = results_q.apply(lambda r: false_breakout_filter(r), axis=1)
    results_q["filtered"] = [x[0] for x in filt]
    results_q["filter_reason"] = [x[1] for x in filt]

    tpsl = results_q.apply(lambda r: tp_sl_row(r), axis=1, result_type="expand")
    tpsl.columns = ["sl_pct","tp1_pct","tp2_pct","sl_price","tp1_price","tp2_price"]
    results_q = pd.concat([results_q, tpsl], axis=1)

    # ---------- Top liste ----------
    st.subheader(f"Top {topn} Anomali (+ Quality + Confidence + Phase)")
    view = results_q.head(int(topn)).copy()

    view["quality"] = view["quality"].round(0).astype(int)
    view["confidence"] = view["confidence"].round(0).astype(int)
    view["score"] = view["score"].round(2)
    view["vol_z"] = view["vol_z"].round(2)
    view["rsi_14"] = pd.to_numeric(view["rsi_14"], errors="coerce").round(1)
    view["spread_pct"] = pd.to_numeric(view["spread_pct"], errors="coerce").round(3)
    view["quote_vol_usdt"] = (pd.to_numeric(view["quote_vol_usdt"], errors="coerce") / 1_000_000).round(1)

    for c in ["ret_5", "ret_15"]:
        view[c] = (view[c] * 100).round(2)

    view["sl_pct"] = (view["sl_pct"] * 100).round(2)
    view["tp1_pct"] = (view["tp1_pct"] * 100).round(2)
    view["tp2_pct"] = (view["tp2_pct"] * 100).round(2)

    st.dataframe(
        view[[
            "symbol","direction","phase","confidence","quality","score",
            "vol_z","ret_5","ret_15","rsi_14",
            "spread_pct","quote_vol_usdt",
            "breakout","giveback",
            "sl_pct","tp1_pct","tp2_pct",
            "close","ts","filtered","filter_reason"
        ]],
        use_container_width=True
    )

    # ---------- Alarm listesi ----------
    st.subheader("🔔 Alarm Listesi (Confidence + Phase + Filter)")
    vol_z_thr_f = float(vol_z_thr)
    ret_thr = float(ret5_thr_pct) / 100.0
    cooldown_sec = int(cooldown_min) * 60

    raw_alerts = results_q[
        (results_q["vol_z"] >= vol_z_thr_f) &
        (results_q["giveback"] == 0) &
        (
            (results_q["ret_5"] >= ret_thr) |
            (results_q["ret_5"] <= -ret_thr)
        )
    ].copy()

    if tg_only_long_short:
        raw_alerts = raw_alerts[raw_alerts["direction"].isin(["LONG", "SHORT"])].copy()

    now_ts = time.time()
    filtered_rows = []
    for _, row in raw_alerts.iterrows():
        sym = row["symbol"]
        last_ts = state["last_alert_ts"].get(sym, 0)
        if now_ts - last_ts >= cooldown_sec:
            filtered_rows.append(row)
            state["last_alert_ts"][sym] = now_ts

    alerts_q = pd.DataFrame(filtered_rows)
    if not alerts_q.empty:
        alerts_q = alerts_q.sort_values(["confidence","quality","score"], ascending=False).head(25).copy()

    st.caption(f"Cooldown: {int(cooldown_min)} dk | Momentum eşiği: ±{ret5_thr_pct:.1f}%")

    if alerts_q.empty:
        st.info("Alarm koşullarına uyan sinyal yok.")
    else:
        a = alerts_q.copy()
        a["quality"] = a["quality"].round(0).astype(int)
        a["confidence"] = a["confidence"].round(0).astype(int)
        a["score"] = a["score"].round(2)
        a["vol_z"] = a["vol_z"].round(2)
        a["rsi_14"] = pd.to_numeric(a["rsi_14"], errors="coerce").round(1)
        a["ret_5"] = (a["ret_5"] * 100).round(2)
        a["ret_15"] = (a["ret_15"] * 100).round(2)
        a["spread_pct"] = pd.to_numeric(a["spread_pct"], errors="coerce").round(3)
        a["quote_vol_usdt"] = (pd.to_numeric(a["quote_vol_usdt"], errors="coerce") / 1_000_000).round(1)

        a["sl_pct"] = (a["sl_pct"] * 100).round(2)
        a["tp1_pct"] = (a["tp1_pct"] * 100).round(2)
        a["tp2_pct"] = (a["tp2_pct"] * 100).round(2)

        st.dataframe(
            a[[
                "symbol","direction","phase","confidence","quality","score",
                "vol_z","ret_5","ret_15","rsi_14",
                "spread_pct","quote_vol_usdt",
                "breakout","filtered","filter_reason",
                "sl_pct","tp1_pct","tp2_pct",
                "close","ts"
            ]],
            use_container_width=True
        )

        # ---------- Log (signals.csv) ----------
        log_path = "signals.csv"
        log_rows = []
        for _, r in alerts_q.iterrows():
            log_rows.append({
                "logged_at_utc": datetime.now(timezone.utc).isoformat(),
                "symbol": r["symbol"],
                "timeframe": timeframe,
                "direction": r.get("direction", ""),
                "phase": r.get("phase", ""),
                "confidence": float(r.get("confidence", 0.0)),
                "quality": float(r.get("quality", 0.0)),
                "score": float(r.get("score", 0.0)),
                "vol_z": float(r.get("vol_z", 0.0)),
                "ret_5": float(r.get("ret_5", 0.0)),
                "rsi_14": float(r.get("rsi_14", 0.0)) if pd.notna(r.get("rsi_14", None)) else None,
                "spread_pct": float(r.get("spread_pct", 0.0)) if pd.notna(r.get("spread_pct", None)) else None,
                "close": float(r.get("close", 0.0)),
                "sl_pct": float(r.get("sl_pct", 0.0)) if pd.notna(r.get("sl_pct", None)) else None,
                "tp1_pct": float(r.get("tp1_pct", 0.0)) if pd.notna(r.get("tp1_pct", None)) else None,
                "tp2_pct": float(r.get("tp2_pct", 0.0)) if pd.notna(r.get("tp2_pct", None)) else None,
                "filtered": bool(r.get("filtered", False)),
                "filter_reason": r.get("filter_reason", ""),
            })

        log_df = pd.DataFrame(log_rows)
        if os.path.exists(log_path):
            log_df.to_csv(log_path, mode="a", header=False, index=False)
        else:
            log_df.to_csv(log_path, mode="w", header=True, index=False)

        st.success(f"{len(alerts_q)} alarm kaydedildi → {log_path}")

        # ---------- Telegram: sadece ciddi + breakout + filtrelerden geçen ----------
        tg_df = alerts_q.copy()

        tg_df = tg_df[tg_df["quality"] >= float(tg_min_quality)].copy()
        tg_df = tg_df[tg_df["confidence"] >= float(tg_min_conf)].copy()
        tg_df = tg_df[tg_df["direction"].isin(["LONG", "SHORT"])].copy()

        if tg_breakout_only:
            tg_df = tg_df[tg_df["breakout"] == 1].copy()

        if tg_skip_late:
            tg_df = tg_df[tg_df["phase"] != "LATE"].copy()

        # false breakout filtresinden geçenler
        tg_df = tg_df[tg_df["filtered"] == False].copy()

        # momentum şartı (yönüne göre)
        thr = float(ret5_thr_pct) / 100.0
        tg_df = tg_df[
            ((tg_df["direction"] == "LONG") & (tg_df["ret_5"] >= thr)) |
            ((tg_df["direction"] == "SHORT") & (tg_df["ret_5"] <= -thr))
        ].copy()

        # Telegram cooldown
        tg_cooldown_sec = int(tg_cooldown_min) * 60
        now2 = time.time()

        tg_send_rows = []
        for _, r in tg_df.iterrows():
            sym = r["symbol"]
            last_tg = state["last_tg_ts"].get(sym, 0)
            if now2 - last_tg >= tg_cooldown_sec:
                tg_send_rows.append(r)

        if tg_enabled and tg_send_rows:
            send_list = tg_send_rows[: int(tg_max_items)]

            def fmt_num(x, nd=2):
                try:
                    return f"{float(x):.{nd}f}"
                except Exception:
                    return "NA"

            def fmt_price(x):
                try:
                    v = float(x)
                    if v >= 1:
                        return f"{v:.4f}"
                    return f"{v:.8f}"
                except Exception:
                    return "NA"

            now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
            header = (
                f"🚨 <b>CİDDİ SİNYAL</b> ✅\n"
                f"⏱ <b>TF:</b> {timeframe} | ⭐ <b>Conf≥</b>{int(tg_min_conf)} | 🧪 <b>Q≥</b>{int(tg_min_quality)} | 🧊 <b>CD:</b>{int(tg_cooldown_min)}dk\n"
                f"🕒 <i>{now_utc}</i>\n"
                f"────────────────────"
            )

            blocks = [header]

            for i, r in enumerate(send_list, start=1):
                sym = r["symbol"]
                direction = r.get("direction", "NA")
                conf = int(round(float(r.get("confidence", 0))))
                q = int(round(float(r.get("quality", 0))))
                phase = str(r.get("phase", "NONE"))
                vz = float(r.get("vol_z", 0.0)) if pd.notna(r.get("vol_z", None)) else None
                ret5 = float(r.get("ret_5", 0.0)) * 100.0
                rsi14 = r.get("rsi_14", None)
                spread = r.get("spread_pct", None)
                close = r.get("close", None)

                slp = r.get("sl_pct", None)
                tp1p = r.get("tp1_pct", None)
                tp2p = r.get("tp2_pct", None)

                dir_emoji = "🟢" if direction == "LONG" else "🔴"
                phase_emoji = "🟡" if phase == "EARLY" else ("🟢" if phase == "MID" else ("🔴" if phase == "LATE" else "⚪"))

                rsi_val = float(rsi14) if (rsi14 is not None and pd.notna(rsi14)) else None
                spr_val = float(spread) if (spread is not None and pd.notna(spread)) else None

                risk = ""
                if rsi_val is not None:
                    if direction == "LONG" and rsi_val >= 75:
                        risk = "  ⚠️ <b>RSI yüksek</b>"
                    if direction == "SHORT" and rsi_val <= 25:
                        risk = "  ⚠️ <b>RSI düşük</b>"

                sl_txt = f"-{fmt_num(float(slp)*100,2)}%" if (slp is not None and pd.notna(slp)) else "NA"
                tp1_txt = f"+{fmt_num(float(tp1p)*100,2)}%" if (tp1p is not None and pd.notna(tp1p)) else "NA"
                tp2_txt = f"+{fmt_num(float(tp2p)*100,2)}%" if (tp2p is not None and pd.notna(tp2p)) else "NA"

                block = (
                    f"\n<b>{i}) {sym}</b>  {dir_emoji} <b>{direction}</b>  |  {phase_emoji} <b>{phase}</b>\n"
                    f"<code>"
                    f"Conf:{conf:>3}  Q:{q:>3}  P:{fmt_price(close):>10}  vol_z:{fmt_num(vz,2):>6}  mom5:{fmt_num(ret5,2):>6}%\n"
                    f"SL:{sl_txt:>8}  TP1:{tp1_txt:>8}  TP2:{tp2_txt:>8}  RSI:{fmt_num(rsi_val,1):>5}  spr:{fmt_num(spr_val,3):>6}%"
                    f"</code>"
                    f"{risk}\n"
                    f"────────────────────"
                )
                blocks.append(block)
                state["last_tg_ts"][sym] = now2

            text = "".join(blocks)
            ok = send_telegram_message(tg_token, tg_chat_id, text)
            if ok:
                st.success(f"Telegram'a {len(send_list)} ciddi sinyal gönderildi.")
            else:
                st.warning("Telegram gönderimi başarısız. Token/ChatID kontrol et ve botla /start yaptığından emin ol.")

    # ---------- Sinyal Takibi (signals.csv) ----------
    st.subheader("📊 Sinyal Takibi (signals.csv)")

    if os.path.exists("signals.csv"):
        try:
            sig = pd.read_csv("signals.csv")
            sig = sig.tail(60).copy()

            # Binance tüm fiyatlar (tek istek)
            price_map = {}
            try:
                prices = requests.get("https://api.binance.com/api/v3/ticker/price", timeout=15).json()
                for it in prices:
                    s = it.get("symbol", "")
                    p = float(it.get("price", 0.0))
                    if s.endswith("USDT"):
                        ccxt_symbol = s[:-4] + "/USDT"
                        price_map[ccxt_symbol] = p
            except Exception:
                pass

            def pnl_now(row):
                sym = row.get("symbol", "")
                entry = row.get("close", None)
                direction = row.get("direction", "NEUTRAL")
                nowp = price_map.get(sym, None)
                if entry is None or pd.isna(entry) or nowp is None:
                    return None
                entry = float(entry)
                nowp = float(nowp)
                ret = (nowp - entry) / entry
                if direction == "SHORT":
                    ret = -ret
                return ret * 100.0

            def age_min(row):
                try:
                    t = pd.to_datetime(row["logged_at_utc"], utc=True)
                    return (datetime.now(timezone.utc) - t.to_pydatetime()).total_seconds() / 60.0
                except Exception:
                    return None

            sig["age_min"] = sig.apply(age_min, axis=1)
            sig["pnl_now_pct"] = sig.apply(pnl_now, axis=1)

            show = sig[[
                "logged_at_utc","symbol","timeframe","direction","phase","confidence","quality","close","age_min","pnl_now_pct","filtered","filter_reason"
            ]].copy()

            show["confidence"] = pd.to_numeric(show["confidence"], errors="coerce").round(0)
            show["quality"] = pd.to_numeric(show["quality"], errors="coerce").round(0)
            show["age_min"] = pd.to_numeric(show["age_min"], errors="coerce").round(1)
            show["pnl_now_pct"] = pd.to_numeric(show["pnl_now_pct"], errors="coerce").round(2)

            st.dataframe(show, use_container_width=True)

            # mini özet
            valid = show[pd.notna(show["pnl_now_pct"])].copy()
            if not valid.empty:
                avg = float(valid["pnl_now_pct"].mean())
                win = float((valid["pnl_now_pct"] > 0).mean() * 100.0)
                st.info(f"Canlı PnL (son {len(valid)} sinyal): Ortalama {avg:.2f}% | Win-rate {win:.1f}% (LONG/SHORT yönüne göre hesaplandı)")
            else:
                st.info("PnL hesaplamak için canlı fiyat çekilemedi veya sembol eşleşmedi.")

        except Exception:
            st.warning("signals.csv okurken hata oldu. Dosya bozulmuş olabilir.")
    else:
        st.info("Henüz signals.csv yok. Bir alarm oluşunca otomatik oluşacak.")

    # ---------- Detay grafiği ----------
    st.subheader("📈 Seçili Coin Detayı")
    picked = st.selectbox("Coin seç", results_q["symbol"].head(int(topn)).tolist())
    if picked:
        ex2 = make_exchange()
        df = fetch_ohlcv_df(ex2, picked, timeframe=timeframe, limit=int(limit))
        if df is not None and len(df) > 60:
            df = compute_features(df, timeframe=timeframe)
            chart_df = df.set_index("ts")[["close", "volume", "vol_z", "rsi_14"]].copy()
            st.line_chart(chart_df[["close"]], height=260)
            st.line_chart(chart_df[["volume"]], height=160)
            st.line_chart(chart_df[["vol_z"]], height=160)
            st.line_chart(chart_df[["rsi_14"]], height=160)
        else:
            st.warning("Bu coin için veri çekilemedi, başka coin dene.")

else:
    st.info("Soldan ayarları yapıp **Taramayı Başlat / Yenile**'ye bas. Auto-refresh açarsan otomatik tarar.")
