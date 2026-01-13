import os
import time
from datetime import datetime, timezone
import pandas as pd

from scanner import (
    make_exchange,
    scan,
    get_top_usdt_symbols_by_quote_volume
)

import requests


def send_telegram(token: str, chat_id: str, text: str) -> bool:
    if not token or not chat_id:
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def main():
    # Secrets env olarak gelecek
    tg_token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.getenv("TELEGRAM_CHAT_ID", "")

    # tarama ayarları (istersen sonra env yaparız)
    timeframe = os.getenv("TF", "1m")
    limit = int(os.getenv("LIMIT", "720"))
    top_symbols = int(os.getenv("TOP_SYMBOLS", "200"))
    min_quote_usdt = float(os.getenv("MIN_QUOTE_USDT", "5000000"))

    # telegram eşikleri
    tg_min_quality = float(os.getenv("TG_MIN_QUALITY", "75"))
    tg_min_conf = float(os.getenv("TG_MIN_CONF", "75"))
    breakout_only = os.getenv("TG_BREAKOUT_ONLY", "1") == "1"

    # --- scan ---
    symbols = get_top_usdt_symbols_by_quote_volume(limit=top_symbols, min_quote_usdt=min_quote_usdt)
    ex = make_exchange()
    df = scan(ex, symbols, timeframe=timeframe, limit=limit)

    if df is None or df.empty:
        print("No results")
        return

    # Not: Senin app.py'de ürettiğin confidence/quality hesapları worker’a da taşınabilir.
    # Şimdilik: scanner'dan gelen direction/breakout/vol_z/ret_5/score ile “ciddi” filtre
    df = df.copy()

    # Eğer scanner’da direction/breakout varsa kullanır:
    if "direction" not in df.columns:
        df["direction"] = "NEUTRAL"
    if "breakout" not in df.columns:
        df["breakout"] = 0
    if "quality" not in df.columns:
        # Worker’da quality hesabı yoksa minimum filtreyi score üzerinden yaparız
        df["quality"] = 0

    # “Ciddi” filtre (basit MVP):
    # - LONG/SHORT
    # - breakout_only ise breakout
    # - score yüksek
    base = df[df["direction"].isin(["LONG", "SHORT"])].copy()
    if breakout_only:
        base = base[base["breakout"] == 1].copy()

    base = base.sort_values("score", ascending=False).head(10)

    if base.empty:
        print("No serious signals")
        return

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    lines = []
    lines.append(f"🚨 <b>CİDDİ SİNYAL</b> | TF: <b>{timeframe}</b>")
    lines.append(f"🕒 <i>{now_utc}</i>")
    lines.append("────────────────────")

    for i, r in enumerate(base.itertuples(index=False), start=1):
        sym = getattr(r, "symbol")
        direction = getattr(r, "direction", "NA")
        score = getattr(r, "score", 0.0)
        vol_z = getattr(r, "vol_z", None)
        ret5 = getattr(r, "ret_5", 0.0) * 100.0
        close = getattr(r, "close", None)
        bo = getattr(r, "breakout", 0)

        dir_emoji = "🟢" if direction == "LONG" else "🔴"
        bo_emoji = "🚀" if int(bo) == 1 else "—"

        price_txt = f"{float(close):.6g}" if close is not None else "NA"
        vz_txt = f"{float(vol_z):.2f}" if vol_z is not None else "NA"

        lines.append(f"<b>{i}) {sym}</b> {dir_emoji} <b>{direction}</b> {bo_emoji}")
        lines.append(f"<code>P:{price_txt}  score:{float(score):.2f}  vol_z:{vz_txt}  mom5:{float(ret5):.2f}%</code>")
        lines.append("────────────────────")

    msg = "\n".join(lines)

    ok = send_telegram(tg_token, tg_chat, msg)
    print("Sent:", ok)


if __name__ == "__main__":
    main()
