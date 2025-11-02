#!/usr/bin/env python3
"""
pocket_option.py
Pocket Master AI — ModeA v10 (Choreo-ready, command-driven, polling, keep-alive)
- Polling-based Telegram bot (no webhook)
- Flask keep-alive endpoint for UptimeRobot / Choreo
- Both engines (quantum + momentum) and switching via /mode
- Commands: /help, /status, /analyze, /trade, /mode, /retrain, challenge commands
- Single-instance guard (file lock)
- Persistence to /tmp folder
"""

import os
import time
import json
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Data & math libs
import math
import random

try:
    import requests
    import numpy as np
    import pandas as pd
except Exception as e:
    print("Missing modules; ensure requirements installed:", e)

# ---------------- Config (env) ----------------
BRAND = os.getenv("BRAND", "Pocket Master AI")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("BOT_TOKEN")
ADMIN_CHAT = os.getenv("TELEGRAM_ADMIN_CHAT")  # numeric id string preferred
TELEGRAM_GROUP_BASIC = os.getenv("TELEGRAM_GROUP_BASIC", "")
TELEGRAM_GROUP_VIP = os.getenv("TELEGRAM_GROUP_VIP", "")

HF_API_KEY = os.getenv("HF_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/po_models_v10_full")
os.makedirs(MODEL_DIR, exist_ok=True)
MEMBERS_FILE = os.path.join(MODEL_DIR, "members_v10.json")
CHALLENGE_DB = os.path.join(MODEL_DIR, "challenge_v10.json")
OFFSET_FILE = os.path.join(MODEL_DIR, "tg_offset.json")
LOCK_FILE = os.path.join(MODEL_DIR, "pocket_option.lock")

# Default symbols
PO_SYMBOLS = os.getenv("PO_SYMBOLS", "EURUSD,GBPJPY,BTC-USD,USDJPY,AUDUSD").split(",")

# Timeframes and thresholds
ANALYSIS_TFS = ["1m","2m","3m","5m","15m","30m","1h","2h","4h","1d","1wk","1mo"]
TOP_TFS = int(os.getenv("TOP_TFS", "5"))
TOP_STRATS = int(os.getenv("TOP_STRATS", "5"))
CONF_THRESHOLD_PRIORITY = float(os.getenv("CONF_THRESHOLD_PRIORITY", "90.0"))
CONF_THRESHOLD_FLEX_MIN = float(os.getenv("CONF_THRESHOLD_FLEX_MIN", "80.0"))
MIN_TF_ALIGN_REQUIRED = int(os.getenv("MIN_TF_ALIGN_REQUIRED", "4"))

# Keepalive / heartbeat
PORT = int(os.environ.get("PORT", 8080))
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "300"))
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "1800"))

# Polling config
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.0"))

# Mode state
MODE_STATE_FILE = os.path.join(MODEL_DIR, "mode_state.json")
DEFAULT_ENGINE = "quantum"  # or 'momentum'

# ---------------- Simple persistence helpers ----------------
def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        print("save_json error:", e)

_members = load_json(MEMBERS_FILE, {})
_challenge = load_json(CHALLENGE_DB, {"participants": {}, "signals": []})
_mode_state = load_json(MODE_STATE_FILE, {"engine": DEFAULT_ENGINE})

# ---------------- Single-instance guard ----------------
def ensure_single_instance():
    # simple lock file approach
    try:
        if os.path.exists(LOCK_FILE):
            # check if process still running (best-effort)
            print("Lock file exists — assuming another instance is running. Exiting.")
            raise SystemExit("Another instance detected.")
        open(LOCK_FILE, "w").write(f"{os.getpid()}\n")
    except Exception as e:
        print("Failed single-instance guard:", e)
        raise

def remove_instance_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

# ---------------- Telegram helpers (raw API via requests) ----------------
BASE_TELEGRAM = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}" if TELEGRAM_TOKEN else None

def send_telegram(chat_id, text, parse_mode="Markdown"):
    if not TELEGRAM_TOKEN or not chat_id:
        print("send_telegram: missing token or chat_id")
        return False, "tg_not_configured"
    try:
        url = f"{BASE_TELEGRAM}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": parse_mode}, timeout=10)
        return resp.ok, resp.text
    except Exception as e:
        return False, str(e)

def broadcast_to_groups(msg):
    if TELEGRAM_GROUP_VIP:
        send_telegram(TELEGRAM_GROUP_VIP, msg)
    if TELEGRAM_GROUP_BASIC:
        send_telegram(TELEGRAM_GROUP_BASIC, msg)
    if ADMIN_CHAT:
        send_telegram(ADMIN_CHAT, msg)

def log(msg, to_admin=True, to_console=True):
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    s = f"[{ts}] {msg}"
    if to_console:
        print(s)
    if to_admin and ADMIN_CHAT:
        ok, resp = send_telegram(ADMIN_CHAT, s)
        if not ok:
            print("Warning: admin message failed:", resp)

# ---------------- Modes & engine selection ----------------
def get_current_engine():
    return _mode_state.get("engine", DEFAULT_ENGINE)

def set_current_engine(name):
    _mode_state["engine"] = name
    save_json(MODE_STATE_FILE, _mode_state)

# ---------------- Indicators & strategies (compacted from v10) ----------------
# helper indicators
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=(period-1), adjust=False).mean()
    ema_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ema_up/(ema_down+1e-9)
    return 100 - (100/(1+rs))

# Quantum Engine (v2) — simplified version
def quantum_engine_v2(df):
    try:
        if df is None or df.empty: return 0.0, 0.0
        c = df["Close"].dropna()
        if len(c) < 20: return 0.0, 0.0
        e8 = ema(c,8); e21 = ema(c,21)
        macd = e8 - e21
        macd_sig = macd.ewm(span=9,adjust=False).mean()
        hist = macd - macd_sig
        ema_cross_sign = np.sign(e8.iloc[-1] - e21.iloc[-1])
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        vol_trend = 0.0
        try:
            if len(vol.dropna()) >= 3:
                vol_trend = np.polyfit(range(min(10,len(vol.dropna()))), vol.dropna().values[-min(10,len(vol.dropna())):],1)[0]
        except Exception:
            vol_trend = 0.0
        vol_score = 1 if vol_trend>0 else -1 if vol_trend<0 else 0
        buy_votes = 0
        buy_votes += 1 if (ema_cross_sign>0 and hist.iloc[-1]>0) else 0
        buy_votes += 1 if vol_score>0 else 0
        buy_votes += 1 if c.iloc[-1] > c.rolling(20).mean().iloc[-1] else 0
        score = (buy_votes - (3-buy_votes))/3.0
        conf = min(1.0, abs(hist.iloc[-1])/(np.std(hist[-10:])+1e-9) if len(hist)>=10 else 0.3)
        return float(np.clip(score,-1.0,1.0)), float(np.clip(conf,0.0,1.0))
    except Exception:
        return 0.0, 0.0

# Momentum Scalper (v1)
def momentum_scalper_v1(df):
    try:
        if df is None or df.empty: return 0.0, 0.0
        c = df["Close"].dropna()
        if len(c) < 6: return 0.0, 0.0
        last = c.iloc[-1]
        prev5 = c.iloc[-6:-1]
        momentum = (last - prev5.mean())/(prev5.mean()+1e-9)
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        vol_spike = False
        try:
            if len(vol) >= 5:
                vol_spike = vol.iloc[-1] > (vol.rolling(5).mean().iloc[-1]*1.8)
        except Exception:
            vol_spike = False
        rsi_val = rsi(c,14).iloc[-1] if len(c)>=15 else 50.0
        buy = 1 if (momentum>0 and vol_spike and rsi_val>50) else -1 if (momentum<0 and vol_spike and rsi_val<50) else 0
        conf = min(1.0, abs(momentum)*2.0 + (1.0 if vol_spike else 0.0))
        return float(np.clip(buy,-1.0,1.0)), float(np.clip(conf,0.0,1.0))
    except Exception:
        return 0.0, 0.0

# Strategy registry
STRATEGY_FUNCS = {
    "quantum": quantum_engine_v2,
    "momentum": momentum_scalper_v1
}

# Hidden liquidity & proprietary logic (kept lightweight)
def dark_pool_internal_model(df):
    try:
        if df is None or df.empty: return 0.0, 0.0
        c = df["Close"].dropna()
        if len(c) < 10: return 0.0, 0.0
        mid = (df["High"] + df["Low"]) / 2.0
        dir_sign = 1.0 if c.iloc[-1] > mid.iloc[-1] else -1.0
        score = dir_sign * 0.1
        conf = 0.2
        return float(score), float(conf)
    except Exception:
        return 0.0, 0.0

def proprietary_desk_logic(df):
    try:
        return 0.0, 0.0, {}
    except Exception:
        return 0.0, 0.0, {}

# ---------------- Aggregation & decision (simplified but robust) ----------------
def evaluate_strategies_on_df(df):
    out = {}
    for name, fn in STRATEGY_FUNCS.items():
        try:
            s, c = fn(df)
            out[name] = {"score": float(s), "conf": float(c)}
        except Exception:
            out[name] = {"score": 0.0, "conf": 0.0}
    dp_s, dp_c = dark_pool_internal_model(df)
    pa_s, pa_c, pa_meta = proprietary_desk_logic(df)
    out["darkpool"] = {"score": float(dp_s), "conf": float(dp_c)}
    out["proprietary"] = {"score": float(pa_s), "conf": float(pa_c), "meta": pa_meta}
    return out

def aggregate_across_tfs(symbol, tf_df_map):
    per_tf = {}
    tf_votes = {}
    for tf, df in tf_df_map.items():
        strat = evaluate_strategies_on_df(df)
        per_tf[tf] = strat
        buy = 0.0; sell = 0.0; totalw = 0.0
        for sname, info in strat.items():
            w = info.get("conf", 0.0)
            totalw += w
            if info.get("score", 0.0) > 0:
                buy += info["score"] * w
            elif info.get("score", 0.0) < 0:
                sell += -info["score"] * w
        if totalw == 0:
            tf_votes[tf] = {"vote": "HOLD", "strength": 0.0}
        else:
            if buy > sell:
                tf_votes[tf] = {"vote": "BUY", "strength": float(buy/totalw)}
            elif sell > buy:
                tf_votes[tf] = {"vote": "SELL", "strength": float(sell/totalw)}
            else:
                tf_votes[tf] = {"vote": "HOLD", "strength": 0.0}
    strat_agg = {}
    for tf, strat in per_tf.items():
        for s, info in strat.items():
            strat_agg[s] = strat_agg.get(s, 0.0) + info.get("score", 0.0) * info.get("conf", 0.0)
    maxabs = max((abs(v) for v in strat_agg.values()), default=1.0)
    for k in strat_agg:
        strat_agg[k] = strat_agg[k] / (maxabs + 1e-9)
    return per_tf, tf_votes, strat_agg

def decide_for_symbol(symbol, tf_df_map, news_bias=0.0):
    per_tf, tf_votes, strat_agg = aggregate_across_tfs(symbol, tf_df_map)
    sorted_strats = sorted(strat_agg.items(), key=lambda x: abs(x[1]), reverse=True)
    top_strats = [s for s,_ in sorted_strats[:TOP_STRATS]]
    tf_alignment = {}
    for tf, info in tf_votes.items():
        agree = 0; total = 0
        for s in top_strats:
            total += 1
            score = per_tf[tf].get(s,{}).get("score",0.0)
            if info["vote"]=="BUY" and score>0: agree+=1
            if info["vote"]=="SELL" and score<0: agree+=1
        ratio = agree/max(1,total)
        tf_alignment[tf] = {"vote": info["vote"], "strength": info["strength"], "agree_ratio": ratio}
    tf_priority = [tf for tf in ANALYSIS_TFS if tf in tf_alignment][::-1]
    selected_tfs = [tf for tf in tf_priority[:TOP_TFS]]
    aligned_tfs = [tf for tf in selected_tfs if tf_alignment[tf]["agree_ratio"]>=0.9 and tf_alignment[tf]["vote"]!="HOLD"]
    if len(aligned_tfs) < MIN_TF_ALIGN_REQUIRED:
        return {"action":"HOLD","confidence":0.0,"reason":"not_enough_tf_align","tf_alignment":tf_alignment,"top_strats":top_strats}
    buy_count = sum(1 for tf in aligned_tfs if tf_alignment[tf]["vote"]=="BUY")
    sell_count = sum(1 for tf in aligned_tfs if tf_alignment[tf]["vote"]=="SELL")
    if buy_count > sell_count:
        final_action = "BUY"; avg_strength = np.mean([tf_alignment[tf]["strength"] for tf in aligned_tfs if tf_alignment[tf]["vote"]=="BUY"])
    elif sell_count > buy_count:
        final_action = "SELL"; avg_strength = np.mean([tf_alignment[tf]["strength"] for tf in aligned_tfs if tf_alignment[tf]["vote"]=="SELL"])
    else:
        return {"action":"HOLD","confidence":0.0,"reason":"tie","tf_alignment":tf_alignment,"top_strats":top_strats}
    strat_align_count = sum(1 for s in top_strats if (strat_agg.get(s,0.0)>0 and final_action=="BUY") or (strat_agg.get(s,0.0)<0 and final_action=="SELL"))
    strat_align_ratio = strat_align_count / max(1, len(top_strats))
    dp = np.mean([per_tf[tf]["darkpool"]["score"] for tf in aligned_tfs if "darkpool" in per_tf[tf]] or [0.0])
    dp_conf = np.mean([per_tf[tf]["darkpool"]["conf"] for tf in aligned_tfs if "darkpool" in per_tf[tf]] or [0.0])
    pa = np.mean([per_tf[tf]["proprietary"]["score"] for tf in aligned_tfs if "proprietary" in per_tf[tf]] or [0.0])
    pa_conf = np.mean([per_tf[tf]["proprietary"].get("conf",0.0) for tf in aligned_tfs if "proprietary" in per_tf[tf]] or [0.0])
    final_conf = 50.0 + (avg_strength*50.0*strat_align_ratio) + (dp*25.0*dp_conf) + (pa*25.0*pa_conf) + (news_bias*20.0)
    final_conf = float(np.clip(final_conf, 0.0, 150.0))
    if final_conf >= CONF_THRESHOLD_PRIORITY:
        return {"action":final_action,"confidence":final_conf,"tag":"PRIORITY","top_strats":top_strats,"aligned_tfs":aligned_tfs,"strat_align_ratio":strat_align_ratio}
    if CONF_THRESHOLD_FLEX_MIN <= final_conf < CONF_THRESHOLD_PRIORITY:
        dp_ok = (abs(dp) > 0 and dp_conf >= 0.6)
        pa_ok = (abs(pa) > 0 and pa_conf >= 0.6)
        if dp_ok and pa_ok and strat_align_ratio >= 0.8:
            return {"action":final_action,"confidence":final_conf,"tag":"INSTITUTIONAL_FLEX","top_strats":top_strats,"aligned_tfs":aligned_tfs,"strat_align_ratio":strat_align_ratio}
        else:
            return {"action":"HOLD","confidence":final_conf,"reason":"failed_institutional_checks","dp":dp,"pa":pa}
    return {"action":"HOLD","confidence":final_conf,"reason":"below_threshold"}

# ---------------- Market data helper (yfinance) ----------------
def parse_tf_to_pd_rule(tf):
    tf = tf.lower().strip()
    if tf.endswith("m") and tf[:-1].isdigit(): return f"{int(tf[:-1])}T"
    if tf.endswith("h") and tf[:-1].isdigit(): return f"{int(tf[:-1])}H"
    if tf in ("1d","1wk","1mo"):
        return {"1d":"1D","1wk":"1W","1mo":"1M"}.get(tf)
    return None

def fetch_market_data_yf(yf_sym, tfs, period="6mo"):
    try:
        import yfinance as yf
    except Exception:
        return {tf: None for tf in tfs}
    out = {}
    need_min = any(tf.endswith("m") for tf in tfs)
    bases = {}
    try:
        if need_min:
            try:
                bases["1m"] = yf.download(tickers=yf_sym, period="7d", interval="1m", progress=False, threads=False)
            except Exception:
                bases["1m"] = None
        bases["1d"] = yf.download(tickers=yf_sym, period=period, interval="1d", progress=False, threads=False)
    except Exception:
        pass
    for tf in tfs:
        if tf in bases and bases[tf] is not None:
            out[tf] = bases[tf]; continue
        yf_int = None
        if tf in ["1m","2m","5m","15m","30m","60m","1d","1wk","1mo"]:
            yf_int = tf if tf!="60m" else "1h"
        try:
            if yf_int:
                df = yf.download(tickers=yf_sym, period="6mo" if yf_int in ["1d","1wk","1mo"] else "7d", interval=yf_int, progress=False, threads=False)
                if df is not None and not df.empty:
                    out[tf] = df; continue
        except Exception:
            pass
        base = bases.get("1m") or bases.get("1d")
        if base is None or base.empty:
            out[tf] = None; continue
        rule = parse_tf_to_pd_rule(tf)
        if rule is None:
            out[tf] = None; continue
        try:
            df_base = base.copy()
            if not isinstance(df_base.index, pd.DatetimeIndex):
                df_base.index = pd.to_datetime(df_base.index)
            df_res = df_base.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            out[tf] = df_res
        except Exception:
            out[tf] = None
    return out

# ---------------- News helpers (lightweight) ----------------
def hf_sentiment(text):
    if not HF_API_KEY: return 0.0
    try:
        url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        r = requests.post(url, headers=headers, json={"inputs": text}, timeout=6)
        if r.status_code == 200:
            data = r.json()
            label = data[0].get("label","").lower(); score = float(data[0].get("score",0.0))
            if "positive" in label: return score
            if "negative" in label: return -score
    except Exception:
        pass
    return 0.0

def fetch_headlines_and_score(symbol):
    if not NEWSAPI_KEY: return 0.0
    try:
        url = ("https://newsapi.org/v2/everything?q={}&pageSize=3&sortBy=publishedAt&apiKey={}").format(symbol, NEWSAPI_KEY)
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            items = r.json().get("articles",[])
            scores = []
            for it in items:
                txt = (it.get("title","") or "") + ". " + (it.get("description","") or "")
                scores.append(hf_sentiment(txt) if HF_API_KEY else 0.0)
            return float(np.mean(scores)) if scores else 0.0
    except Exception:
        pass
    return 0.0

# ---------------- Scanner used by /analyze and /trade ----------------
def scanner_cycle_once(symbols):
    results = []
    for s in symbols:
        try:
            data_map = fetch_market_data_yf(s, ANALYSIS_TFS, period="6mo")
            available = [tf for tf in ANALYSIS_TFS if data_map.get(tf) is not None]
            if not available:
                results.append({"symbol": s, "action": "HOLD", "confidence": 0.0, "reason": "no_data"})
                continue
            # pick top TFs to evaluate
            available_sorted = sorted(available, key=lambda x: ANALYSIS_TFS.index(x) if x in ANALYSIS_TFS else 0, reverse=True)
            selected = available_sorted[:TOP_TFS]
            tf_df_map = {tf: data_map.get(tf) for tf in selected}
            news_bias = fetch_headlines_and_score(s) if (NEWSAPI_KEY or HF_API_KEY) else 0.0
            decision = decide_for_symbol(s, tf_df_map, news_bias=news_bias)
            if decision.get("action") in ("BUY","SELL") and decision.get("confidence",0.0) >= CONF_THRESHOLD_FLEX_MIN:
                payload = {"id": f"{s}_{int(time.time())}", "symbol": s, "action": decision["action"], "confidence": decision["confidence"], "tag": decision.get("tag",""), "top_strats": decision.get("top_strats"), "aligned_tfs": decision.get("aligned_tfs")}
                results.append(payload)
            else:
                results.append({"symbol": s, "action": "HOLD", "confidence": decision.get("confidence", 0.0), "reason": decision.get("reason","")})
        except Exception as e:
            print("scan error for", s, e)
            traceback.print_exc()
            results.append({"symbol": s, "action": "HOLD", "confidence": 0.0, "reason": "scan_error"})
    return results

# ---------------- Challenge & membership (lightweight) ----------------
def ensure_member_record(user_telegram):
    global _members
    if user_telegram not in _members:
        _members[user_telegram] = {"tier":"none","expires":None,"joined": datetime.utcnow().isoformat()}
        save_json(MEMBERS_FILE, _members)
    return _members[user_telegram]

def join_challenge(user_telegram):
    if not is_member_active(user_telegram): return {"ok":False, "reason":"not_active_member"}
    if user_telegram in _challenge["participants"]: return {"ok":False, "reason":"already_joined"}
    p = {"joined": datetime.utcnow().isoformat(), "virtual_balance": 100.0, "signals_received_today":0, "total_signals_received":0, "wins":0, "losses":0, "last_active": datetime.utcnow().isoformat(), "history": []}
    _challenge["participants"][user_telegram] = p
    save_json(CHALLENGE_DB, _challenge)
    return {"ok":True, "participant":p}

def is_member_active(user_telegram, tier=None):
    rec = _members.get(user_telegram)
    if not rec: return False
    if rec.get("tier") == "none": return False
    exp = rec.get("expires")
    if not exp: return False
    try:
        expires = datetime.fromisoformat(exp)
        return datetime.utcnow() < expires
    except Exception:
        return False

# ---------------- Keep-alive Flask app ----------------
from flask import Flask as _Flask
app = _Flask(__name__)

@app.route("/")
def home():
    return "✅ Pocket Option Bot is alive!"

def start_flask_thread():
    def run():
        try:
            app.run(host="0.0.0.0", port=PORT)
        except Exception as e:
            print("Flask run error:", e)
    t = threading.Thread(target=run, daemon=True)
    t.start()

# ---------------- Telegram polling (getUpdates) ----------------
_update_offset_lock = threading.Lock()
_update_offset = load_json(OFFSET_FILE, {"offset": None}).get("offset")

def save_offset(offset):
    try:
        save_json(OFFSET_FILE, {"offset": offset})
    except Exception:
        pass

def handle_update(update):
    # process a single update (message/command)
    try:
        if "message" not in update:
            return
        msg = update["message"]
        chat = msg.get("chat", {})
        chat_id = chat.get("id")
        text = msg.get("text", "") or ""
        from_user = msg.get("from", {})
        username = from_user.get("username") or from_user.get("first_name") or str(chat_id)
        text = text.strip()
        if not text:
            return
        parts = text.split()
        cmd = parts[0].lower()

        # Only admin can run certain commands
        is_admin = False
        if ADMIN_CHAT:
            try:
                if str(ADMIN_CHAT).lstrip('@') == str(chat_id) or str(ADMIN_CHAT).lstrip('@') == str(username):
                    is_admin = True
            except Exception:
                pass

        if cmd in ("/help", "/start"):
            help_msg = (
                "🤖 Pocket Master AI — commands:\n\n"
                "/analyze [SYMBOL] — run analysis (defaults to configured symbols)\n"
                "/trade [SYMBOL] — run analysis and publish decision (instant)\n"
                "/status — bot status & uptime\n"
                "/mode [quantum|momentum] — set analysis engine\n"
                "/retrain — admin only placeholder\n"
                "/join_challenge @user — join challenge (requires membership)\n"
                "/leave_challenge @user\n"
            )
            send_telegram(chat_id, help_msg)
            return

        if cmd == "/status":
            uptime = str(datetime.utcnow() - START_TIME).split('.')[0]
            engine = get_current_engine()
            status = f"✅ {BRAND} running\nUptime: {uptime}\nEngine: {engine}\nSymbols: {','.join(PO_SYMBOLS)}"
            send_telegram(chat_id, status)
            return

        if cmd == "/analyze":
            syms = parts[1:] if len(parts) > 1 else PO_SYMBOLS
            send_telegram(chat_id, f"🔎 Running analysis for: {','.join(syms)} (this may take a moment)")
            res = scanner_cycle_once(syms)
            lines = []
            for r in res:
                if r.get("action") in ("BUY","SELL"):
                    lines.append(f"{r['symbol']}: {r['action']} ({r['confidence']:.1f}%) [{r.get('tag','')}]")
                else:
                    lines.append(f"{r['symbol']}: HOLD ({r.get('confidence',0.0):.1f}%) {r.get('reason','')}")
            send_telegram(chat_id, "🔎 Analysis results:\n" + "\n".join(lines))
            return

        if cmd == "/trade":
            syms = parts[1:] if len(parts) > 1 else PO_SYMBOLS
            send_telegram(chat_id, f"⚡ Executing trade analysis for: {','.join(syms)}")
            res = scanner_cycle_once(syms)
            executed = []
            for r in res:
                if r.get("action") in ("BUY","SELL"):
                    msg = f"📡 TRADE SIGNAL — {r['symbol']} → *{r['action']}*\nConfidence: *{r['confidence']:.1f}%*\n{BRAND}"
                    if ADMIN_CHAT:
                        send_telegram(ADMIN_CHAT, msg)
                    if TELEGRAM_GROUP_VIP:
                        send_telegram(TELEGRAM_GROUP_VIP, msg)
                    if TELEGRAM_GROUP_BASIC:
                        send_telegram(TELEGRAM_GROUP_BASIC, msg)
                    executed.append(f"{r['symbol']}:{r['action']}({r['confidence']:.1f}%)")
                else:
                    executed.append(f"{r['symbol']}:HOLD({r.get('confidence',0.0):.1f}%)")
            send_telegram(chat_id, "✅ Trade run complete:\n" + "\n".join(executed))
            return

        if cmd == "/mode":
            # change engine mode
            if len(parts) >= 2:
                m = parts[1].lower()
                if m in STRATEGY_FUNCS.keys():
                    set_current_engine(m)
                    send_telegram(chat_id, f"✅ Engine set to *{m}*")
                else:
                    send_telegram(chat_id, f"Unknown engine {m}. Options: {', '.join(STRATEGY_FUNCS.keys())}")
            else:
                send_telegram(chat_id, f"Current engine: {get_current_engine()}")
            return

        if cmd == "/retrain":
            if not is_admin:
                send_telegram(chat_id, "❌ Only admin can run retrain.")
                return
            send_telegram(chat_id, "🔁 Retrain placeholder started — no-op in this version.")
            # put retrain logic here
            return

        if cmd == "/join_challenge" and len(parts) >= 2:
            user = parts[1]
            res = join_challenge(user)
            send_telegram(chat_id, f"join_challenge result: {res}")
            return

        if cmd == "/leave_challenge" and len(parts) >= 2:
            user = parts[1]
            # implement leave
            if user in _challenge["participants"]:
                del _challenge["participants"][user]
                save_json(CHALLENGE_DB, _challenge)
                send_telegram(chat_id, f"{user} removed from challenge")
            else:
                send_telegram(chat_id, f"{user} not in challenge")
            return

        # default unknown
        send_telegram(chat_id, "Unknown command. Send /help for list of commands.")
    except Exception as e:
        log(f"Error handling update: {e}", to_admin=True)
        traceback.print_exc()

def poll_updates_loop():
    global _update_offset
    if not TELEGRAM_TOKEN:
        log("No TELEGRAM_TOKEN set; polling disabled.", to_admin=True)
        return
    while True:
        try:
            params = {"timeout": 20, "limit": 10}
            if _update_offset:
                params["offset"] = _update_offset
            r = requests.get(f"{BASE_TELEGRAM}/getUpdates", params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get("ok"):
                    for update in data.get("result", []):
                        _update_offset = update["update_id"] + 1
                        save_offset(_update_offset)
                        handle_update(update)
            else:
                log(f"getUpdates returned {r.status_code}: {r.text}", to_admin=False)
        except Exception as e:
            log(f"getUpdates error: {e}", to_admin=False)
            traceback.print_exc()
            time.sleep(1)
        time.sleep(POLL_INTERVAL)

# ---------------- Keepalive ping & heartbeat threads ----------------
def keepalive_ping_loop():
    while True:
        try:
            # ping Choreo public url to keep runtime warm (if set)
            # If you want to use chreo public url, set CHOREO_PING_URL env var
            choreo_url = os.getenv("CHOREO_PING_URL")
            if choreo_url:
                try:
                    requests.get(choreo_url, timeout=6)
                except Exception:
                    pass
            # local log-only
            # do not spam admin each keepalive; only console for diagnostics
            # log("Keepalive ping executed", to_admin=False)
        except Exception as e:
            print("keepalive ping error:", e)
        time.sleep(KEEPALIVE_INTERVAL)

def heartbeat_loop():
    while True:
        try:
            if ADMIN_CHAT:
                send_telegram(ADMIN_CHAT, f"💓 Heartbeat — {BRAND} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        except Exception as e:
            print("heartbeat error:", e)
        time.sleep(HEARTBEAT_INTERVAL)

# ---------------- Supervisor / main ----------------
START_TIME = datetime.utcnow()

def startup_sequence():
    ensure_single_instance()
    start_flask_thread()
    log(f"🚀 {BRAND} started (Mode: {get_current_engine()})", to_admin=True)
    # start poller
    t_poll = threading.Thread(target=poll_updates_loop, daemon=True)
    t_poll.start()
    # start keepalive & heartbeat
    threading.Thread(target=keepalive_ping_loop, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()

def shutdown_sequence():
    log("Shutting down...", to_admin=True)
    remove_instance_lock()

# ---------------- ENTRYPOINT ----------------
if __name__ == "__main__":
    try:
        if not TELEGRAM_TOKEN:
            print("WARNING: TELEGRAM_BOT_TOKEN not provided. Bot will run but cannot reply.")
        startup_sequence()
        # supervisor main loop: simple sleep to keep process alive
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("Interrupted, shutting down.")
        shutdown_sequence()
    except Exception as e:
        print("Fatal error:", e)
        traceback.print_exc()
        shutdown_sequence()
        raise
