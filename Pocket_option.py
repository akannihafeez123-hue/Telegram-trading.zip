#!/usr/bin/env python3
# pocketoption_unified_bot_modeA_v10_choreo_direct_commands.py
# Pocket Master AI — ModeA v10 (PocketOption, direct Telegram, command-driven, Choreo-ready)
# - Decision-only (Pocket Option) mode
# - No endpoint/webhook (uses getUpdates long-polling)
# - Commands: /analyze, /trade, /status, /retrain, /help, challenge/member commands
# - Keep-alive + heartbeat + auto-restart
# - Uses ModeA v10 strategies, aggregation, and decision logic (from user's original file)

import os, time, threading, math, json, random, traceback
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG ----------------
BRAND = os.getenv("BRAND", "Pocket Master AI")
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "@kleezband")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ADMIN_CHAT = os.getenv("TELEGRAM_ADMIN_CHAT", "")  # numeric chat id or @username
TELEGRAM_GROUP_BASIC = os.getenv("TELEGRAM_GROUP_BASIC", "")  # optional group/channel id
TELEGRAM_GROUP_VIP = os.getenv("TELEGRAM_GROUP_VIP", "")      # optional vip group/channel id

# Optional integrations
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK", "")
SYNDICATE_ENDPOINT = os.getenv("SYNDICATE_ENDPOINT", "")

HF_API_KEY = os.getenv("HF_API_KEY", "")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")

# Persistence
MODEL_DIR = os.getenv("MODEL_DIR", "/tmp/po_models_v10_full")
os.makedirs(MODEL_DIR, exist_ok=True)
MEMBERS_FILE = os.path.join(MODEL_DIR, "members_v10.json")
PAYMENTS_DB = os.path.join(MODEL_DIR, "payments_v10.json")
CHALLENGE_DB = os.path.join(MODEL_DIR, "challenge_v10.json")
MODELS_FILE = os.path.join(MODEL_DIR, "models_v10.joblib")

# Decision thresholds
CONF_THRESHOLD_PRIORITY = float(os.getenv("CONF_THRESHOLD_PRIORITY", "90.0"))
CONF_THRESHOLD_FLEX_MIN = float(os.getenv("CONF_THRESHOLD_FLEX_MIN", "80.0"))
MIN_TF_ALIGN_REQUIRED = int(os.getenv("MIN_TF_ALIGN_REQUIRED", "4"))
MIN_STRAT_ALIGN_REQUIRED = int(os.getenv("MIN_STRAT_ALIGN_REQUIRED", "4"))
TOP_STRATS = int(os.getenv("TOP_STRATS", "5"))
TOP_TFS = int(os.getenv("TOP_TFS", "5"))

ANALYSIS_TFS = ["1m","2m","3m","5m","15m","30m","1h","2h","4h","1d","1wk","1mo"]

# Challenge config
CHALLENGE_MAX_SIGNALS_PER_DAY = int(os.getenv("CHALLENGE_MAX_SIGNALS_PER_DAY", "5"))
CHALLENGE_DEPOSIT = float(os.getenv("CHALLENGE_DEPOSIT", "100.0"))
CHALLENGE_GOAL = float(os.getenv("CHALLENGE_GOAL", "1000.0"))
CHALLENGE_DURATION_DAYS = int(os.getenv("CHALLENGE_DURATION_DAYS", "7"))
CHALLENGE_STAKE_PERCENT = float(os.getenv("CHALLENGE_STAKE_PERCENT", "3.0"))

# Scanner and polling config
SCAN_INTERVAL = float(os.getenv("SCAN_INTERVAL", "30.0"))
POLL_INTERVAL = float(os.getenv("POLL_INTERVAL", "2.0"))  # getUpdates polling
PO_SYMBOLS = os.getenv("PO_SYMBOLS", "EURUSD,GBPJPY,BTC-USD,USDJPY,AUDUSD").split(",")

# Keepalive / heartbeat
KEEPALIVE_INTERVAL = int(os.getenv("KEEPALIVE_INTERVAL", "300"))   # ping every 5 minutes
HEARTBEAT_INTERVAL = int(os.getenv("HEARTBEAT_INTERVAL", "1800"))  # Telegram heartbeat every 30min

# ---------------- persistence helpers ----------------
def load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default

def save_json(path, obj):
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(obj, f, indent=2, default=str)
    except Exception as e:
        log_console(f"save_json error: {e}")

_members = load_json(MEMBERS_FILE, {})
_payments = load_json(PAYMENTS_DB, {})
_challenge = load_json(CHALLENGE_DB, {"participants": {}, "signals": []})

# ---------------- telegram helpers (getUpdates polling) ----------------
BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

def send_telegram(chat_id, text, parse_mode="Markdown"):
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False, "tg_not_configured"
    try:
        url = f"{BASE_URL}/sendMessage"
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        r = requests.post(url, json=payload, timeout=8)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def log_console(s):
    print(s)

def log(message, to_admin=True, to_console=True, silent_on_fail=False):
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    formatted = f"[{timestamp}] {message}"
    if to_console:
        print(formatted)
    if to_admin and TELEGRAM_ADMIN_CHAT:
        ok, resp = send_telegram(TELEGRAM_ADMIN_CHAT, formatted)
        if not ok and not silent_on_fail:
            log_console(f"⚠️ Telegram send failed: {resp}")

# ---------------- membership & challenge ----------------
def ensure_member_record(user_telegram):
    global _members
    if user_telegram not in _members:
        _members[user_telegram] = {"tier":"none","expires":None,"joined": datetime.utcnow().isoformat()}
        save_json(MEMBERS_FILE, _members)
    return _members[user_telegram]

def is_member_active(user_telegram, tier=None):
    rec = _members.get(user_telegram)
    if not rec: return False
    if rec.get("tier") == "none": return False
    exp = rec.get("expires")
    if not exp: return False
    try:
        expires = datetime.fromisoformat(exp)
        ok = datetime.utcnow() < expires
        if tier is None: return ok
        if tier == "basic": return ok and rec.get("tier") in ("basic","vip")
        return ok and rec.get("tier")==tier
    except Exception:
        return False

def add_member_by_payment(user_telegram, tier, days=30):
    rec = ensure_member_record(user_telegram)
    rec["tier"] = tier
    rec["expires"] = (datetime.utcnow() + timedelta(days=days)).isoformat()
    save_json(MEMBERS_FILE, _members)
    return rec

def revoke_member(user_telegram):
    rec = ensure_member_record(user_telegram)
    rec["tier"] = "none"; rec["expires"] = None
    save_json(MEMBERS_FILE, _members)
    return rec

def join_challenge(user_telegram):
    if not is_member_active(user_telegram): return {"ok":False, "reason":"not_active_member"}
    if user_telegram in _challenge["participants"]: return {"ok":False, "reason":"already_joined"}
    p = {"joined": datetime.utcnow().isoformat(), "virtual_balance": CHALLENGE_DEPOSIT, "signals_received_today":0,
         "total_signals_received":0, "wins":0, "losses":0, "last_active": datetime.utcnow().isoformat(), "history": []}
    _challenge["participants"][user_telegram] = p
    save_json(CHALLENGE_DB, _challenge)
    return {"ok":True, "participant":p}

def leave_challenge(user_telegram):
    if user_telegram in _challenge["participants"]:
        del _challenge["participants"][user_telegram]
        save_json(CHALLENGE_DB, _challenge)
        return {"ok":True}
    return {"ok":False, "reason":"not_in_challenge"}

def record_challenge_signal(user_telegram, signal_id, action, stake_amount, expiry_seconds, outcome=None):
    p = _challenge["participants"].get(user_telegram)
    if not p: return {"ok":False, "reason":"not_participant"}
    p["signals_received_today"] = p.get("signals_received_today",0)+1
    p["total_signals_received"] = p.get("total_signals_received",0)+1
    p["last_active"] = datetime.utcnow().isoformat()
    entry = {"ts": datetime.utcnow().isoformat(), "signal_id": signal_id, "action":action, "stake":stake_amount, "expiry":expiry_seconds, "outcome": outcome}
    p["history"].append(entry)
    if outcome is not None:
        payout = 0.8
        if outcome:
            profit = stake_amount * payout
            p["virtual_balance"] += profit; p["wins"] = p.get("wins",0)+1
        else:
            p["virtual_balance"] -= stake_amount; p["losses"] = p.get("losses",0)+1
    save_json(CHALLENGE_DB, _challenge)
    return {"ok":True, "participant":p}

def reset_daily_challenge_counters():
    for user,p in _challenge["participants"].items():
        p["signals_received_today"] = 0
    save_json(CHALLENGE_DB, _challenge)

# ---------------- Technical indicators & strategies ----------------
def ema(series, span): return series.ewm(span=span, adjust=False).mean()
def rsi(series, period=14):
    delta = series.diff(); up = delta.clip(lower=0); down = -1*delta.clip(upper=0)
    ema_up = up.ewm(com=(period-1), adjust=False).mean(); ema_down = down.ewm(com=(period-1), adjust=False).mean()
    rs = ema_up/(ema_down+1e-9); return 100 - (100/(1+rs))

def quantum_engine_v2(df):
    try:
        if df is None or df.empty: return 0.0,0.0
        c = df["Close"].dropna()
        if len(c)<20: return 0.0,0.0
        e8 = ema(c,8); e21 = ema(c,21)
        macd = e8 - e21; macd_sig = macd.ewm(span=9,adjust=False).mean(); hist = macd - macd_sig
        ema_cross_sign = np.sign(e8.iloc[-1] - e21.iloc[-1])
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        vol_trend = np.polyfit(range(min(10,len(vol.dropna()))), vol.dropna().values[-min(10,len(vol.dropna())):],1)[0] if len(vol.dropna())>=3 else 0.0
        vol_score = 1 if vol_trend>0 else -1 if vol_trend<0 else 0
        buy_votes = 0
        buy_votes += 1 if (ema_cross_sign>0 and hist.iloc[-1]>0) else 0
        buy_votes += 1 if vol_score>0 else 0
        buy_votes += 1 if c.iloc[-1] > c.rolling(20).mean().iloc[-1] else 0
        score = (buy_votes - (3-buy_votes))/3.0
        conf = min(1.0, abs(hist.iloc[-1])/(np.std(hist[-10:])+1e-9) if len(hist)>=10 else 0.3)
        return float(np.clip(score,-1.0,1.0)), float(np.clip(conf,0.0,1.0))
    except Exception:
        return 0.0,0.0

def momentum_scalper_v1(df):
    try:
        if df is None or df.empty: return 0.0,0.0
        c = df["Close"].dropna()
        if len(c)<6: return 0.0,0.0
        last = c.iloc[-1]; prev5 = c.iloc[-6:-1]
        momentum = (last - prev5.mean())/(prev5.mean()+1e-9)
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        vol_spike = vol.iloc[-1] > (vol.rolling(5).mean().iloc[-1]*1.8 if len(vol)>=5 else vol.iloc[-1])
        rsi_val = rsi(c,14).iloc[-1] if len(c)>=15 else 50.0
        buy = 1 if (momentum>0 and vol_spike and rsi_val>50) else -1 if (momentum<0 and vol_spike and rsi_val<50) else 0
        conf = min(1.0, abs(momentum)*2.0 + (1.0 if vol_spike else 0.0))
        return float(np.clip(buy,-1.0,1.0)), float(np.clip(conf,0.0,1.0))
    except Exception:
        return 0.0,0.0

def breakout_hunter_v1(df):
    try:
        if df is None or df.empty: return 0.0,0.0
        h = df["High"].dropna(); l = df["Low"].dropna(); c = df["Close"].dropna()
        if len(c)<20: return 0.0,0.0
        recent_high = h[-10:].max(); recent_low = l[-10:].min()
        last = c.iloc[-1]
        breakout_up = last > recent_high and c.iloc[-2] <= recent_high
        breakout_down = last < recent_low and c.iloc[-2] >= recent_low
        vol = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        vol_confirm = vol.iloc[-1] > (vol.rolling(10).mean().iloc[-1]*1.5 if len(vol)>=10 else vol.iloc[-1])
        if breakout_up and vol_confirm: return 1.0,0.9
        if breakout_down and vol_confirm: return -1.0,0.9
        return 0.0,0.0
    except Exception:
        return 0.0,0.0

def mean_reversion_v1(df):
    try:
        if df is None or df.empty: return 0.0,0.0
        c = df["Close"].dropna()
        if len(c)<20: return 0.0,0.0
        ma = c.rolling(20).mean(); std = c.rolling(20).std()
        lower = ma - 2*std; upper = ma + 2*std
        last = c.iloc[-1]
        if last < lower.iloc[-1]: return 1.0,0.8
        if last > upper.iloc[-1]: return -1.0,0.8
        return 0.0,0.0
    except Exception:
        return 0.0,0.0

STRATEGY_FUNCS = {
    "quantum": quantum_engine_v2,
    "momentum": momentum_scalper_v1,
    "breakout": breakout_hunter_v1,
    "meanreversion": mean_reversion_v1
}

# ---------------- Hidden Liquidity & desk logic ----------------
def dark_pool_internal_model(df):
    try:
        if df is None or df.empty: return 0.0, 0.0
        c = df["Close"].dropna(); v = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        if len(c) < 10: return 0.0, 0.0
        ranges = (df["High"] - df["Low"]).dropna()
        recent_range = ranges[-5:].mean() if len(ranges)>=5 else ranges.mean()
        recent_vol = v[-5:].mean() if len(v.dropna())>=5 else v.mean()
        metric = 0.0
        if recent_range > (np.mean(ranges)+1e-9) and recent_vol < (v.rolling(10).mean().iloc[-1]+1e-9):
            metric = (recent_range / (np.mean(ranges)+1e-9)) - (recent_vol / (v.rolling(10).mean().iloc[-1]+1e-9))
        mid = (df["High"]+df["Low"])/2.0
        dir_sign = 1.0 if c.iloc[-1] > mid.iloc[-1] else -1.0
        score = np.tanh(metric) * dir_sign
        conf = min(1.0, abs(metric)/2.0)
        return float(np.clip(score,-1.0,1.0)), float(np.clip(conf,0.0,1.0))
    except Exception:
        return 0.0,0.0

def proprietary_desk_logic(df):
    try:
        if df is None or df.empty: return 0.0, 0.0, {}
        c = df["Close"].dropna(); v = df["Volume"] if "Volume" in df.columns else pd.Series(np.zeros(len(df)), index=df.index)
        if len(c) < 15: return 0.0, 0.0, {}
        recent_returns = c.pct_change().dropna()[-10:]
        drift = recent_returns.mean() if len(recent_returns)>0 else 0.0
        vol_trend = np.polyfit(range(min(10,len(v.dropna()))), v.dropna().values[-min(10,len(v.dropna())):],1)[0] if len(v.dropna())>=3 else 0.0
        now = datetime.utcnow(); hour = now.hour
        tod_bias = 0.2 if 8 <= hour <= 16 else 0.0
        score = np.tanh(drift*3.0 + vol_trend/1e6 + tod_bias)
        conf = min(1.0, (abs(drift)*10.0) + (abs(vol_trend)/1e6) + 0.2)
        meta = {"drift": drift, "vol_trend": vol_trend}
        return float(np.clip(score,-1.0,1.0)), float(np.clip(conf,0.0,1.0)), meta
    except Exception:
        return 0.0,0.0,{}

# ---------------- News (lightweight) ----------------
def hf_sentiment(text):
    if not HF_API_KEY: return 0.0
    try:
        url = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment-latest"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        resp = requests.post(url, headers=headers, json={"inputs": text}, timeout=6)
        if resp.status_code == 200:
            data = resp.json()
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
        if r.status_code==200:
            items = r.json().get("articles",[])
            scores = []
            for it in items:
                txt = (it.get("title","") or "") + ". " + (it.get("description","") or "")
                scores.append(hf_sentiment(txt) if HF_API_KEY else 0.0)
            return float(np.mean(scores)) if scores else 0.0
    except Exception:
        pass
    return 0.0

# ---------------- Aggregation & decision logic ----------------
def evaluate_strategies_on_df(df):
    out = {}
    for name, fn in STRATEGY_FUNCS.items():
        try:
            s,c = fn(df)
            out[name] = {"score":float(s),"conf":float(c)}
        except Exception:
            out[name] = {"score":0.0,"conf":0.0}
    dp_s, dp_c = dark_pool_internal_model(df)
    pa_s, pa_c, pa_meta = proprietary_desk_logic(df)
    out["darkpool"] = {"score":float(dp_s), "conf":float(dp_c)}
    out["proprietary"] = {"score":float(pa_s), "conf":float(pa_c), "meta": pa_meta}
    return out

def aggregate_across_tfs(symbol, tf_df_map):
    per_tf = {}; tf_votes = {}
    for tf, df in tf_df_map.items():
        strat = evaluate_strategies_on_df(df)
        per_tf[tf] = strat
        buy = 0.0; sell = 0.0; totalw = 0.0
        for sname, info in strat.items():
            w = info.get("conf",0.0)
            totalw += w
            if info.get("score",0.0) > 0:
                buy += info["score"] * w
            elif info.get("score",0.0) < 0:
                sell += -info["score"] * w
        if totalw == 0:
            tf_votes[tf] = {"vote":"HOLD","strength":0.0}
        else:
            if buy > sell: tf_votes[tf] = {"vote":"BUY","strength": float(buy/totalw)}
            elif sell > buy: tf_votes[tf] = {"vote":"SELL","strength": float(sell/totalw)}
            else: tf_votes[tf] = {"vote":"HOLD","strength":0.0}
    strat_agg = {}
    for tf, strat in per_tf.items():
        for s,info in strat.items():
            strat_agg[s] = strat_agg.get(s,0.0) + info.get("score",0.0) * info.get("conf",0.0)
    maxabs = max((abs(v) for v in strat_agg.values()), default=1.0)
    for k in strat_agg: strat_agg[k] = strat_agg[k]/(maxabs+1e-9)
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
        tag = "PRIORITY"; return {"action":final_action,"confidence":final_conf,"tag":tag,"top_strats":top_strats,"aligned_tfs":aligned_tfs,"strat_align_ratio":strat_align_ratio}
    if CONF_THRESHOLD_FLEX_MIN <= final_conf < CONF_THRESHOLD_PRIORITY:
        dp_ok = (abs(dp) > 0 and dp_conf >= 0.6)
        pa_ok = (abs(pa) > 0 and pa_conf >= 0.6)
        if dp_ok and pa_ok and strat_align_ratio >= 0.8:
            tag = "INSTITUTIONAL_FLEX"; return {"action":final_action,"confidence":final_conf,"tag":tag,"top_strats":top_strats,"aligned_tfs":aligned_tfs,"strat_align_ratio":strat_align_ratio}
        else:
            return {"action":"HOLD","confidence":final_conf,"reason":"failed_institutional_checks","dp":dp,"pa":pa,"dp_conf":dp_conf,"pa_conf":pa_conf}
    return {"action":"HOLD","confidence":final_conf,"reason":"below_threshold"}

# ---------------- Market data helpers (yfinance) ----------------
def parse_tf_to_pd_rule(tf):
    tf = tf.lower().strip()
    if tf.endswith("m") and tf[:-1].isdigit(): return f"{int(tf[:-1])}T"
    if tf.endswith("h") and tf[:-1].isdigit(): return f"{int(tf[:-1])}H"
    if tf in ("1d","1wk","1mo"): return {"1d":"1D","1wk":"1W","1mo":"1M"}.get(tf)
    return None

def fetch_market_data_yf(yf_sym, tfs, period="6mo"):
    import yfinance as yf
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
        if tf in ["1m","2m","5m","15m","30m","60m","1d","1wk","1mo"]: yf_int = tf if tf!="60m" else "1h"
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
        if rule is None: out[tf] = None; continue
        try:
            df_base = base.copy()
            if not isinstance(df_base.index, pd.DatetimeIndex):
                df_base.index = pd.to_datetime(df_base.index)
            df_res = df_base.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"}).dropna()
            out[tf] = df_res
        except Exception:
            out[tf] = None
    return out

# ---------------- Publishing helpers ----------------
def publish_to_telegram(chat_id, text):
    ok, resp = send_telegram(chat_id, text)
    return ok, resp

def post_discord(text):
    if not DISCORD_WEBHOOK: return False, "discord_not_configured"
    try:
        r = requests.post(DISCORD_WEBHOOK, json={"content": text}, timeout=8)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

def post_endpoint(payload):
    if not SYNDICATE_ENDPOINT: return False, "endpoint_not_configured"
    try:
        r = requests.post(SYNDICATE_ENDPOINT, json=payload, timeout=8)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)

# Broadcast (respects members/challenge)
def broadcast_signal_to_members(signal_payload, tier="basic"):
    sent = {"ok":[], "skipped":[], "errors":[]}
    recipients = []
    for user,rec in _members.items():
        if is_member_active(user, tier="basic"):
            recipients.append((user, rec["tier"]))
    if tier=="vip":
        recipients = [(u,r) for (u,r) in recipients if _members.get(u,{}).get("tier")=="vip"]
    for user, t in recipients:
        if user in _challenge["participants"]:
            p = _challenge["participants"][user]
            if p.get("signals_received_today",0) >= CHALLENGE_MAX_SIGNALS_PER_DAY:
                sent["skipped"].append({"user":user,"reason":"max_signals_today"}); continue
        group = TELEGRAM_GROUP_VIP if _members[user]["tier"]=="vip" else TELEGRAM_GROUP_BASIC
        try:
            label = "[CHALLENGE]" if signal_payload.get("tag","") in ("PRIORITY","INSTITUTIONAL_FLEX") else signal_payload.get("tag","")
            msg = f"{label} {signal_payload['symbol']} → {signal_payload['action']} | conf {signal_payload['confidence']:.1f}% | tag: {signal_payload.get('tag','')}\n{BRAND}"
            ok, resp = publish_to_telegram(group or user, msg)
            if ok:
                sent["ok"].append(user)
                if user in _challenge["participants"]:
                    stake = (_challenge["participants"][user]["virtual_balance"] * CHALLENGE_STAKE_PERCENT/100.0)
                    record_challenge_signal(user, signal_payload.get("id","na"), signal_payload["action"], stake, signal_payload.get("expiry",180))
            else:
                sent["errors"].append({"user":user,"error":resp})
        except Exception as e:
            sent["errors"].append({"user":user,"error":str(e)})
    return sent

# ---------------- Scanner loop (used for /analyze or /trade) ----------------
def scanner_cycle_once(symbols):
    results = []
    for s in symbols:
        try:
            data_map = fetch_market_data_yf(s, ANALYSIS_TFS, period="6mo")
            available = [tf for tf in ANALYSIS_TFS if data_map.get(tf) is not None]
            available_sorted = sorted(available, key=lambda x: ANALYSIS_TFS.index(x) if x in ANALYSIS_TFS else 0, reverse=True)
            selected = available_sorted[:TOP_TFS]
            tf_df_map = {tf: data_map.get(tf) for tf in selected}
            news_bias = fetch_headlines_and_score(s) if (NEWSAPI_KEY or HF_API_KEY) else 0.0
            decision = decide_for_symbol(s, tf_df_map, news_bias=news_bias)
            if decision.get("action") in ("BUY","SELL") and decision.get("confidence",0.0) >= CONF_THRESHOLD_FLEX_MIN:
                payload = {"id": f"{s}_{int(time.time())}", "symbol":s, "action":decision["action"], "confidence":decision["confidence"], "tag": decision.get("tag",""), "top_strats":decision.get("top_strats"), "aligned_tfs":decision.get("aligned_tfs")}
                results.append(payload)
            else:
                # include held signals for transparency (optional)
                results.append({"id": f"{s}_{int(time.time())}", "symbol":s, "action": "HOLD", "confidence": decision.get("confidence",0.0), "reason": decision.get("reason","")})
        except Exception as e:
            log(f"scan error for {s}: {e}", to_admin=False)
            traceback.print_exc()
    return results

# ---------------- Keepalive and heartbeat ----------------
def keep_alive_ping():
    while True:
        try:
            requests.get("https://choreo.dev", timeout=6)
            log("🌐 Keep-alive ping sent (choreo).", to_admin=False)
        except Exception as e:
            log(f"⚠️ Keep-alive ping failure: {e}", to_admin=False)
        time.sleep(KEEPALIVE_INTERVAL)

def heartbeat_loop():
    while True:
        try:
            if TELEGRAM_ADMIN_CHAT:
                send_telegram(TELEGRAM_ADMIN_CHAT, f"💓 Heartbeat: {BRAND} — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}")
        except Exception as e:
            log(f"Heartbeat send failure: {e}", to_admin=False)
        time.sleep(HEARTBEAT_INTERVAL)

# ---------------- Telegram command polling ----------------
_update_offset = 0
_last_activity = datetime.utcnow()

def handle_update(update):
    global _last_activity
    try:
        if "message" not in update: return
        msg = update["message"]
        chat = msg.get("chat",{})
        chat_id = chat.get("id")
        from_user = msg.get("from",{})
        user_name = from_user.get("username") or from_user.get("first_name") or str(chat_id)
        text = msg.get("text","").strip()
        _last_activity = datetime.utcnow()

        # Admin-only actions: check admin id/username
        is_admin = False
        if TELEGRAM_ADMIN_CHAT:
            try:
                # compare numeric ids if ADMIN chat is numeric
                if str(TELEGRAM_ADMIN_CHAT).lstrip('@') == str(chat_id) or str(TELEGRAM_ADMIN_CHAT).lstrip('@') == str(user_name):
                    is_admin = True
            except Exception:
                pass

        # parse commands
        if not text:
            return

        parts = text.split()
        cmd = parts[0].lower()

        if cmd in ("/help", "/start"):
            help_msg = (
                "🤖 Pocket Master AI — commands:\n\n"
                "/analyze [SYMBOL] — run analysis (if symbol omitted, uses default list)\n"
                "/trade [SYMBOL] — run analysis + instantly publish decision signal (Pocket Option decision only)\n"
                "/status — show bot status and uptime\n"
                "/retrain — (admin) run retrain placeholder\n"
                "/join_challenge @user — join challenge (user must be active member)\n"
                "/leave_challenge @user — leave challenge\n"
                "/my_challenge_status @user — show challenge status\n"
                "/list_members — (admin) list members\n"
            )
            send_telegram(chat_id, help_msg)
            return

        if cmd == "/status":
            uptime = datetime.utcnow() - START_TIME
            status_msg = f"✅ {BRAND} status\nUptime: {str(uptime).split('.')[0]}\nLast activity: {_last_activity.strftime('%Y-%m-%d %H:%M:%S UTC')}\nSymbols: {','.join(PO_SYMBOLS)}"
            send_telegram(chat_id, status_msg)
            return

        if cmd == "/analyze":
            syms = parts[1:] if len(parts)>1 else PO_SYMBOLS
            send_telegram(chat_id, f"🔎 Running analysis for: {','.join(syms)}. This may take a moment.")
            res = scanner_cycle_once(syms)
            out = []
            for r in res:
                if r.get("action") in ("BUY","SELL"):
                    out.append(f"{r['symbol']}: {r['action']} ({r['confidence']:.1f}%) [{r.get('tag','')}]")
                else:
                    out.append(f"{r['symbol']}: HOLD ({r.get('confidence',0.0):.1f}%) {r.get('reason','')}")
            send_telegram(chat_id, "🔎 Analysis results:\n" + "\n".join(out))
            return

        if cmd == "/trade":
            # Instant execution: run analysis for symbol(s) and publish signals
            syms = parts[1:] if len(parts)>1 else PO_SYMBOLS
            send_telegram(chat_id, f"⚡ Executing trade analysis for: {','.join(syms)}")
            res = scanner_cycle_once(syms)
            executed = []
            for r in res:
                if r.get("action") in ("BUY","SELL"):
                    # publish to admin & groups
                    msg = f"📡 TRADE SIGNAL — {r['symbol']} → *{r['action']}*\nConfidence: *{r['confidence']:.1f}%*\nTag: {r.get('tag','')}\n{BRAND}"
                    send_telegram(TELEGRAM_ADMIN_CHAT, msg)
                    if TELEGRAM_GROUP_VIP:
                        send_telegram(TELEGRAM_GROUP_VIP, msg)
                    if TELEGRAM_GROUP_BASIC:
                        send_telegram(TELEGRAM_GROUP_BASIC, msg)
                    # record for challenge participants
                    for user in list(_challenge["participants"].keys()):
                        try:
                            stake = (_challenge["participants"][user]["virtual_balance"] * CHALLENGE_STAKE_PERCENT/100.0)
                            record_challenge_signal(user, r.get("id","na"), r["action"], stake, expiry_seconds=180)
                        except Exception:
                            pass
                    executed.append(f"{r['symbol']}:{r['action']}({r['confidence']:.1f}%)")
                else:
                    executed.append(f"{r['symbol']}:HOLD({r.get('confidence',0.0):.1f}%)")
            send_telegram(chat_id, "✅ Trade run complete:\n" + "\n".join(executed))
            return

        if cmd == "/retrain":
            if not is_admin:
                send_telegram(chat_id, "❌ Only admin can run retrain.")
                return
            send_telegram(chat_id, "🔁 Retrain placeholder started — (no-op in this version).")
            # Place retrain logic here if you have training data/models
            return

        if cmd == "/join_challenge" and len(parts)>=2:
            user = parts[1]
            res = join_challenge(user)
            send_telegram(chat_id, f"join_challenge result: {res}")
            return

        if cmd == "/leave_challenge" and len(parts)>=2:
            user = parts[1]
            res = leave_challenge(user)
            send_telegram(chat_id, f"leave_challenge result: {res}")
            return

        if cmd == "/my_challenge_status" and len(parts)>=2:
            user = parts[1]
            send_telegram(chat_id, str(_challenge["participants"].get(user, "not_found")))
            return

        if cmd == "/list_members":
            if not is_admin:
                send_telegram(chat_id, "❌ Only admin can list members.")
                return
            send_telegram(chat_id, "Members:\n" + "\n".join([f"{u} => {r}" for u,r in _members.items()][:200]))
            return

        # Unknown
        send_telegram(chat_id, "Unknown command. Send /help for list of commands.")
    except Exception as e:
        log(f"Error handling update: {e}", to_admin=True)
        traceback.print_exc()

def poll_updates_loop():
    global _update_offset
    while True:
        try:
            url = f"{BASE_URL}/getUpdates"
            params = {"timeout": 20, "offset": _update_offset, "limit": 10}
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                data = r.json()
                if data.get("ok"):
                    for update in data.get("result", []):
                        _update_offset = update["update_id"] + 1
                        handle_update(update)
            else:
                log(f"getUpdates returned {r.status_code}: {r.text}", to_admin=False)
        except Exception as e:
            log(f"getUpdates error: {e}", to_admin=False)
            traceback.print_exc()
            time.sleep(1)
        time.sleep(POLL_INTERVAL)

# ---------------- Supervisor / main daemon ----------------
_scanner_thread = None
_scanner_stop_evt = None

def start_scanner(symbols, interval=SCAN_INTERVAL):
    global _scanner_thread, _scanner_stop_evt
    if _scanner_thread and _scanner_thread.is_alive():
        log("Scanner already running"); return
    _scanner_stop_evt = threading.Event()
    def _loop():
        while not _scanner_stop_evt.is_set():
            try:
                res = scanner_cycle_once(symbols)
                # only used when running automated scanning; in command-driven mode we don't auto-publish
                # but we can log priority signals to admin
                for p in res:
                    try:
                        if p.get("action") in ("BUY","SELL") and p.get("confidence",0.0) >= CONF_THRESHOLD_PRIORITY:
                            send_telegram(TELEGRAM_ADMIN_CHAT, f"[AUTO-PRIORITY] {p['symbol']} → {p['action']} | {p['confidence']:.1f}%")
                    except Exception:
                        pass
                time.sleep(interval)
            except Exception as e:
                log(f"Scanner loop error: {e}", to_admin=False)
                traceback.print_exc()
                time.sleep(5)
    _scanner_thread = threading.Thread(target=_loop, daemon=True)
    _scanner_thread.start()
    log("Scanner started (background)")

def stop_scanner():
    global _scanner_thread, _scanner_stop_evt
    if _scanner_stop_evt: _scanner_stop_evt.set()
    if _scanner_thread: _scanner_thread.join(timeout=3.0)
    _scanner_thread = None; _scanner_stop_evt = None
    log("Scanner stopped")

START_TIME = datetime.utcnow()

def run_main_daemon():
    # start keepalive & heartbeat
    threading.Thread(target=keep_alive_ping, daemon=True).start()
    threading.Thread(target=heartbeat_loop, daemon=True).start()
    # start Telegram polling
    threading.Thread(target=poll_updates_loop, daemon=True).start()
    # optionally start background scanner if desired (not mandatory in command-only mode)
    # threading.Thread(target=start_scanner, args=(PO_SYMBOLS,SCAN_INTERVAL), daemon=True).start()

    log(f"🚀 {BRAND} — Pocket Option Mode (Command-driven). Polling for commands...", to_admin=True)
    # Supervisor loop to keep alive
    while True:
        try:
            time.sleep(5)
            # daily reset of challenge counters at midnight UTC
            now = datetime.utcnow()
            if now.hour == 0 and now.minute == 0 and now.second < 6:
                reset_daily_challenge_counters()
            # nothing else; commands and threads handle work
        except Exception as e:
            log(f"🔥 Supervisor caught exception: {e}. Restarting internals...", to_admin=True)
            traceback.print_exc()
            try:
                stop_scanner()
            except Exception:
                pass
            time.sleep(5)

if __name__ == "__main__":
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_ADMIN_CHAT:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_ADMIN_CHAT must be set in environment variables.")
        raise SystemExit("Missing TELEGRAM config")
    startup_msg = (
        f"🚀 {BRAND} — Pocket Option Mode (Command-driven)\n"
        f"Started at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
        f"Commands: /analyze /trade /status /retrain /help\n"
        f"No endpoint mode — using Telegram polling (getUpdates)."
    )
    log(startup_msg, to_admin=True)
    try:
        run_main_daemon()
    except KeyboardInterrupt:
        log("Shutdown requested by user. Exiting.", to_admin=True)
    except Exception as e:
        log(f"Critical failure: {e}", to_admin=True)
        traceback.print_exc()
