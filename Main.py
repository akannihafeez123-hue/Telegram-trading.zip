import os
import json
import time
import random
import logging
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
from strategies import StrategyEngine
from trading import TradeExecutor
from logger import SummaryLogger

# ✅ Load environment variables
BOT_TOKEN = os.environ.get("BOT_TOKEN")
ADMIN_CHAT_ID = os.environ.get("ADMIN_CHAT_ID")
BYBIT_API_KEY = os.environ.get("BYBIT_API_KEY")
BYBIT_API_SECRET = os.environ.get("BYBIT_API_SECRET")
DEMO_MODE = os.environ.get("DEMO_MODE", "true").lower() == "true"
CONF_THRESH = float(os.environ.get("CONF_THRESH", 0.75))
LEVERAGE = int(os.environ.get("LEVERAGE", 5))
RISK_PCT = float(os.environ.get("RISK_PCT", 0.02))

# ✅ Initialize core components
engine = StrategyEngine()
executor = TradeExecutor(api_key=BYBIT_API_KEY, api_secret=BYBIT_API_SECRET, demo=DEMO_MODE)
logger = SummaryLogger(path="trade_attempts.csv")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("QuantumBot")

# ⚙️ Helper — format result message
def format_alignment(alignment):
    result = []
    for sym, data in alignment.items():
        result.append(f"🔹 {sym} → {data['direction'].upper()} ({data['confidence']*100:.1f}%)")
    return "\n".join(result)

# 🚀 Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🚀 Quantum AutoTrader Online!\n"
        "Available Engines:\n"
        "• /quantum\n• /momentum\n• /breakout\n• /meanreversion\n\n"
        "Auto alignment & smart execution enabled ✅"
    )

async def quantum(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⚛️ Running Quantum Engine Alignment... please wait...")
    alignment = engine.run_all("QUANTUM")
    summary = format_alignment(alignment)
    await update.message.reply_text(summary)

    # Filter by confidence
    strong_signals = {s: d for s, d in alignment.items() if d['confidence'] >= CONF_THRESH}
    if strong_signals:
        for symbol, data in strong_signals.items():
            direction = data['direction']
            confidence = data['confidence']
            logger.log(symbol, direction, confidence, "EXECUTED")
            executor.execute_trade(symbol, direction, leverage=LEVERAGE, risk_pct=RISK_PCT)
            await update.message.reply_text(f"✅ {symbol}: {direction.upper()} ({confidence*100:.1f}%) executed!")
    else:
        logger.log("ALL", "SKIP", 0, "NO SIGNAL")
        await update.message.reply_text("⚠️ No alignment reached 75% confidence. Trade skipped.")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(ADMIN_CHAT_ID):
        return await update.message.reply_text("🚫 Not authorized.")
    msg = " ".join(context.args)
    await context.bot.send_message(chat_id=ADMIN_CHAT_ID, text=f"📢 Broadcast:\n{msg}")

# 🧠 Register commands
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("quantum", quantum))
    app.add_handler(CommandHandler("broadcast", broadcast))
    log.info("🚀 Quantum AutoTrader started!")
    app.run_polling()

if __name__ == "__main__":
    main()
