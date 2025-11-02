import os
import time
import threading
import random
from flask import Flask
from telegram import Update, Bot
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ===========================================
# 🔐 ENVIRONMENT VARIABLES
# ===========================================
BOT_TOKEN = os.getenv("BOT_TOKEN", "YOUR_TELEGRAM_BOT_TOKEN_HERE")
ADMIN_ID = int(os.getenv("ADMIN_ID", "123456789"))
PORT = int(os.getenv("PORT", "8080"))

# ===========================================
# 🌍 FLASK KEEP ALIVE SERVER
# ===========================================
app = Flask(__name__)

@app.route('/')
def home():
    return "✅ Pocket Option AI Decision Bot is Alive!"

def run_keepalive():
    app.run(host="0.0.0.0", port=PORT)

def start_keepalive():
    t = threading.Thread(target=run_keepalive)
    t.daemon = True
    t.start()

# ===========================================
# ⚙️ GLOBAL STATE
# ===========================================
bot_mode = "decision"  # "decision" or "signal+trade"
last_signal = None
trade_history = []

# ===========================================
# ⚡ CORE DECISION LOGIC (SIMULATED)
# ===========================================
def generate_decision():
    """Simulate AI decision based on quantum + momentum + volume logic"""
    logic = random.choice(["Quantum", "Momentum", "Order Block", "Volume Spike"])
    decision = random.choice(["BUY", "SELL"])
    confidence = round(random.uniform(80, 98), 2)
    return {
        "decision": decision,
        "logic": logic,
        "confidence": confidence,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

# ===========================================
# 🤖 TELEGRAM BOT COMMANDS
# ===========================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *Pocket Option Institutional Bot*\n\n"
        "Commands:\n"
        "• /trade - run AI decision instantly\n"
        "• /mode - switch between Decision & Signal+Trade\n"
        "• /status - show live status\n"
        "• /history - view last 5 trades\n",
        parse_mode="Markdown"
    )

async def trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global last_signal, trade_history
    user_id = update.effective_user.id
    if user_id != ADMIN_ID:
        await update.message.reply_text("🚫 Unauthorized access.")
        return

    await update.message.reply_text("🧠 Running AI analysis... please wait")

    data = generate_decision()
    last_signal = data
    trade_history.insert(0, data)
    trade_history = trade_history[:5]

    msg = (
        f"📊 *Decision:* {data['decision']}\n"
        f"🧩 Logic: {data['logic']}\n"
        f"⚙️ Mode: {bot_mode.upper()}\n"
        f"📈 Confidence: {data['confidence']}%\n"
        f"🕒 Time: {data['timestamp']}"
    )

    await update.message.reply_text(msg, parse_mode="Markdown")

async def mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global bot_mode
    user_id = update.effective_user.id
    if user_id != ADMIN_ID:
        await update.message.reply_text("🚫 Unauthorized access.")
        return

    bot_mode = "signal+trade" if bot_mode == "decision" else "decision"
    await update.message.reply_text(f"🔁 Mode switched to: *{bot_mode.upper()}*", parse_mode="Markdown")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"🟢 *Bot Status*\n"
        f"Mode: {bot_mode.upper()}\n"
        f"Last Signal: {last_signal['decision'] if last_signal else 'None yet'}\n"
        f"Confidence: {last_signal['confidence']}%" if last_signal else ""
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not trade_history:
        await update.message.reply_text("📭 No trade history yet.")
        return

    msg = "📜 *Last 5 Trades:*\n"
    for i, t in enumerate(trade_history, start=1):
        msg += f"\n{i}. {t['decision']} ({t['confidence']}%) [{t['logic']}]"
    await update.message.reply_text(msg, parse_mode="Markdown")

# ===========================================
# 🚀 MAIN RUNNER
# ===========================================
def main():
    print("🚀 Starting Pocket Option Bot...")
    start_keepalive()

    app_telegram = ApplicationBuilder().token(BOT_TOKEN).build()

    app_telegram.add_handler(CommandHandler("start", start))
    app_telegram.add_handler(CommandHandler("trade", trade))
    app_telegram.add_handler(CommandHandler("mode", mode))
    app_telegram.add_handler(CommandHandler("status", status))
    app_telegram.add_handler(CommandHandler("history", history))

    print("🤖 Telegram bot active — awaiting commands...")
    app_telegram.run_polling()

if __name__ == "__main__":
    main()
