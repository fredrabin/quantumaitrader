import flask
from flask import Flask, request, jsonify
import numpy as np
# TensorFlow removed for cloud deployment
import logging
import requests
import os
import pandas as pd
from datetime import datetime
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import threading
import time
import asyncio

# Configuration Flask
app = Flask(__name__)

# Configuration pour scalping
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "USDJPY", "EURUSD", "SP500"]
SCALPING_CONFIG = {
    "XAUUSD": {"base_sl": 0.0015, "base_tp": 0.0030, "risk": 0.4},
    "BTCUSD": {"base_sl": 0.0020, "base_tp": 0.0040, "risk": 0.3},
    "ETHUSD": {"base_sl": 0.0025, "base_tp": 0.0050, "risk": 0.3},
    "USDJPY": {"base_sl": 0.0008, "base_tp": 0.0016, "risk": 0.5},
    "EURUSD": {"base_sl": 0.0006, "base_tp": 0.0012, "risk": 0.5},
    "SP500": {"base_sl": 0.0010, "base_tp": 0.0020, "risk": 0.4}
}

# Configuration Telegram
TELEGRAM_TOKEN = "8396377413:AAGtSWquXrolQR2LlqRdh3a75zd8Zt5UOfg"
CHAT_ID = None  # Sera défini quand vous démarrerez le bot

# Variables globales
bot = telegram.Bot(token=TELEGRAM_TOKEN)
trade_history = []

# Routes principales
@app.route('/')
def home():
    return "🚀 Quantum AI Trading Server - Cloud Version"

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "timestamp": datetime.utcnow().isoformat(),
        "symbols": SYMBOLS
    })

@app.route('/scalping-predict', methods=['POST'])
def scalping_predict():
    """Endpoint de prédiction pour le scalping"""
    try:
        data = request.get_json()
        symbol = data.get("symbol", "").upper()
        features = data.get("features", [])
        
        print(f"📥 Requête reçue - Symbol: {symbol}, Features: {len(features)}")
        
        if symbol not in SYMBOLS:
            return jsonify({"error": f"Symbole {symbol} non supporté"}), 400
        
        if len(features) != 25:
            return jsonify({"error": "25 features requises"}), 400
        
        # Simulation de prédiction (à remplacer par vos modèles)
        action, confidence, probabilities = simulate_prediction(symbol, features)
        
        # Calcul des paramètres de scalping
        sl, tp, lots = calculate_scalping_parameters(symbol, confidence)
        
        response = {
            "symbol": symbol,
            "action": action,  # -1=SELL, 0=HOLD, 1=BUY
            "confidence": confidence,
            "scalping_sl": sl,
            "scalping_tp": tp,
            "suggested_lots": lots,
            "trailing_start": tp * 0.25,
            "probabilities": probabilities,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        print(f"✅ Prédiction générée: {response}")
        
        # Notification Telegram si action de trading
        if action != 0 and confidence > 0.6:
            send_telegram_alert(f"🎯 Signal {symbol}: {'BUY' if action == 1 else 'SELL'} | Confiance: {confidence:.2f} | Lots: {lots}")
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erreur dans scalping_predict: {str(e)}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return jsonify({"error": str(e)}), 500

def simulate_prediction(symbol, features):
    """Simule une prédiction AI (à remplacer par vos vrais modèles)"""
    # Pour l'instant, simulation aléatoire
    np.random.seed(int(datetime.utcnow().timestamp()) % 1000)
    
    probabilities = np.random.dirichlet(np.ones(3), size=1)[0]
    action = np.argmax(probabilities) - 1  # -1, 0, 1
    confidence = np.max(probabilities)
    
    prob_dict = {
        "SELL": float(probabilities[0]),
        "HOLD": float(probabilities[1]),
        "BUY": float(probabilities[2])
    }
    
    return action, confidence, prob_dict

def calculate_scalping_parameters(symbol, confidence):
    """Calcule SL/TP dynamiques pour le scalping"""
    config = SCALPING_CONFIG[symbol]
    
    # Ajustement basé sur la confiance
    confidence_factor = 0.7 + (confidence * 0.3)  # 0.7 à 1.0
    
    sl = config["base_sl"] / confidence_factor
    tp = config["base_tp"] * confidence_factor
    
    # Calcul des lots (capital fixe pour démo)
    account_balance = 10000
    risk_amount = account_balance * (config["risk"] / 100)
    lots = min(risk_amount / (sl * 10000), 0.3)  # Max 0.3 lot pour scalping
    
    return round(sl, 5), round(tp, 5), round(lots, 2)

def send_telegram_alert(message):
    """Envoie une alerte Telegram"""
    try:
        if CHAT_ID:
            bot.send_message(chat_id=CHAT_ID, text=message)
            print(f"📱 Alert sent: {message}")
    except Exception as e:
        print(f"❌ Telegram error: {e}")

# Commandes Telegram
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /start pour Telegram"""
    global CHAT_ID
    CHAT_ID = update.effective_chat.id
    await update.message.reply_text(
        "🤖 Quantum AI Trader Activé!\n\n"
        "Statut: ✅ EN LIGNE\n"
        "Mode: SCALPING CLOUD\n"
        "Symboles: XAUUSD, BTCUSD, ETHUSD, USDJPY, EURUSD, SP500\n\n"
        "Vous recevrez les signaux de trading ici."
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Commande /status pour Telegram"""
    status_msg = (
        "📊 STATUT DU SYSTÈME\n"
        f"🕒 Heure: {datetime.utcnow().strftime('%H:%M:%S')}\n"
        f"📈 Symboles actifs: {len(SYMBOLS)}\n"
        f"🔗 Serveur: ✅ ONLINE\n"
        f"💳 Compte: DÉMO\n"
        f"📱 Dernier trade: {trade_history[-1] if trade_history else 'Aucun'}"
    )
    await update.message.reply_text(status_msg)

def run_telegram_bot():
    """Lance le bot Telegram en arrière-plan avec sa propre event loop"""
    def start_bot():
        try:
            # Créer une nouvelle event loop pour ce thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            application = Application.builder().token(TELEGRAM_TOKEN).build()
            
            # Command handlers
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("status", status_command))
            
            print("✅ Bot Telegram configuré et en attente de /start")
            
            # Lance le bot
            application.run_polling()
            
        except Exception as e:
            print(f"❌ Erreur bot Telegram: {e}")
            import traceback
            print(f"🔍 Stack trace: {traceback.format_exc()}")

    # Lancer dans un thread séparé
    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    return bot_thread

# Middleware de logging pour debug
@app.before_request
def log_request_info():
    if request.method == 'POST' and request.path == '/scalping-predict':
        print(f"📍 POST /scalping-predict - Content-Type: {request.content_type}")

@app.after_request
def log_response_info(response):
    if request.path == '/scalping-predict':
        print(f"📍 Response /scalping-predict: {response.status_code}")
    return response

# Démarrage
if __name__ == '__main__':
    print("🚀 Démarrage du Quantum AI Cloud Server...")
    
    # Lance le bot Telegram dans un thread séparé
    telegram_thread = run_telegram_bot()
    
    print("✅ Serveur AI prêt")
    print("✅ Bot Telegram en attente de /start")
    print("🌐 URL: https://votre-app.onrender.com")
    
    # Démarrer Flask
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
