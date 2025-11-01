import flask
from flask import Flask, request, jsonify
import numpy as np
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
import json

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

# Middleware de logging
@app.before_request
def log_request_info():
    if request.path == '/scalping-predict':
        print(f"📍 {request.method} {request.path} - Content-Type: {request.headers.get('Content-Type')}")

@app.after_request
def log_response_info(response):
    if request.path == '/scalping-predict':
        print(f"📍 Response /scalping-predict: {response.status_code}")
    return response

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
    """Endpoint de prédiction pour le scalping avec support multi-format"""
    try:
        # 🔥 CORRECTION : Gérer les différents Content-Type
        content_type = request.headers.get('Content-Type', '').lower()
        print(f"📨 Content-Type reçu: {content_type}")
        
        # Extraire les données selon le format
        if 'application/x-www-form-urlencoded' in content_type:
            # MetaTrader envoie en form-data
            if request.form:
                # Essayer de trouver les données JSON dans les form fields
                data_str = None
                for key in ['data', 'json', 'payload']:
                    if key in request.form:
                        data_str = request.form[key]
                        break
                
                # Si pas trouvé, prendre le premier champ
                if not data_str and request.form:
                    data_str = list(request.form.keys())[0]
                
                if data_str:
                    try:
                        data = json.loads(data_str)
                        print(f"📦 Données form-data converties: {data}")
                    except json.JSONDecodeError:
                        # Si c'est déjà un dict, utiliser directement
                        data = dict(request.form)
                        print(f"📦 Données form-data brutes: {data}")
                else:
                    data = {}
            else:
                data = {}
                
        elif 'application/json' in content_type:
            # Déjà en JSON
            data = request.get_json()
            print(f"📦 Données JSON directes: {data}")
            
        else:
            # Essayer de parser comme JSON de toute façon
            try:
                data = request.get_json(force=True)
                print(f"📦 Données forcées en JSON: {data}")
            except:
                # Dernier recours : prendre les raw data
                raw_data = request.get_data(as_text=True)
                if raw_data:
                    try:
                        data = json.loads(raw_data)
                        print(f"📦 Données raw converties: {data}")
                    except:
                        data = {"raw": raw_data}
                        print(f"📦 Données raw brutes: {raw_data[:100]}...")
                else:
                    data = {}
        
        # Validation des données
        if not data:
            return jsonify({"error": "Aucune donnée reçue", "content_type": content_type}), 400
        
        # Extraire symbol et features
        symbol = data.get("symbol", "").upper() or data.get("symbole", "").upper()
        features = data.get("features", []) or data.get("feature", []) or data.get("data", [])
        
        print(f"🎯 Symbol: {symbol}, Features: {len(features)}")
        
        if not symbol:
            return jsonify({"error": "Symbole manquant"}), 400
        
        if symbol not in SYMBOLS:
            return jsonify({"error": f"Symbole {symbol} non supporté. Symboles valides: {SYMBOLS}"}), 400
        
        # Si features n'est pas une liste, essayer de convertir
        if not isinstance(features, list):
            if isinstance(features, str):
                try:
                    features = json.loads(features)
                except:
                    features = [float(x) for x in features.split(',')]
            elif isinstance(features, (int, float)):
                features = [features]
        
        if len(features) != 25:
            print(f"⚠️  Nombre de features: {len(features)} (attendu: 25)")
            # Padding ou troncature pour avoir 25 features
            if len(features) < 25:
                features = features + [0.0] * (25 - len(features))
            else:
                features = features[:25]
        
        # Simulation de prédiction
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
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }
        
        print(f"✅ Prédiction générée: {response}")
        
        # Sauvegarder dans l'historique
        trade_history.append({
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Garder seulement les 100 derniers trades
        if len(trade_history) > 100:
            trade_history.pop(0)
        
        # Notification Telegram si action de trading
        if action != 0 and confidence > 0.6:
            send_telegram_alert(f"🎯 Signal {symbol}: {'BUY' if action == 1 else 'SELL'} | Confiance: {confidence:.2f} | Lots: {lots}")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"❌ Erreur dans scalping_predict: {str(e)}"
        print(error_msg)
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        return jsonify({"error": error_msg, "status": "error"}), 500

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
    print(f"🔌 Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
