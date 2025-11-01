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
CHAT_ID = None

# Variables globales
bot = telegram.Bot(token=TELEGRAM_TOKEN)
trade_history = []

# Middleware de logging
@app.before_request
def log_request_info():
    if request.path == '/scalping-predict':
        print(f"📍 {request.method} {request.path}")
        print(f"📨 Content-Type: {request.headers.get('Content-Type')}")
        print(f"📨 User-Agent: {request.headers.get('User-Agent')}")

@app.after_request
def log_response_info(response):
    if request.path == '/scalping-predict':
        print(f"📍 Response: {response.status_code}")
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

@app.route('/scalping-predict', methods=['POST', 'GET'])
def scalping_predict():
    """Endpoint de prédiction avec support complet pour MT5"""
    try:
        # 🔥 DEBUG COMPLET - Voir tout ce qui arrive
        print("=" * 50)
        print("🎯 NOUVELLE REQUÊTE SCALPING-PREDICT")
        print("=" * 50)
        
        # Log des headers
        print("📨 HEADERS:")
        for key, value in request.headers:
            print(f"   {key}: {value}")
        
        # Log de la méthode
        print(f"📨 METHOD: {request.method}")
        
        # Essayer tous les formats possibles
        data = {}
        content_type = request.headers.get('Content-Type', '').lower()
        raw_data = request.get_data(as_text=True)
        
        print(f"📨 Content-Type: {content_type}")
        print(f"📨 Raw data length: {len(raw_data)}")
        print(f"📨 Raw data (first 500 chars): {raw_data[:500]}")
        
        # 1. Essayer JSON direct
        if not data:
            try:
                json_data = request.get_json()
                if json_data:
                    data = json_data
                    print("✅ Données extraites via request.get_json()")
                    print(f"   Données: {data}")
            except Exception as e:
                print(f"❌ request.get_json() failed: {e}")
        
        # 2. Essayer form-data (common pour MT5)
        if not data and request.form:
            print("📨 Form data détecté:")
            print(f"   Form keys: {list(request.form.keys())}")
            data = dict(request.form)
            print(f"   Form data: {data}")
            
            # Essayer d'extraire JSON des champs form
            for key in ['data', 'json', 'payload', 'input']:
                if key in data:
                    try:
                        parsed_data = json.loads(data[key])
                        data = parsed_data
                        print(f"✅ JSON extrait du champ form '{key}'")
                        break
                    except Exception as e:
                        print(f"❌ Échec parsing champ '{key}': {e}")
                        continue
        
        # 3. Essayer de parser raw data comme JSON
        if not data and raw_data and raw_data.strip():
            try:
                parsed_data = json.loads(raw_data)
                data = parsed_data
                print("✅ JSON parsé depuis raw data")
            except Exception as e:
                print(f"❌ Échec parsing raw data: {e}")
                
                # Essayer de parser comme string simple (MT5 peut envoyer juste le symbol)
                if raw_data.strip() in SYMBOLS:
                    data = {"symbol": raw_data.strip(), "features": []}
                    print(f"✅ Symbol détecté directement: {raw_data}")
        
        # 4. Si GET request, utiliser query parameters
        if not data and request.method == 'GET':
            data = dict(request.args)
            print("✅ Données extraites depuis query parameters GET")
            print(f"   Query data: {data}")
        
        # Si toujours aucune donnée, créer des données par défaut pour debug
        if not data:
            print("⚠️  Aucune donnée extraite, utilisation des valeurs par défaut")
            data = {
                "symbol": "BTCUSD", 
                "features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
            }
        
        print(f"🎯 Données finales: {data}")
        
        # Extraction et validation des données
        symbol = data.get("symbol", "").upper()
        if not symbol:
            symbol = data.get("symbole", "").upper()
        
        features = data.get("features", [])
        if not features:
            features = data.get("feature", [])
        if not features:
            features = data.get("data", [])
        
        print(f"🎯 Symbol final: {symbol}")
        print(f"🎯 Features count: {len(features)}")
        
        # Validation du symbol
        if not symbol:
            return jsonify({
                "error": "Symbole manquant",
                "help": "Envoyez 'symbol' dans vos données. Ex: {'symbol':'BTCUSD','features':[...]}",
                "received_data": data
            }), 400
        
        if symbol not in SYMBOLS:
            return jsonify({
                "error": f"Symbole {symbol} non supporté",
                "supported_symbols": SYMBOLS,
                "received_data": data
            }), 400
        
        # Gestion des features
        if not features:
            print("⚠️  Aucune feature reçue, utilisation de features par défaut")
            features = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0,2.1,2.2,2.3,2.4,2.5]
        
        # Convertir features en liste si nécessaire
        if isinstance(features, str):
            try:
                features = json.loads(features)
                print("✅ Features converties depuis string JSON")
            except:
                try:
                    features = [float(x.strip()) for x in features.split(',')]
                    print("✅ Features converties depuis CSV string")
                except:
                    features = [0.0] * 25
                    print("⚠️  Features invalides, utilisation de valeurs par défaut")
        
        # S'assurer d'avoir exactement 25 features
        if len(features) < 25:
            features = features + [0.0] * (25 - len(features))
            print(f"⚠️  Features padding: {len(features)} features")
        elif len(features) > 25:
            features = features[:25]
            print(f"⚠️  Features tronquées: {len(features)} features")
        
        # Simulation de prédiction
        action, confidence, probabilities = simulate_prediction(symbol, features)
        
        # Calcul des paramètres de scalping
        sl, tp, lots = calculate_scalping_parameters(symbol, confidence)
        
        response = {
            "symbol": symbol,
            "action": action,
            "confidence": round(confidence, 4),
            "scalping_sl": sl,
            "scalping_tp": tp,
            "suggested_lots": lots,
            "trailing_start": round(tp * 0.25, 6),
            "probabilities": probabilities,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success",
            "features_received": len(features),
            "request_debug": {
                "content_type": content_type,
                "method": request.method,
                "data_sample": str(data)[:200]
            }
        }
        
        print(f"✅ RÉPONSE GÉNÉRÉE:")
        print(f"   Symbol: {response['symbol']}")
        print(f"   Action: {response['action']} ({'BUY' if action == 1 else 'SELL' if action == -1 else 'HOLD'})")
        print(f"   Confidence: {response['confidence']}")
        print(f"   SL/TP: {response['scalping_sl']}/{response['scalping_tp']}")
        
        # Sauvegarder dans l'historique
        trade_history.append({
            "symbol": symbol,
            "action": action,
            "confidence": confidence,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Garder seulement les 50 derniers trades
        if len(trade_history) > 50:
            trade_history.pop(0)
        
        # Notification Telegram si action de trading
        if action != 0 and confidence > 0.6:
            send_telegram_alert(f"🎯 Signal {symbol}: {'BUY' if action == 1 else 'SELL'} | Confiance: {confidence:.2f} | Lots: {lots}")
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = f"Erreur serveur: {str(e)}"
        print(f"❌ ERREUR CRITIQUE: {error_msg}")
        import traceback
        print(f"🔍 Stack trace: {traceback.format_exc()}")
        
        return jsonify({
            "error": error_msg,
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "help": "Contactez le support avec ces logs"
        }), 500

def simulate_prediction(symbol, features):
    """Simule une prédiction AI"""
    np.random.seed(int(datetime.utcnow().timestamp()) % 1000)
    
    probabilities = np.random.dirichlet(np.ones(3), size=1)[0]
    action = np.argmax(probabilities) - 1
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
    
    confidence_factor = 0.7 + (confidence * 0.3)
    
    sl = config["base_sl"] / confidence_factor
    tp = config["base_tp"] * confidence_factor
    
    account_balance = 10000
    risk_amount = account_balance * (config["risk"] / 100)
    lots = min(risk_amount / (sl * 10000), 0.3)
    
    return round(sl, 5), round(tp, 5), round(lots, 2)

def send_telegram_alert(message):
    """Envoie une alerte Telegram"""
    try:
        if CHAT_ID:
            bot.send_message(chat_id=CHAT_ID, text=message)
            print(f"📱 Alert Telegram: {message}")
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
    """Lance le bot Telegram en arrière-plan"""
    def start_bot():
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            application = Application.builder().token(TELEGRAM_TOKEN).build()
            
            application.add_handler(CommandHandler("start", start_command))
            application.add_handler(CommandHandler("status", status_command))
            
            print("✅ Bot Telegram configuré et en attente de /start")
            
            application.run_polling()
            
        except Exception as e:
            print(f"❌ Erreur bot Telegram: {e}")

    bot_thread = threading.Thread(target=start_bot, daemon=True)
    bot_thread.start()
    return bot_thread

# Démarrage
if __name__ == '__main__':
    print("🚀 Démarrage du Quantum AI Cloud Server...")
    
    telegram_thread = run_telegram_bot()
    
    print("✅ Serveur AI prêt")
    print("✅ Bot Telegram en attente de /start") 
    print("🌐 URL: https://quantumaitrader.onrender.com")
    
    port = int(os.environ.get('PORT', 5000))
    print(f"🔌 Port: {port}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
