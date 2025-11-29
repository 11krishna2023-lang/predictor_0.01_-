from datetime import datetime
import pytz, json, joblib
from alpaca_trade_api import REST
from data_sources import load_price_history
from prediction_core import predict_next_move   # <— NEW clean import

cfg = json.load(open("config.json"))


# ----------------- MARKET HOUR CHECK -----------------
def within_market_hours():
    if not cfg["market_hours"]["trade_only_during_market"]:
        return True
    
    tz = pytz.timezone(cfg["market_hours"]["timezone"])
    now = datetime.now(tz).time()
    open_t = datetime.strptime(cfg["market_hours"]["open"], "%H:%M").time()
    close_t = datetime.strptime(cfg["market_hours"]["close"], "%H:%M").time()

    return open_t <= now <= close_t


# ------------------- TRADE EXECUTION -------------------
def place_trade(action, ticker):
    if not cfg["alpaca"]["enabled"]:
        print("⚠ Trading disabled — prediction only mode")
        return
    
    api = REST(cfg["alpaca"]["api_key"], cfg["alpaca"]["secret_key"], cfg["alpaca"]["base_url"])
    qty = cfg["dollar_per_trade"]

    print(f"\n🔁 Executing: {action} — {ticker} — ${qty}")
    
    api.submit_order(
        ticker, 
        notional=qty, 
        side=action.lower(), 
        type="market", 
        time_in_force="gtc"
    )


# ------------------- MAIN RUNTIME -------------------
print("\n📈 Market Prediction —", datetime.now(), "\n")
signal = predict_next_move()                       # <— Tiny model script now plugged in
print("Predicted move →", signal)

if not within_market_hours():
    print("\n⛔ Market Closed → HOLD\n")
    exit()

if signal == "BUY":
    place_trade("BUY", cfg["tickers"][0])
elif signal == "SELL":
    place_trade("SELL", cfg["tickers"][0])
else:
    print("\n🟡 Recommendation → HOLD\n")
