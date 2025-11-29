from datetime import datetime
import pytz, json, joblib
from data_sources import load_price_history
from alpaca_trade_api import REST

cfg = json.load(open("config.json"))

def within_market_hours():
    if not cfg["market_hours"]["trade_only_during_market"]:
        return True

    tz = pytz.timezone(cfg["market_hours"]["timezone"])
    now = datetime.now(tz).time()
    open_t = datetime.strptime(cfg["market_hours"]["open"], "%H:%M").time()
    close_t = datetime.strptime(cfg["market_hours"]["close"], "%H:%M").time()

    return open_t <= now <= close_t


def place_trade(action,ticker):
    if not cfg["alpaca"]["enabled"]:
        print("Alpaca not enabled")
        return

    api = REST(cfg["alpaca"]["api_key"], cfg["alpaca"]["secret_key"], cfg["alpaca"]["base_url"])
    qty = cfg["dollar_per_trade"]

    if action == "BUY":
        api.submit_order(ticker, notional=qty, side="buy", type="market", time_in_force="gtc")
    elif action == "SELL":
        api.submit_order(ticker, notional=qty, side="sell", type="market", time_in_force="gtc")



# ---- RUN PREDICTION SAME AS BEFORE ----
model = joblib.load(cfg["model_path"])
scaler = joblib.load("model_store/scaler.pkl")
df = load_price_history(cfg["tickers"], period="30d")

df["EMA20"] = df["Close"].ewm(span=20).mean()
df["RSI"] = 100-(100/(1+df["Close"].pct_change().rolling(14).mean()))
df.dropna(inplace=True)

X = df[["Close","EMA20","RSI"]].tail(1)
pred = model.predict(scaler.transform(X))[0]
curr = float(df["Close"].iloc[-1])

if not within_market_hours():
    print("Market closed — HOLD")
    exit()

if pred > curr*1.01:
    place_trade("BUY", df.columns[0])
elif pred < curr*0.99:
    place_trade("SELL", df.columns[0])
else:
    print("HOLD")

