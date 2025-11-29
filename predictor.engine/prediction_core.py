import json, joblib, numpy as np, pandas as pd
from data_sources import load_price_history

cfg = json.load(open("config.json"))

def build_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA3"]   = df["Close"].rolling(3).mean()
    df["MA5"]   = df["Close"].rolling(5).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()
    df["Vol"]   = df["Close"].rolling(20).std()
    df["RSI"]   = 100 - (100 / (1 + df["Return"].rolling(14).mean()))
    df.dropna(inplace=True)

    return df[[
        "Close","Return","MA3","MA5","EMA20","Vol","RSI"
    ]]  # consistent feature schema


def prepare_data():
    print("Loading market data…")
    df = load_price_history(cfg["tickers"], period=f"{cfg['lookback_days']}d")
    df = df.groupby(level=0, axis=1).last()   # collapse multiindex
    features = build_features(df.copy())
    return features


def load_latest_model():
    model = joblib.load(cfg["model_path"])
    scaler = joblib.load("model_store/latest_scaler.pkl")
    return model, scaler


def predict_next_move():
    model, scaler = load_latest_model()
    feat = prepare_data().tail(1)
    X = scaler.transform(feat)

    price     = float(feat["Close"])
    prediction = model.predict(X)[0]
    confidence = model.predict_proba(X)[0][prediction]

    direction = "UP" if prediction == 1 else "DOWN"
    future_est = price * (1 + (0.015 if direction=="UP" else -0.015))  # short-term target

    return {
        "direction"   : direction,
        "current"     : price,
        "predicted"   : future_est,
        "confidence"  : round(confidence, 4),
        "difference%" : round((future_est-price)/price*100, 3)
    }

