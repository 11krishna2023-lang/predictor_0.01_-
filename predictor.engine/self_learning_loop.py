import time, json, joblib, os
from datetime import datetime, timedelta
from prediction_core import predict_next_move, cfg
from train_self_improving import train_model
from data_sources import load_price_history
import pandas as pd

LOG_FILE = "logs/self_learning_log.csv"
os.makedirs("logs", exist_ok=True)

def record_result(row: dict):
    df = pd.DataFrame([row])
    if not os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, index=False)
    else:
        df.to_csv(LOG_FILE, index=False, mode="a", header=False)


def self_learning_runner():
    print("\n🔥 Autonomous Self-Learning Engine Started 🔥\n")
    last_train = datetime.now() - timedelta(days=cfg["train_interval_days"])

    while True:

        # RUN A NEW PREDICTION
        pred = predict_next_move()
        print("\n📈 Live Prediction:", pred)

        # WAIT UNTIL FUTURE OUTCOME EXISTS
        wait_hrs = 9  # editable ← how long to wait before verifying results
        print(f"⏳ Waiting {wait_hrs} hours to validate prediction...\n")
        time.sleep(wait_hrs * 3600)

        # CHECK THE ACTUAL MARKET MOVE
        data = load_price_history(cfg["tickers"][0], period="3d")  # validate using first ticker
        actual_price = float(data["Close"].iloc[-1])
        was_up = actual_price > pred["current"]
        correct = 1 if (was_up and pred["direction"]=="UP") or (not was_up and pred["direction"]=="DOWN") else 0

        # SAVE RESULT
        record_result({
            "time"       : datetime.now(),
            "predicted"  : pred["direction"],
            "confidence" : pred["confidence"],
            "future_est" : pred["predicted"],
            "actual"     : actual_price,
            "correct"    : correct
        })

        print(f"✔ Prediction accuracy: {correct}\n")

        # TRAIN IF NEEDED
        if datetime.now() - last_train > timedelta(days=cfg["train_interval_days"]):
            print("\n🧠 Training new model automatically...")
            train_model()
            last_train = datetime.now()

        # LOOP ∞
        print("\n🔁 Next learning cycle in 1 hour...\n")
        time.sleep(3600)

