import os, json, joblib, numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from prediction_core import prepare_data, cfg

MODEL_DIR = "model_store"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_model():
    print("\n=== 🔥 TRAINING NEW MODEL 🔥 ===")

    df = prepare_data()
    df["Future"] = df["Close"].shift(-cfg["prediction_horizon"])
    df.dropna(inplace=True)

    y = (df["Future"] > df["Close"]).astype(int)  # 1=up, 0=down
    X = df.drop(["Future"],axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(hidden_layer_sizes=(64,64,32), max_iter=2000)
    model.fit(X_scaled, y)

    # version control
    ver = len([f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]) + 1
    model_file  = f"{MODEL_DIR}/model_v{ver}.pkl"
    scaler_file = f"{MODEL_DIR}/scaler_v{ver}.pkl"

    joblib.dump(model, model_file)
    joblib.dump(scaler, scaler_file)

    # update latest pointers
    joblib.dump(model,  f"{MODEL_DIR}/latest_model.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/latest_scaler.pkl")

    print(f"\n✔ Model Saved → {model_file}")
    print(f"✔ Latest Updated → model_store/latest_model.pkl\n")
    return model_file


if __name__ == "__main__":
    train_model()

