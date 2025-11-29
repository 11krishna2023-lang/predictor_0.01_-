import os, pandas as pd, yfinance as yf, requests, json
cfg = json.load(open("config.json"))

def alpaca_get_bars(symbol, timeframe="1Day", limit=500):
    if not cfg["alpaca"]["enabled"]:
        return None

    url = f"{cfg['alpaca']['base_url']}/v2/stocks/{symbol}/bars"
    headers = {
        "APCA-API-KEY-ID": cfg["alpaca"]["api_key"],
        "APCA-API-SECRET-KEY": cfg["alpaca"]["secret_key"]
    }
    params = {"timeframe": timeframe, "limit": limit}

    r = requests.get(url, headers=headers, params=params)

    if r.status_code != 200:
        print(f"[WARN] Alpaca failed for {symbol}, using Yahoo")
        return None

    data = r.json().get("bars", [])
    if not data:
        return None

    df = pd.DataFrame(data)
    df["Date"] = pd.to_datetime(df["t"])
    df.set_index("Date", inplace=True)
    df.rename(columns={"o":"Open","h":"High","l":"Low","c":"Close","v":"Volume"}, inplace=True)
    return df[["Open","High","Low","Close","Volume"]]


def load_price_history(tickers, period="2y", interval="1d"):
    final_data = {}

    for t in tickers:
        alpaca_data = alpaca_get_bars(t)
        if alpaca_data is not None:
            final_data[t] = alpaca_data
            continue
        
        if cfg["yahoo_fallback"]:
            print(f"[YAHOO] Fallback for {t}")
            final_data[t] = yf.download(t, period=period, interval=interval, auto_adjust=True)
    
    return pd.concat(final_data, axis=1)

