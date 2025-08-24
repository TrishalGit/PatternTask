import ccxt
import pandas as pd

exchange = ccxt.binance({
    'options': {'defaultType': 'future'}  # ensure futures data
})

symbol = "BTC/USDT"
since = exchange.parse8601("2024-01-01T00:00:00Z")

# Download in batches
ohlcv = []
while since < exchange.parse8601("2025-01-01T00:00:00Z"):
    candles = exchange.fetch_ohlcv(symbol, timeframe="1m", since=since, limit=1000)
    if not candles:
        break
    ohlcv += candles
    since = candles[-1][0] + 60000  # move to next minute
    print(since)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.to_csv("BTCUSDT_futures_1m.csv", index=False)
print(df.head())
