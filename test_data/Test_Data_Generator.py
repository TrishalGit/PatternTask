import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:], 
                    np.abs(high[1:] - close[:-1]), 
                    np.abs(low[1:] - close[:-1]))
    atr = np.zeros_like(close)
    if len(tr) >= period:
        atr[period:] = np.convolve(tr, np.ones(period)/period, mode='valid')
    return atr

def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0

def generate_cup_and_handle_pattern(valid=True, n_patterns=5, seed=42):
    np.random.seed(seed)
    all_patterns = []

    for _ in range(n_patterns):
        # Cup & handle sizes
        cup_size = np.random.randint(30, 301)
        handle_size = np.random.randint(5, 51)
        avg_candle_size = np.random.uniform(1, 5)

        # Rim reference price
        rim_price = 100
        left_rim_val = rim_price + np.random.uniform(-0.1*rim_price, 0.1*rim_price)
        right_rim_val = rim_price + np.random.uniform(-0.1*rim_price, 0.1*rim_price)

        # Cup depth
        min_rim_price = min(left_rim_val, right_rim_val)
        
        if valid:
            # For valid patterns, ensure depth is at least min_rim_price/4
            min_required_depth = min_rim_price / 4
            cup_depth = np.random.uniform(min_required_depth, min_required_depth * 2)
        else:
            # For invalid patterns, use insufficient depth (less than min_rim_price/4)
            max_invalid_depth = min_rim_price / 4
            cup_depth = np.random.uniform(avg_candle_size, max_invalid_depth * 0.8)
        
        bottom_price = min_rim_price - cup_depth
        handle_depth = np.random.uniform(0.1*cup_depth, 0.8*cup_depth)

        if valid:
            handle_depth = min(handle_depth, 0.4*cup_depth)
        else:
            handle_depth = max(handle_depth, 0.5*cup_depth)  # too deep

        handle_high = min(left_rim_val, right_rim_val) - handle_depth
        if not valid:
            handle_high = max(left_rim_val, right_rim_val) + handle_depth  # invalid handle

        # Cup curve (U-shape)
        x_cup = np.arange(cup_size)
        y_cup = bottom_price + (cup_depth * (1 - ((x_cup - cup_size/2)/(cup_size/2))**2))

        # R² check for valid cups
        if valid:
            fit_coef = np.polyfit(x_cup, y_cup, 2)
            y_fit = np.polyval(fit_coef, x_cup)
            if r_squared(y_cup, y_fit) < 0.85:
                y_cup += np.random.uniform(0, cup_depth*0.1, size=cup_size)

        # Handle curve
        x_handle = np.arange(handle_size)
        handle_curve = handle_high - 0.05*x_handle
        if not valid:
            handle_curve += np.random.uniform(0.5, 5, size=handle_size)

        # Merge prices
        prices = np.concatenate([y_cup, handle_curve])
        n_rows = len(prices)

        # Timestamps
        start_time = datetime(2024,10,3,6,29,0)
        timestamps = [start_time + timedelta(minutes=i) for i in range(n_rows)]

        # OHLCV
        ohlcv = []
        for ts, p in zip(timestamps, prices):
            o = p + np.random.uniform(-0.2, 0.2)
            h = max(p + np.random.uniform(0.1, 0.5), p)
            l = min(p - np.random.uniform(0.1, 0.5), p)
            c = p
            v = np.random.randint(100, 500)
            ohlcv.append([ts, o,h,l,c,v])

        df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])

        # ---- Rim ordering and handle breakout filter ----
        right_pos = len(y_cup) - 1  # right rim at end of cup
        if right_pos < 30:
            continue

        # left rim must be in (right-300, right-30)
        left_start = max(0, right_pos - 300)
        left_end = right_pos - 30
        left_window = df.loc[left_start:left_end]
        if len(left_window) == 0:
            continue
        left_pos = left_window["high"].idxmax()

        # enforce left < right
        if left_pos >= right_pos:
            if valid:
                continue

        left_rim_high = df.loc[left_pos, "high"]
        right_rim_high = df.loc[right_pos, "high"]

        # --- Updated Depth condition: depth >= min(left_rim, right_rim)/4 ---
        bottom_idx = df.loc[left_pos:right_pos, "low"].idxmin()
        bottom_price_actual = df.loc[bottom_idx, "low"]
        actual_depth = min(left_rim_high, right_rim_high) - bottom_price_actual
        required_depth = min(left_rim_high, right_rim_high) / 4
        
        if valid and actual_depth < required_depth:
            continue
        elif not valid and actual_depth >= required_depth:
            continue

        # --- ensure no point between left & right has high > max(left_rim, right_rim) ---
        max_mid_high = df.loc[left_pos:right_pos, "high"].max()
        if valid and max_mid_high >= max(left_rim_high, right_rim_high):
            continue

        # handle breakout filter
        if right_pos + 50 < len(df):
            cond1 = df.loc[right_pos:right_pos+4, "high"].max() <= df.loc[right_pos, "high"]
            cond2 = (df.loc[right_pos+5:right_pos+50, "high"] > df.loc[right_pos, "high"]).any()
        else:
            cond1, cond2 = True, True

        if valid and not (cond1 and cond2):
            continue

        # ---- Calculate ATR[14] ----
        atr14 = calculate_atr(df['high'].values, df['low'].values, df['close'].values, 14)
        atr14 = np.append(np.zeros(len(df)-len(atr14)), atr14)

        # Breakout candle
        breakout_price = handle_high + 1.5 * (atr14[-1] if atr14[-1]>0 else avg_candle_size)
        avg_vol20 = df['volume'][-20:].mean() if len(df) >= 20 else df['volume'].mean()
        breakout_vol = 3 * avg_vol20
        breakout_time = df['timestamp'].iloc[-1] + timedelta(minutes=1)

        breakout_candle = [breakout_time,
                           breakout_price,
                           breakout_price+0.2,
                           breakout_price-0.2,
                           breakout_price,
                           int(breakout_vol)]

        df.loc[len(df)] = breakout_candle
        all_patterns.append(df)

    return all_patterns

# Generate datasets
valid_patterns = generate_cup_and_handle_pattern(valid=True, n_patterns=3)
invalid_patterns = generate_cup_and_handle_pattern(valid=False, n_patterns=3)

# Save CSVs
for i, df in enumerate(valid_patterns):
    df.to_csv(f"test_data/valid_pattern_{i}.csv", index=False)

for i, df in enumerate(invalid_patterns):
    df.to_csv(f"test_data/invalid_pattern_{i}.csv", index=False)

print("CSV files generated with updated depth condition: depth >= min(left_rim, right_rim)/4")