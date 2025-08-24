import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit
import bisect
import talib

# ---------------- Rim level check functions ---------------- #
def rim_level_check1(left_rim, right_rim):
    """Check rim diff using (rim_left - rim_right)/max"""
    if abs(left_rim - right_rim) / max(left_rim, right_rim) > 0.1:
        return False
    return True

def rim_level_check2(left_rim, right_rim):
    """Check rim diff using abs(rim_left - rim_right)/min"""
    if abs(left_rim - right_rim) / min(left_rim, right_rim) > 0.1:
        return False
    return True

# ---------------- Right Rim limit functions ---------------- #
def right_rim_limit1(left_rim):
    return [0.9*left_rim, (left_rim/0.9)]

def right_rim_limit2(left_rim):
    return [(left_rim/1.1), 1.1*left_rim]

# ---------------- Cup Handle detection function ---------------- #
def detect_cup_handle_patterns(df, depth_type="close", rim_level="high", right_rim_limit=right_rim_limit1,
                        min_size=30, max_size=300, min_r2=0.85):
    """
    Identify valid cup-handle formations in OHLCV data
    Args:
        df: DataFrame with ['timestamp','open','high','low','close','volume']
        depth_type: "close" or "low" -> defines cup depth
        rim_level: "close" or "high" -> defines rim levels
        rim_level_check: function to compare rim similarity
        min_size, max_size: allowed cup width in candles
        min_r2: minimum R^2 for parabola fit
    Returns:
        list of [left_index, right_index, r2_value]
    """
    results = []
    past = 0
    df["high-low"] = df["high"] - df["low"]
    df["sum(high-low)"] = df["high-low"].cumsum()
    prev_close = df["close"].shift(1)
    df["TR"] = df[["high", "low"]].apply(lambda row: max(
        row["high"] - row["low"],
        abs(row["high"] - prev_close[row.name]),
        abs(row["low"] - prev_close[row.name])
    ), axis = 1)
    # print(len(df))
    # df["ATR14"] = df["TR"].rolling(14).mean().shift(1)
    df['ATR14'] = talib.ATR(
        df['high'], df['low'], df['close'],
        timeperiod=14
    )
    df["High14"] = df["high"].rolling(14).max().shift(1)
    df["High300"] = df["high"].rolling(301).max()
    df["Low300"] = df["low"].rolling(301).min()
    df["Volume20"] = talib.SMA(df['volume'], timeperiod=20).shift(1)

    # df["Volume20"] = df["volume"].rolling(20).mean().shift(1)

    # print(df.head(20))
    df["Breakout"] = df["high"] - 1.5 * df["ATR14"] - df["High14"]
    # df["Breakout"] = df["Breakout"].shift(1)
    # print(df.head(20))

    breakout_indices = df[(df["Breakout"] > 0.0) & (df["volume"] > 2 * df["Volume20"])].index
    # return results

    # Updating left maximas and right maximas for faster Cup filtering
    vals = df[rim_level].values
    # window = 5
    # n = len(vals)

    # # Condition: i > i-1
    # cond1 = vals[1:-window] > vals[:-window-1]

    # # Condition: i >= max(i+1 ... i+5)
    # next_max = np.maximum.reduce([vals[j:n-window+j] for j in range(1, window+1)])
    # cond2 = vals[1:-window] >= next_max

    # # Indices satisfying both
    # idx = np.where(cond1 & cond2)[0] + 1
    idx = np.where((vals[1:-1] > vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1
    filtered_idx = []
    for i in idx:
        end1 = min(i+4, len(df) - 1)       # clamp to array end
        future1 = df[i+1:end1]   # lookahead 4
        # print(future1)
        if np.any(future1[rim_level] > df.at[i, rim_level]):
            continue
        end2 = min(i+50, len(df) - 1)       # clamp to array end
        future2 = df[i+5:end2]   # lookahead 5–50
        if np.any(future2[rim_level] > df.at[i, rim_level]):
            filtered_idx.append(i)

    right_maximas = df.iloc[filtered_idx]
    # right_maximas = df[df["right_maxima"] == 1]


    # print(right_maximas)

    # df["left_maxima"] = 0
    left_indices = np.where((vals[1:-1] > vals[2:]))[0] + 1
    left_indices_filtered = []
    for li in left_indices:
        pos = bisect.bisect_right(breakout_indices, li)

        if pos < len(breakout_indices) and breakout_indices[pos] <= li + 350:
            left_indices_filtered.append(li)
    
    print("Left indices: ", len(left_indices_filtered))
    # df.loc[idx, "left_maxima"] = 1

    for left in left_indices_filtered:
        if left <= past:
            continue
        # print("x")
        # break
        # Filter out Right rims < 10% difference of Left rim
        # right_rim_df = df.iloc[left + min_size: min(left+max_size, len(df) - 1)]
        right_rim_data = right_maximas[(right_maximas.index >= left + min_size) & (right_maximas.index <= min(left+max_size, len(df) - 1))]
        # print(right_rim_data.index)
        # print(left + min_size)
        # if left > 200:
        #     break
        gap =  2*df["Low300"].iloc[min(left+300, len(df)-1)] - (df["High300"].iloc[min(left+300, len(df) - 1)])
        # gap = 0
        rim_limits = right_rim_limit(df[rim_level].iloc[left] - gap)
        right_rim_filtered = right_rim_data[(right_rim_data[rim_level] - gap > rim_limits[0]) & (right_rim_data[rim_level] - gap < rim_limits[1])]
        
        left_cumsum = 0 if left == 0 else df["sum(high-low)"].iloc[left-1]

        for right in right_rim_filtered.index:
            
            # Subset data for the candidate cup
            cup_df = df.iloc[left:right+1]

            # Average candle size only within this cup (high - low)
            # avg_candle_size = (cup_df["high-low"]).mean()
            avg_candle_size = (cup_df["sum(high-low)"].iloc[-1] - left_cumsum)/(right - left + 1)

            
            # Cup data for parabola fit
            y = cup_df[depth_type].values
            x = np.arange(len(cup_df))
            # x = np.arange(len(y))
            
            # Find cup depth vs rim
            # print(cup_df)
            # print(left)
            left_rim = cup_df[rim_level].loc[left]
            # print(left, left_rim)
            right_rim = cup_df[rim_level].loc[right]
            bottom = y.min()
            tip = cup_df[rim_level].values.max()
            if tip > max(left_rim, right_rim):
                continue 

            depth = min(left_rim, right_rim) - bottom
            # print(left_rim, right_rim, bottom, depth, avg_candle_size)

            # Check for 40% depth crossing of handle
            end = min(right + 50, len(df) - 1)
            handle_vals = df[right+1:end+1]
            # if right == 925:
            #     print(handle_vals)

            # print(len(handle_vals), right)
            # print(handle_vals)
            # return results

            breakout_idx = None
            min_handle_val = right_rim
            handle_high = 0
            min_handle_idx = None
            for j in handle_vals.index:
                # if j == right:
                #     continue
                handle_high = max(handle_high, handle_vals.loc[j, "high"])
                # if j >= 943 and j <= 947:
                #     print(handle_vals)
                #     print("Breakout",j, right, right_rim, handle_high, handle_vals.loc[j+1, "high"], 1.5*handle_vals.loc[j+1, "ATR14"])
                #     print((right_rim + left_rim)/2.0, handle_vals.loc[j+1, rim_level])
                #     print(j <= right + 49)
                #     print( handle_high + 1.5*handle_vals.loc[j+1, "ATR14"] < handle_vals.loc[j+1, "high"])
                #     print((right_rim + left_rim)/2.0 < handle_vals.loc[j+1, rim_level])
                #     print(((j <= right + 49) and ((right_rim + left_rim)/2.0 < handle_vals.loc[j+1, rim_level]) and (handle_high + 1.5*handle_vals.loc[j+1, "ATR14"] < handle_vals.loc[j+1, "high"])))
                #     print(j, right + 49)
                    # return results
                if (int(j) <= int(right) + 49) and ((right_rim + left_rim)/2.0 < handle_vals.loc[j+1, rim_level]) and (handle_high + 1.5*handle_vals.loc[j+1, "ATR14"] < handle_vals.loc[j+1, "high"]):
                        # if int(j) >= 942 and int(j) <= 962:
                        #     print("Yo")
                        # if j >= 943 and j <= 947:
                        #     print("hello")
                        pos = bisect.bisect_right(breakout_indices, j)
                        # if j >= 943 and j <= 947:
                        #     print(breakout_indices[pos], j)
                        if pos < len(breakout_indices):
                            if breakout_indices[pos] == j+1 and breakout_indices[pos] >= right + 5 and breakout_indices[pos] <= right + 50:
                                breakout_idx = breakout_indices[pos]
                                break
                            elif breakout_indices[pos] > right + 50:
                                continue
                        else:
                            break
                else:
                    if min_handle_val > handle_vals.loc[j, "low"]:
                        min_handle_idx = j
                        min_handle_val = handle_vals.loc[j, "low"]
                    # min_handle_val = min(min_handle_val, handle_vals.loc[j, "low"])
    
            if breakout_idx is not None:
                # handle_len = breakout_idx - i
                # handle_fil = handle_vals[right+1:breakout_idx]
                if (right_rim - min_handle_val) > 0.4*depth:
                    # print(right_rim, min_handle_val, depth, right, left_rim, right_rim, bottom)
                    # if breakout_idx != 944:
                    continue
            else:
                continue
            
            # Condition 1: depth >= 2 * avg_candle_size
            if depth < 2 * avg_candle_size:
                continue
            
            # # Condition 2: rim similarity check
            # if not rim_level_check(left_rim, right_rim):
            #     continue
            
            # Condition 3: Parabola fit (use closing price)
            coefs = polyfit(x, y, 2)  # quadratic fit
            c, b, a = coefs
            # Upward curve
            if a < 0:
                continue
            # print("Parabola coeff", a, b, c)
            y_fit = np.polyval(coefs[::-1], x)
            r2 = r2_score(y, y_fit)
            # print("R2", r2, cup_df["timestamp"].loc[left], cup_df["timestamp"].loc[right], cup_df["timestamp"].iloc[0], cup_df["timestamp"].iloc[-1])
            if r2 <= min_r2:
                continue

            # ✅ If all conditions met, store the cup
            print(left, right)
            valid_cup = []
            valid_cup.append(cup_df["timestamp"].loc[left])  # Left rim / start time
            valid_cup.append(cup_df["timestamp"].loc[right])  # Right rim / end time
            valid_cup.append(depth)   # Cup depth
            valid_cup.append(right - left + 1)  # Cup duration
            valid_cup.append(right_rim - min_handle_val)   # Handle depth
            valid_cup.append(min_handle_idx - right)    # handle duration
            valid_cup.append(r2)    # R2
            valid_cup.append(df["timestamp"].iloc[breakout_idx])   # Breakout timestamp

            # valid_cup = [df["timestamp"].iloc[breakout_idx], r2]
            print(valid_cup)
            # print(cup_df)
            results.append(valid_cup)
            # return results
            past = right
            break

    return results
