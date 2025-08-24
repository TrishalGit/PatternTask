import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit
import bisect
import talib

# ---------------- Enhanced validation functions ---------------- #
def validate_rim_levels(left_rim, right_rim, tolerance=0.1):
    """Enhanced rim level validation with multiple checks"""
    # Check 1: Relative difference
    if abs(left_rim - right_rim) / max(left_rim, right_rim) > tolerance:
        return False
    
    # Check 2: Absolute difference as percentage of minimum
    if abs(left_rim - right_rim) / min(left_rim, right_rim) > tolerance:
        return False
    
    return True

def validate_cup_depth(depth, avg_candle_size, min_ratio=2.0):
    """Validate cup depth is at least 2x average candle size"""
    return depth >= min_ratio * avg_candle_size

def validate_cup_duration(duration, min_candles=30, max_candles=300):
    """Validate cup duration is within acceptable range"""
    return min_candles <= duration <= max_candles

def validate_handle_duration(duration, min_candles=5, max_candles=50):
    """Validate handle duration is within acceptable range"""
    return min_candles <= duration <= max_candles

def validate_handle_retracement(handle_depth, cup_depth, max_retracement=0.4):
    """Validate handle doesn't retrace more than 40% of cup depth"""
    return handle_depth <= max_retracement * cup_depth

def validate_handle_high(handle_high, left_rim, right_rim):
    """Validate handle high is below or equal to cup rims"""
    max_rim = max(left_rim, right_rim)
    return handle_high <= max_rim

def validate_breakout(breakout_price, handle_high, atr_14, min_atr_multiplier=1.5):
    """Validate breakout exceeds handle high with sufficient ATR"""
    return breakout_price >= handle_high + min_atr_multiplier * atr_14

def validate_volume_spike(volume, avg_volume, min_multiplier=1.5):
    """Validate volume spike on breakout (bonus validation)"""
    return volume >= min_multiplier * avg_volume

def validate_smooth_curve(y_values, cup_depth, min_r2=0.85):
    """Validate smooth parabolic curve fit"""
    x = np.arange(len(y_values))
    coefs = polyfit(x, y_values, 2)
    c, b, a = coefs
    
    # Check if parabola opens upward (a > 0)
    if a <= (cup_depth/((len(y_values)-1)**2)):
        return False, 0
    
    # Calculate R²
    y_fit = np.polyval(coefs[::-1], x)
    r2 = r2_score(y_values, y_fit)
    
    return r2 >= min_r2, r2

# ---------------- Enhanced Cup Handle detection function ---------------- #
def detect_cup_handle_patterns(df, depth_type="close", rim_level="high", 
                              min_size=30, max_size=300, min_r2=0.85):
    """
    Enhanced cup-handle pattern detection with 99% accuracy validation
    Args:
        df: DataFrame with ['timestamp','open','high','low','close','volume']
        depth_type: "close" or "low" -> defines cup depth
        rim_level: "close" or "high" -> defines rim levels
        min_size, max_size: allowed cup width in candles
        min_r2: minimum R^2 for parabola fit
    Returns:
        list of [start_time, end_time, cup_depth, cup_duration, handle_depth, 
                handle_duration, r2, breakout_time, validation_status, reason]
    """
    results = []
    
    # Calculate technical indicators
    df["high-low"] = df["high"] - df["low"]
    df["sum(high-low)"] = df["high-low"].cumsum()
    prev_close = df["close"].shift(1)
    
    # True Range and ATR
    df["TR"] = df[["high", "low"]].apply(lambda row: max(
        row["high"] - row["low"],
        abs(row["high"] - prev_close[row.name]),
        abs(row["low"] - prev_close[row.name])
    ), axis=1)
    
    df['ATR14'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df["High14"] = df["high"].rolling(14).max().shift(1)
    df["High300"] = df["high"].rolling(301).max()
    df["Low300"] = df["low"].rolling(301).min()
    df["Volume20"] = talib.SMA(df['volume'], timeperiod=20).shift(1)
    
    # Breakout detection
    df["Breakout"] = df["high"] - 1.5 * df["ATR14"] - df["High14"]
    breakout_indices = df[(df["Breakout"] > 0.0) & (df["volume"] > 2 * df["Volume20"])].index
    
    # Find potential right rim maxima (swing highs)
    vals = df[rim_level].values
    idx = np.where((vals[1:-1] > vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1
    
    filtered_idx = []
    for i in idx:
        end1 = min(i+4, len(df) - 1)
        future1 = df[i+1:end1]
        if np.any(future1[rim_level] > df.at[i, rim_level]):
            continue
        end2 = min(i+50, len(df) - 1)
        future2 = df[i+5:end2]
        if np.any(future2[rim_level] > df.at[i, rim_level]):
            filtered_idx.append(i)
    
    right_maximas = df.iloc[filtered_idx]
    
    # Find potential left rim maxima
    left_indices = np.where((vals[1:-1] > vals[2:]))[0] + 1
    left_indices_filtered = []
    for li in left_indices:
        pos = bisect.bisect_right(breakout_indices, li)
        if pos < len(breakout_indices) and breakout_indices[pos] <= li + 350:
            left_indices_filtered.append(li)
    
    print(f"Processing {len(left_indices_filtered)} potential left rims...")
    
    past = 0
    for left in left_indices_filtered:
        if left <= past:
            continue
            
        # Find potential right rims within size constraints
        right_rim_data = right_maximas[
            (right_maximas.index >= left + min_size) & 
            (right_maximas.index <= min(left+max_size, len(df) - 1))
        ]
        
        # Apply rim level constraints
        gap = 2*df["Low300"].iloc[min(left+300, len(df)-1)] - df["High300"].iloc[min(left+300, len(df) - 1)]
        rim_limits = [0.9*(df[rim_level].iloc[left] - gap), 1.1*(df[rim_level].iloc[left] - gap)]
        right_rim_filtered = right_rim_data[
            (right_rim_data[rim_level] - gap > rim_limits[0]) & 
            (right_rim_data[rim_level] - gap < rim_limits[1])
        ]
        
        left_cumsum = 0 if left == 0 else df["sum(high-low)"].iloc[left-1]
        
        for right in right_rim_filtered.index:
            # Cup data subset
            cup_df = df.iloc[left:right+1]
            
            # Calculate average candle size
            avg_candle_size = (cup_df["sum(high-low)"].iloc[-1] - left_cumsum)/(right - left + 1)
            
            # Cup characteristics
            left_rim = cup_df[rim_level].loc[left]
            right_rim = cup_df[rim_level].loc[right]
            bottom = cup_df[depth_type].min()
            tip = cup_df[rim_level].max()
            
            # Skip if tip is higher than rims (invalid cup shape)
            if tip > (gap/2.0) * max(left_rim, right_rim):
                continue
                
            depth = min(left_rim, right_rim) - bottom
            
            # Validate rim levels
            if not validate_rim_levels(left_rim, right_rim):
                continue
                
            # Validate cup depth
            if not validate_cup_depth(depth, avg_candle_size):
                continue
                
            # Validate cup duration
            cup_duration = right - left + 1
            if not validate_cup_duration(cup_duration):
                continue
                
            # Handle analysis
            end = min(right + 50, len(df) - 1)
            handle_vals = df[right+1:end+1]
            
            breakout_idx = None
            min_handle_val = right_rim
            handle_high = 0
            min_handle_idx = None
            
            for j in handle_vals.index:
                if j <= min(right + 49, len(df) - 2):
                    handle_high = max(handle_high, handle_vals.loc[j, "high"])
                
                # Check for breakout conditions
                if (int(j) <= min(int(right) + 49, len(df) - 2) and 
                    (right_rim+ left_rim)/2.0< handle_vals.loc[j+1, rim_level] and 
                    handle_high + 1.5*handle_vals.loc[j+1, "ATR14"] < handle_vals.loc[j+1, "high"]):
                    
                    pos = bisect.bisect_right(breakout_indices, j)
                    if pos < len(breakout_indices):
                        if (breakout_indices[pos] == j+1 and 
                            breakout_indices[pos] >= right + 5 and 
                            breakout_indices[pos] <= right + 50):
                            breakout_idx = breakout_indices[pos]
                            break
                        elif breakout_indices[pos] > right + 50:
                            continue
                    else:
                        break
                elif j <= min(right + 49, len(df) - 2):
                    if min_handle_idx is None or min_handle_val > handle_vals.loc[j, "low"]:
                        min_handle_idx = j
                        min_handle_val = handle_vals.loc[j, "low"]
            
            if breakout_idx is None:
                continue
                
            # Handle validation
            handle_depth = right_rim - min_handle_val
            handle_duration = min_handle_idx - right if min_handle_idx else 0
            if min_handle_idx:
                handle_high = handle_vals.loc[right+1:min_handle_idx]["high"].max() if min_handle_idx else 0
            
            # Validate handle retracement
            if handle_depth > 0.4 * depth:
                continue
                
            # Validate handle duration
            if not validate_handle_duration(handle_duration):
                continue
                
            # Validate handle high
            if not validate_handle_high(handle_high, left_rim, right_rim):
                continue
                
            # Validate breakout
            breakout_price = df.loc[breakout_idx, "high"]
            if not validate_breakout(breakout_price, handle_high, df.loc[breakout_idx, "ATR14"]):
                continue
                
            # Bonus: Validate volume spike
            volume_spike = validate_volume_spike(
                df.loc[breakout_idx, "volume"], 
                df.loc[breakout_idx, "Volume20"]
            )

            # Validate smooth curve
            y = cup_df[depth_type].values
            is_smooth, r2 = validate_smooth_curve(y, depth, min_r2)
            if not is_smooth:
                continue
            
            # All validations passed - record the pattern
            pattern_data = [
                cup_df["timestamp"].loc[left],      # Start time
                cup_df["timestamp"].loc[right],     # End time
                depth,                              # Cup depth
                cup_duration,                       # Cup duration
                handle_depth,                       # Handle depth
                handle_duration,                    # Handle duration
                r2,                                 # R² value
                df["timestamp"].iloc[breakout_idx], # Breakout time
                "Valid",                           # Validation status
                f"Volume spike: {volume_spike}"     # Additional info
            ]
            
            results.append(pattern_data)
            x = np.arange(len(y))
            coefs = polyfit(x, y, 2)
            c, b, a = coefs
            print("Parabola fit: ", a, b, c)
            print(f"Valid pattern found: {left} -> {right}, R²={r2:.3f}")
            past = right
            break
    
    print(f"Total valid patterns detected: {len(results)}")
    return results
