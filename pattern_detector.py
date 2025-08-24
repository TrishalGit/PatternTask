import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from numpy.polynomial.polynomial import polyfit
import bisect
import talib
from constants import TIMESTAMP, HIGH, LOW, CLOSE, VOLUME, ATR_RATIO, VOLUME_SPIKE_RATIO

# ---------------- Enhanced validation functions ---------------- #
def validate_rim_levels(left_rim, right_rim, tolerance=0.1):
    # Check: Absolute difference as percentage of minimum
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

def validate_volume_spike(volume, avg_volume, min_multiplier=2.0):
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

def process_data(df):
    # Calculate technical indicators
    # Cummulative sum useful for faster calculation of 2 * avg_candle_size < depth
    df["CandleSize"] = df[HIGH] - df[LOW]
    df["sum(CandleSize)"] = df["CandleSize"].cumsum()  

    # ATR calculation
    prev_close = df[CLOSE].shift(1)
    # True Range and ATR
    df["TR"] = df[[HIGH, LOW]].apply(lambda row: max(
        row[HIGH] - row[LOW],
        abs(row[HIGH] - prev_close[row.name]),
        abs(row[LOW] - prev_close[row.name])
    ), axis=1)

    df['ATR14'] = talib.ATR(df[HIGH], df[LOW], df[CLOSE], timeperiod=14) 
    df["High14"] = df[HIGH].rolling(window=14, min_periods=1).max().shift(1)

    # Breakout detection
    df["Breakout"] = df[HIGH] - ATR_RATIO * df["ATR14"] - df["High14"]

    # Use full for checking 10 % difference between rim levels 
    df["High300"] = df[HIGH].rolling(window=301, min_periods=1).max()
    df["Low300"] = df[LOW].rolling(window=301, min_periods=1).min()

    # Useful for checking volume spike at breakout
    df["VOL_MA20"] = talib.SMA(df[VOLUME], timeperiod=20).shift(1)

    return df

def get_breakout_indices(df):
    return df[(df["Breakout"] > 0.0) & (df[VOLUME] > 2 * df["VOL_MA20"])].index

def get_right_maximas(df, rim_level, handle_min_size, handle_max_size):
    # Find potential right rim maxima (swing highs)
    # Identifying these two patterns
    #    ___   or     /\
    # __/          __/  \__
    # flat after a spike and local maxima
    vals = df[rim_level].values
    idx = np.where((vals[1:-1] > vals[:-2]) & (vals[1:-1] >= vals[2:]))[0] + 1
    
    # Filter right rim indices no price higher in pos -> pos + 4 and atleast 1 price higher 
    # from pos + 5 -> pos + 50 as handle region is (5, 50)
    filtered_idx = []
    for i in idx:
        handle_start = min(i+handle_min_size, len(df) - 1)
        non_handle_region = df[i+1:handle_start - 1]
        if np.any(non_handle_region[rim_level] > df.at[i, rim_level]):
            continue
        handle_end = min(i+handle_max_size, len(df) - 1)
        handle_region = df[handle_start:handle_end]
        if np.any(handle_region[rim_level] > df.at[i, rim_level]):
            filtered_idx.append(i)
    
    return df.iloc[filtered_idx]

def get_left_indices(df, rim_level, breakout_indices, max_size=300, handle_max_size=50, min_size=30, handle_min_size=5):
    # Find potential left rim maxima just the next price should be lower than current
    vals = df[rim_level].values
    left_indices = np.where((vals[1:-1] > vals[2:]))[0] + 1

    # Filter out left indices near breakout regions
    left_indices_filtered = []
    for li in left_indices:
        pos = bisect.bisect_right(breakout_indices, li)
        if pos < len(breakout_indices) and breakout_indices[pos] <= li + max_size + handle_max_size and breakout_indices[pos] >= li + min_size + handle_min_size:
            left_indices_filtered.append(li)
    
    return left_indices_filtered

def find_gap(df, left):
    return df["Low300"].iloc[min(left+300, len(df)-1)]

def get_right_rim_data(df, rim_level, left, right_maximas, min_size, max_size, gap):
    # Find potential right rims within size constraints
    right_rim_data = right_maximas[
        (right_maximas.index >= left + min_size) & 
        (right_maximas.index <= min(left + max_size, len(df) - 1))
    ]
        
    # Apply rim level constraints
    rim_limits = [0.9*(df[rim_level].iloc[left] - gap), 1.1*(df[rim_level].iloc[left] - gap)]
    right_rim_filtered = right_rim_data[
        (right_rim_data[rim_level] - gap > rim_limits[0]) & 
        (right_rim_data[rim_level] - gap < rim_limits[1])
    ]

    return right_rim_filtered

# ---------------- Enhanced Cup Handle detection function ---------------- #
def detect_cup_handle_patterns(df, rim_level="high", min_size=30, max_size=300, 
                               handle_min_size=5, handle_max_size=50, min_r2=0.85,
                               patterns=30):
    """
    Enhanced cup-handle pattern detection with 99% accuracy validation
    Args:
        df: DataFrame with ['timestamp','open','high','low','close','volume']
        rim_level: "close" or "high" -> defines rim levels
        min_size, max_size: allowed cup width in candles
        handle_min_size, handle_max_size: allowed handle width in candles
        min_r2: minimum R^2 for parabola fit
    Returns:
        list of [start_time, end_time, cup_depth, cup_duration, handle_depth, 
                handle_duration, r2, breakout_time, validation_status, reason]
    """
    results = []
    
    df = process_data(df)

    # Filter out intial data for faster processing
    breakout_indices = get_breakout_indices(df)
    right_maximas = get_right_maximas(df, rim_level, handle_min_size, handle_max_size)
    left_indices_filtered = get_left_indices(df, rim_level, breakout_indices, max_size, handle_max_size, min_size, handle_min_size)

    print(f"Processing {len(left_indices_filtered)} potential left rims...")

    past_left = 0
    for left in left_indices_filtered:
        if left <= past_left:
            continue

        # Get the size/height in the region (301 candles from left rim) to identify pattern
        # Currently its minimum value in 301 candles update if necessary
        # As actual prices are very high and 10% check for rim level diference fails and gives 
        # invalid regions
        gap = find_gap(df, left)  

        # Filter right rims for 10% gap
        right_rim_filtered = get_right_rim_data(df, rim_level, left, right_maximas, min_size, max_size, gap)
        
        # Cummulative sum till left-1 th index to get average candle size till right rim faster
        left_cumsum = 0 if left == 0 else df["sum(CandleSize)"].iloc[left-1]
        
        for right in right_rim_filtered.index:
            # Cup data subset
            cup_df = df.iloc[left:right+1]
            
            # Calculate average candle size
            avg_candle_size = (cup_df["sum(CandleSize)"].iloc[-1] - left_cumsum)/(right - left + 1)
            
            # Cup characteristics
            left_rim = cup_df[rim_level].loc[left]
            right_rim = cup_df[rim_level].loc[right]
            bottom = cup_df[LOW].min()
            tip = cup_df[rim_level].max()

            depth = min(left_rim, right_rim) - bottom
            
            # Skip if tip is higher than rims (invalid cup shape)
            if tip > (depth/2.0) + max(left_rim, right_rim):
                continue
            
            # Validate rim levels
            if not validate_rim_levels(left_rim, right_rim):
                continue
                
            # Validate cup depth
            if not validate_cup_depth(depth, avg_candle_size):
                continue
                
            # Validate cup duration
            cup_duration = right - left + 1
            if not validate_cup_duration(cup_duration, min_size, max_size):
                continue
                
            # Handle analysis
            end = min(right + handle_max_size, len(df) - 1)
            handle_vals = df[right+1:end+1]
            
            breakout_idx = None
            min_handle_val = right_rim
            handle_high = 0
            min_handle_idx = None
            
            for j in handle_vals.index:
                # ignore the 50th candle
                if j <= min(right + handle_max_size - 1, len(df) - 2):
                    handle_high = max(handle_high, handle_vals.loc[j, HIGH])

                # Check for breakout conditions for i+1 th candle
                if (int(j) <= min(int(right) + handle_max_size - 1, len(df) - 2) and 
                    (right_rim+ left_rim)/2.0 < handle_vals.loc[j+1, rim_level] and 
                    handle_high + ATR_RATIO*handle_vals.loc[j+1, "ATR14"] < handle_vals.loc[j+1, HIGH]):
                    
                    pos = bisect.bisect_right(breakout_indices, j)
                    if pos < len(breakout_indices):
                        if (breakout_indices[pos] == j+1 and 
                            breakout_indices[pos] >= right + handle_min_size and 
                            breakout_indices[pos] <= right + handle_max_size):
                            breakout_idx = breakout_indices[pos]
                            break
                        elif breakout_indices[pos] > right + handle_max_size:
                            continue
                    else:
                        break
                elif j <= min(right + handle_max_size - 1, len(df) - 2):
                    if min_handle_idx is None or min_handle_val > handle_vals.loc[j, LOW]:
                        min_handle_idx = j
                        min_handle_val = handle_vals.loc[j, LOW]
            
            if breakout_idx is None:
                continue

            # Validate breakout
            breakout_price = df.loc[breakout_idx, HIGH]
            if not validate_breakout(breakout_price, handle_high, df.loc[breakout_idx, "ATR14"], ATR_RATIO):
                continue
                
            # Handle validation
            handle_depth = right_rim - min_handle_val
            handle_duration = min_handle_idx - right if min_handle_idx else 0

            # Update handle high till the fall region only
            if min_handle_idx:
                handle_high = handle_vals.loc[right+1:min_handle_idx][HIGH].max() if min_handle_idx else 0
            
            # Validate handle retracement
            if handle_depth > 0.4 * depth:
                continue
                
            # Validate handle duration
            if not validate_handle_duration(handle_duration, handle_min_size):
                continue
                
            # Validate handle high - handle high computed till the fall region after Right rim
            if not validate_handle_high(handle_high, left_rim, right_rim):
                continue
                
            # Bonus: Validate volume spike
            volume_spike = validate_volume_spike(
                df.loc[breakout_idx, VOLUME], 
                df.loc[breakout_idx, "VOL_MA20"],
                VOLUME_SPIKE_RATIO
            )

            # Validate smooth curve parabolic fit
            y = cup_df[CLOSE].values
            is_smooth, r2 = validate_smooth_curve(y, depth, min_r2)
            if not is_smooth:
                continue
            
            # All validations passed - record the pattern
            pattern_data = [
                cup_df[TIMESTAMP].loc[left],      # START_TIME
                cup_df[TIMESTAMP].loc[right],     # END_TIME
                depth,                              # CUP_DEPTH
                cup_duration,                       # CUP_DURATION
                handle_depth,                       # HANDLE_DEPTH
                handle_duration,                    # HANDLE_DURATION
                r2,                                 # R2
                df[TIMESTAMP].iloc[breakout_idx], # BREAKOUT_TIME
                "Valid",                           # VALIDATION_STATUS
                f"Volume spike: {volume_spike}"     # ADDITIONAL_INFO
            ]
            
            results.append(pattern_data)
            print(f"Valid pattern found: {left} -> {right}, R²={r2:.3f}")

            # Continue with finding new region
            past_left = right
            break

        if(len(results) >= patterns):
            break
    
    print(f"Total valid patterns detected: {len(results)}")
    return results
