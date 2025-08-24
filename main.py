import pandas as pd
from pattern_detector import detect_cup_handle_patterns
from plot_utils import plot_interval
import csv

# Example: load OHLCV CSV (replace with Binance data or your source)
df = pd.read_csv("BTCUSDT_futures_1m.csv")

# # Run detection
df = df.head(100000)
cups = detect_cup_handle_patterns(df)

header = ["Start time", "End time", "Cup Depth", "Cup Duration", "Handle Depth", "Handle Duration", "R2", "Breakout time"]

with open('ValidCups.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(cups)

# Print results
print("Detected cup formations:")
print(len(cups))

# valid_cups_df = pd.read_csv("ValidCups.csv")

# print(valid_cups_df.head())

# Load your intervals dataset
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)
intervals = pd.read_csv("ValidCups.csv", parse_dates=["Start time", "End time", "Breakout time"])
# --------------------------
# Example: Plot for the first interval
# --------------------------
# row = intervals.iloc[0]
plot_interval(df, intervals, 13, 14)
# plot_interval(df, intervals, 0, len(intervals) - 1)

# --------------------------
# Loop over all intervals if needed
# --------------------------
# for _, row in intervals.iterrows():
#     plot_interval(row["start_timestamp"], row["end_timestamp"])
# for c in cups:
#     print(c)
