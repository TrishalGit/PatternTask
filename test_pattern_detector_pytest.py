import pytest
import pandas as pd
import numpy as np
from pattern_detector import detect_cup_handle_patterns
from constants import OPEN, HIGH, LOW, CLOSE, VOLUME, TIMESTAMP

# def df_valid():
#     return pd.DataFrame({
#         TIMESTAMP: pd.date_range(start="2024-01-01", periods=100, freq="min"),
#         OPEN: np.linspace(100, 110, 100),
#         HIGH: np.linspace(101, 111, 100),
#         LOW: np.linspace(99, 109, 100),
#         CLOSE: np.linspace(100, 110, 100),
#         VOLUME: np.random.randint(100, 200, 100)
#     })

# def df_no_pattern():
#     return pd.DataFrame({
#         TIMESTAMP: pd.date_range(start="2024-01-01", periods=50, freq="min"),
#         OPEN: np.ones(50) * 100,
#         HIGH: np.ones(50) * 100,
#         LOW: np.ones(50) * 100,
#         CLOSE: np.ones(50) * 100,
#         VOLUME: np.ones(50) * 100
#     })

# def df_short():
#     return pd.DataFrame({
#         TIMESTAMP: pd.date_range(start="2024-01-01", periods=10, freq="min"),
#         OPEN: np.linspace(100, 110, 10),
#         HIGH: np.linspace(101, 111, 10),
#         LOW: np.linspace(99, 109, 10),
#         CLOSE: np.linspace(100, 110, 10),
#         VOLUME: np.random.randint(100, 200, 10)
#     })

def test_detect_valid_patterns():
    df_valid1 = pd.read_csv("test_data/valid_pattern_0.csv")
    patterns = detect_cup_handle_patterns(df_valid1)
    print(len(patterns))

    # assert isinstance(patterns, list)

    print(patterns)
    
    df_invalid1 = pd.read_csv("test_data/valid_pattern_1.csv")
    patterns = detect_cup_handle_patterns(df_invalid1)
    # assert isinstance(patterns, list)
    print(patterns)

    # patterns = detect_cup_handle_patterns(df_valid())
    # assert isinstance(patterns, list)

    # patterns = detect_cup_handle_patterns(df_valid())
    # assert isinstance(patterns, list)
    # Should not raise error, may or may not find patterns

# def test_no_patterns_found():
#     patterns = detect_cup_handle_patterns(df_no_pattern())
#     assert len(patterns) == 0

# def test_short_data():
#     patterns = detect_cup_handle_patterns(df_short())
#     assert len(patterns) == 0

def test_invalid_parameters():
    with pytest.raises(Exception):
        detect_cup_handle_patterns(None)

# def test_missing_columns():
#     df = df_valid().drop(columns=[HIGH])
#     with pytest.raises(Exception):
#         detect_cup_handle_patterns(df)
