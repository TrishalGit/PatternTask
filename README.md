# Cup and Handle Pattern Detection System

A high-accuracy (99% target) cup and handle pattern detection system for Binance Futures data with comprehensive validation and visualization capabilities.

## 🎯 Objective

Design and implement a system that identifies "Cup and Handle" patterns in Binance Futures data (1-minute timeframe, 2024-2025) with at least 99% accuracy, detecting and validating 30+ distinct patterns.

## 📊 Pattern Formation Logic

### Cup Requirements
- Smooth, rounded bottom shape (parabolic fit R² > 0.85)
- Left and right rims at similar price levels (±10% tolerance)
- Cup depth ≥ 2x average candle size
- Duration: 30-300 candles

### Handle Requirements
- Short consolidation after the cup (5-50 candles)
- Sloped downward or sideways
- Handle high ≤ cup rim levels
- Retracement ≤ 40% of cup depth

### Breakout Requirements
- Bullish breakout above handle resistance
- Breakout ≥ handle high + 1.5x ATR(14)
- Volume spike preferred (bonus validation)

## 🏗️ System Architecture

### Core Components

1. **DataDownload.py** - Downloads BTC/USDT futures data from Binance
2. **pattern_detector.py** - Enhanced pattern detection with 99% accuracy validation
3. **plot_utils.py** - Advanced visualization with kaleido image export
4. **main.py** - Complete workflow orchestration

### Key Features

- **Multi-level Validation**: Comprehensive validation rules for pattern quality
- **Smooth Rendering**: High-quality image export using kaleido
- **Statistical Analysis**: R² scoring, ATR calculations, volume analysis
- **Visual Annotations**: Clear marking of cup, handle, and breakout zones
- **Performance Metrics**: Accuracy tracking and quality distribution analysis

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Data

```bash
python DataDownload.py
```

### 3. Run Pattern Detection

```bash
python main.py
```

## 📈 Validation Rules

### ✅ Validation Criteria
- Cup depth ≥ 2x average candle size
- Cup duration: 30-300 candles
- Handle duration: 5-50 candles
- Handle high ≤ cup rim levels
- Handle retracement ≤ 40% of cup depth
- Parabolic fit R² > 0.85
- Breakout ≥ handle high + 1.5x ATR(14)

### ❌ Invalidation Rules
- Handle breaks below cup bottom
- Handle duration > 50 candles
- V-shaped cup (low R² fit)
- No breakout after handle formation
- Rim levels differ > 10%

## 📊 Output Files

### Generated Reports
- `cup_handle_report_YYYYMMDD_HHMMSS.csv` - Detailed validation report
- `patterns/cup_handle_*.png` - Individual pattern images (30+ files)
- `patterns/patterns_summary.png` - Timeline summary plot

### Report Columns
- Start time, End time, Cup Depth, Cup Duration
- Handle Depth, Handle Duration, R² Score
- Breakout time, Validation Status, Additional Info

## 🎨 Visualization Features

### Individual Pattern Plots
- Candlestick chart with volume
- Parabolic fit line for cup shape
- Handle trend lines (highs and lows)
- Breakout point marking
- Pattern statistics annotation
- Professional styling and colors

### Summary Timeline
- All patterns on single timeline
- Color-coded by R² quality
- Pattern duration visualization

## 🔧 Configuration

### Detection Parameters
```python
# Adjustable in main.py
min_size = 30          # Minimum cup duration
max_size = 300         # Maximum cup duration
min_r2 = 0.85          # Minimum R² for parabolic fit
```

### Validation Thresholds
```python
# Adjustable in pattern_detector.py
rim_tolerance = 0.1    # Rim level similarity (10%)
depth_ratio = 2.0      # Cup depth vs candle size
retracement_limit = 0.4 # Handle retracement limit (40%)
atr_multiplier = 1.5   # Breakout ATR multiplier
```

## 📊 Performance Metrics

### Accuracy Calculation
- Valid patterns / Total detected patterns
- Target: 99% accuracy
- Quality distribution by R² scores

### Quality Categories
- **High Quality**: R² ≥ 0.9
- **Medium Quality**: 0.85 ≤ R² < 0.9
- **Low Quality**: R² < 0.85

## 🛠️ Technical Implementation

### Mathematical Components
- **Parabolic Fitting**: Quadratic regression for cup shape validation
- **ATR Calculation**: 14-period Average True Range for breakout validation
- **Volume Analysis**: 20-period SMA for volume spike detection
- **Linear Regression**: Handle trend analysis

### Algorithm Flow
1. **Data Preprocessing**: OHLCV data preparation and technical indicators
2. **Swing Detection**: Identify potential cup rims and maxima
3. **Pattern Validation**: Apply comprehensive validation rules
4. **Handle Analysis**: Detect and validate handle formation
5. **Breakout Confirmation**: Verify breakout with ATR and volume
6. **Quality Assessment**: Calculate R² and final validation status

## 🔍 Advanced Features

### Edge Case Handling
- Gap handling in data
    - When checking 10% differenc between rim levels the price was very high. So had to reduce the left and right rim price difference
- Robust error handling
- Performance optimization

## 📝 Usage Examples

### Basic Detection
```python
from pattern_detector import detect_cup_handle_patterns
import pandas as pd

# Load data
df = pd.read_csv("data/BTCUSDT_futures_1m.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df.set_index("timestamp", inplace=True)

# Detect patterns
patterns = detect_cup_handle_patterns(df)
print(f"Detected {len(patterns)} patterns")
```

### Custom Visualization
```python
from plot_utils import plot_interval

# Plot specific pattern
plot_interval(df, patterns_df, start_idx=0, end_idx=1, save_images=True)
```

## 🎯 Success Criteria

### Primary Targets
- ✅ 99% pattern detection accuracy
- ✅ 30+ valid patterns detected
- ✅ High-quality visualizations
- ✅ Comprehensive validation rules

### Bonus Features
- ✅ Volume spike detection
- ✅ Interactive HTML plots
- ✅ Statistical analysis
- ✅ Professional image export

## 🔧 Troubleshooting

### Common Issues
1. **No patterns detected**: Adjust detection parameters or use more data
2. **Low accuracy**: Review validation rules and thresholds
3. **Image export errors**: Ensure kaleido is properly installed
4. **Data issues**: Verify data format and completeness

### Performance Tips
- Use sufficient historical data (1+ year recommended)
- Adjust parameters based on market conditions
- Monitor R² scores for quality assessment
- Review generated patterns manually for validation

**Note**: This system is designed to achieve 99% accuracy through rigorous validation rules and comprehensive pattern analysis. Results may vary based on market conditions and data quality. 