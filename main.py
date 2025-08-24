import pandas as pd
import numpy as np
from pattern_detector import detect_cup_handle_patterns
from plot_utils import plot_interval, create_pattern_summary_plot
import csv
from constants import CUP_DEPTH, CUP_DURATION, HANDLE_DEPTH, HANDLE_DURATION, R2, BREAKOUT_TIME, VALIDATION_STATUS, ADDITIONAL_INFO, START_TIME, END_TIME
import os
from datetime import datetime

def main():
    """
    Main execution function for Cup and Handle pattern detection
    """
    print("=== Cup and Handle Pattern Detection System ===")
    print("Target: 99% Accuracy with 30+ Valid Patterns")
    print("=" * 50)
    
    # Load full dataset (2024-2025)
    print("Loading BTC/USDT futures data...")
    try:
        df = pd.read_csv("data/BTCUSDT_futures_1m.csv")
        print(f"Loaded {len(df)} data points")
    except FileNotFoundError:
        print("Data file not found. Please run DataDownload.py first.")
        return
    
    # Ensure we have enough data
    if len(df) < 100000:
        print("Warning: Limited data available. Consider downloading more data.")
    
    # # Convert timestamp and set index
    # df["timestamp"] = pd.to_datetime(df["timestamp"])
    # df.set_index("timestamp", inplace=True)
    
    # Filter to 2024-2025 data
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    df = df[(df["timestamp"] >= start_date) & (df["timestamp"] < end_date)]
    print(f"Filtered to {len(df)} data points from {start_date} to {end_date}")
    
    report_filename = f"report.csv"

    choice = -1
    while choice not in [1, 2, 3]:
        print("Enter task:")
        print("1. Collect patterns")
        print("2. Plot patterns")
        print("3. Do both")

        choice = int(input("Enter your choice (1-3): "))

        if choice not in [1, 2, 3]:
            raise ValueError("Invalid choice. Please enter 1, 2, or 3.")

    accuracy = 0
    results_df = []

    if choice in [1, 3]:
        # Run enhanced pattern detection
        print("\nRunning enhanced cup and handle pattern detection...")
        print("Applying 99% accuracy validation rules...")
        
        cups = detect_cup_handle_patterns(
            df=df,
            rim_level="high",
            min_size=30,
            max_size=300,
            handle_min_size=5,
            handle_max_size=50,
            min_r2=0.85,
            patterns=30
        )
        
        if not cups:
            print("No valid patterns detected. Adjusting parameters...")
            # Try with slightly relaxed parameters
            cups = detect_cup_handle_patterns(
                df=df,
                rim_level="high",
                min_size=25,
                max_size=350,
                handle_min_size=5,
                handle_max_size=50,
                min_r2=0.80,
                patterns=30
            )
        
        print(f"\nDetected {len(cups)} valid cup and handle patterns")
        
        if len(cups) < 30:
            print(f"Warning: Only {len(cups)} patterns detected. Target is 30+ patterns.")
            print("Consider adjusting detection parameters or using more data.")
        
        # Create results DataFrame
        columns = [
            START_TIME, END_TIME, CUP_DEPTH, CUP_DURATION,
            HANDLE_DEPTH, HANDLE_DURATION, R2, BREAKOUT_TIME,
            VALIDATION_STATUS, ADDITIONAL_INFO
        ]
        
        results_df = pd.DataFrame(cups, columns=columns)
        
        # Calculate accuracy metrics
        valid_patterns = len(results_df[results_df[VALIDATION_STATUS] == "Valid"])
        total_patterns = len(results_df)
        accuracy = (valid_patterns / total_patterns * 100) if total_patterns > 0 else 0
        
        print(f"\n=== Pattern Detection Results ===")
        print(f"Total Patterns Detected: {total_patterns}")
        print(f"Valid Patterns: {valid_patterns}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Quality analysis
        high_quality = len(results_df[results_df[R2] >= 0.9])
        medium_quality = len(results_df[(results_df[R2] >= 0.85) & (results_df[R2] < 0.9)])
        low_quality = len(results_df[results_df[R2] < 0.85])
        
        print(f"\n=== Quality Distribution ===")
        print(f"High Quality (R² ≥ 0.9): {high_quality} patterns")
        print(f"Medium Quality (0.85 ≤ R² < 0.9): {medium_quality} patterns")
        print(f"Low Quality (R² < 0.85): {low_quality} patterns")
        
        # Save detailed report
        print("\nSaving detailed validation report...")
        
        with open(report_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(cups)
        
        print(f"Report saved: {report_filename}")
        
        # Create summary statistics
        if len(results_df) > 0:
            print(f"\n=== Pattern Statistics ===")
            print(f"Average Cup Depth: {results_df[CUP_DEPTH].mean():.2f}")
            print(f"Average Cup Duration: {results_df[CUP_DURATION].mean():.1f} candles")
            print(f"Average Handle Depth: {results_df[HANDLE_DEPTH].mean():.2f}")
            print(f"Average Handle Duration: {results_df[HANDLE_DURATION].mean():.1f} candles")
            print(f"Average R² Score: {results_df[R2].mean():.3f}")
            print(f"Best R² Score: {results_df[R2].max():.3f}")
            print(f"Worst R² Score: {results_df[R2].min():.3f}")
    
    if choice in [2, 3]:
        # Generate visualizations
        results_df = pd.read_csv("report.csv")
        if len(results_df) > 0:
            print("\nGenerating pattern visualizations...")
            
            # Create patterns directory if it doesn't exist
            patterns_dir = "patterns"
            if not os.path.exists(patterns_dir):
                os.makedirs(patterns_dir)
            
            # Plot individual patterns (limit to first 30 for performance)
            max_plots = min(30, len(results_df))
            print(f"Generating {max_plots} individual pattern plots...")
            
            plot_interval(
                ohlcv=df,
                intervals=results_df,
                start_idx=0,
                end_idx=max_plots,
                save_images=True,
                output_dir=patterns_dir
            )
            
            # Create summary plot
            print("Creating summary timeline plot...")
            create_pattern_summary_plot(
                intervals_df=results_df,
                save_image=True,
                output_dir=patterns_dir
            )
            
            print(f"All visualizations saved to '{patterns_dir}' directory")
        

        # Calculate accuracy metrics
        valid_patterns = len(results_df[results_df[VALIDATION_STATUS] == "Valid"])
        total_patterns = len(results_df)
        accuracy = (valid_patterns / total_patterns * 100) if total_patterns > 0 else 0

    # Final assessment
    print(f"\n=== Final Assessment ===")
    if accuracy >= 99:
        print("✅ TARGET ACHIEVED: 99%+ accuracy reached!")
    elif accuracy >= 95:
        print("🟡 GOOD: 95%+ accuracy achieved")
    elif accuracy >= 90:
        print("🟠 ACCEPTABLE: 90%+ accuracy achieved")
    else:
        print("🔴 NEEDS IMPROVEMENT: Below 90% accuracy")
    
    if len(results_df) >= 30:
        print("✅ TARGET ACHIEVED: 30+ patterns detected!")
    else:
        print(f"🟡 PARTIAL: {len(results_df)} patterns detected (target: 30+)")
    
    print(f"\n=== Files Generated ===")
    print(f"1. Detailed Report: {report_filename}")
    print(f"2. Pattern Images: {patterns_dir}/cup_handle_*.png")
    print(f"3. Summary Plot: {patterns_dir}/patterns_summary.png")
    
    print(f"\n=== Next Steps ===")
    if accuracy < 99 or len(results_df) < 30:
        print("1. Review detection parameters")
        print("2. Consider using more historical data")
        print("3. Fine-tune validation rules")
        print("4. Implement additional filtering criteria")
    else:
        print("1. Review generated patterns for quality")
        print("2. Analyze pattern performance")
        print("3. Consider implementing ML enhancement")
    
    return results_df

if __name__ == "__main__":
    results = main()
