import pandas as pd
import plotly.graph_objects as go
from numpy.polynomial.polynomial import polyfit
import numpy as np
from sklearn.metrics import r2_score
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os
from constants import CUP_DEPTH, CUP_DURATION, HANDLE_DEPTH, HANDLE_DURATION, R2, BREAKOUT_TIME, START_TIME, END_TIME

# --------------------------
# Enhanced plot function for each interval with kaleido export
# --------------------------
def plot_interval(ohlcv, intervals, start_idx, end_idx, save_images=True, output_dir="patterns"):
    """
    Enhanced plotting function with kaleido image export
    Args:
        ohlcv: OHLCV DataFrame with datetime index
        intervals: DataFrame with pattern intervals
        start_idx, end_idx: Range of patterns to plot
        save_images: Whether to save images using kaleido
        output_dir: Directory to save images
    """
    # Ensure output directory exists
    if save_images and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Ensure timestamp is datetime
    if not isinstance(ohlcv.index, pd.DatetimeIndex):
        ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])
        ohlcv.set_index("timestamp", inplace=True)
    
    for i in range(start_idx, end_idx):
        intervals[START_TIME] = pd.to_datetime(intervals[START_TIME])
        intervals[END_TIME] = pd.to_datetime(intervals[END_TIME])
        intervals[BREAKOUT_TIME] = pd.to_datetime(intervals[BREAKOUT_TIME])
        start_ts = intervals.loc[i, START_TIME]
        end_ts = intervals.loc[i, END_TIME]
        breakout_ts = intervals.loc[i, BREAKOUT_TIME]
        
        # Add buffer for better visualization
        plot_start = start_ts - pd.Timedelta(minutes=30)
        plot_end = breakout_ts + pd.Timedelta(minutes=30)
        
        # Slice OHLCV data
        data = ohlcv.loc[plot_start:plot_end]
        cup_df = ohlcv.loc[start_ts:end_ts]
        
        # Handle data (from end of cup to falling region before breakout)
        handle_df = ohlcv.loc[end_ts:breakout_ts]
        min_idx = handle_df['low'].idxmin()
        handle_df = handle_df.loc[:min_idx]
        
        # Calculate parabolic fit for cup
        y = cup_df["close"].values
        x = np.arange(len(cup_df))
        coefs = polyfit(x, y, 2)
        c, b, a = coefs
        y_fit = np.polyval(coefs[::-1], x)
        r2 = r2_score(y, y_fit)
        
        # Create enhanced figure
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
            increasing_fillcolor="#26A69A",
            decreasing_fillcolor="#EF5350",
            name="Price"
        ))
        
        # Add volume as bar chart
        fig.add_trace(go.Bar(
            x=data.index,
            y=data["volume"],
            name="Volume",
            marker_color="rgba(100, 149, 237, 0.3)",
            yaxis="y2"
        ))
        
        # Add parabolic fit line
        fig.add_trace(go.Scatter(
            x=cup_df.index,
            y=y_fit,
            mode="lines",
            name=f"Cup Fit (R²={r2:.3f})",
            line=dict(color="#FF6B6B", width=3, dash="dot"),
            showlegend=True
        ))
        
        # Add handle trend lines if handle exists
        if len(handle_df) > 1:
            # Linear regression on handle highs
            x_handle = np.arange(len(handle_df)).reshape(-1, 1)
            reg_high = LinearRegression().fit(x_handle, handle_df["high"].values)
            y_high_fit = reg_high.predict(x_handle)
            
            fig.add_trace(go.Scatter(
                x=handle_df.index,
                y=y_high_fit,
                mode="lines",
                name="Handle Highs Trend",
                line=dict(color="#FFA726", width=2, dash="dash")
            ))
            
            # Linear regression on handle lows
            reg_low = LinearRegression().fit(x_handle, handle_df["low"].values)
            y_low_fit = reg_low.predict(x_handle)
            
            fig.add_trace(go.Scatter(
                x=handle_df.index,
                y=y_low_fit,
                mode="lines",
                name="Handle Lows Trend",
                line=dict(color="#42A5F5", width=2, dash="dash")
            ))
        
        # Add breakout point
        breakout_price = ohlcv.loc[breakout_ts, "high"]
        fig.add_trace(go.Scatter(
            x=[breakout_ts],
            y=[breakout_price],
            mode="markers+text",
            marker=dict(symbol="star", size=20, color="#FFD700", line=dict(color="#000", width=2)),
            text=["BREAKOUT"],
            textposition="top center",
            textfont=dict(size=14, color="#000"),
            name="Breakout Point"
        ))
        
        # Add cup and handle annotations
        fig.add_annotation(
            x=start_ts,
            y=cup_df["high"].max(),
            text="CUP START",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF6B6B",
            font=dict(size=12, color="#FF6B6B")
        )
        
        fig.add_annotation(
            x=end_ts,
            y=cup_df["high"].max(),
            text="CUP END",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#FF6B6B",
            font=dict(size=12, color="#FF6B6B")
        )
        
        # Add pattern statistics
        cup_depth = intervals.loc[i, CUP_DEPTH]
        cup_duration = intervals.loc[i, CUP_DURATION]
        handle_depth = intervals.loc[i, HANDLE_DEPTH]
        handle_duration = intervals.loc[i, HANDLE_DURATION]
        
        stats_text = f"""
        <b>Pattern Statistics:</b><br>
        Cup Depth: {cup_depth:.1f}<br>
        Cup Duration: {cup_duration} candles<br>
        Handle Depth: {handle_depth:.1f}<br>
        Handle Duration: {handle_duration} candles<br>
        R² Score: {r2:.3f}
        """
        
        fig.add_annotation(
            x=0.02,
            y=0.98,
            xref="paper",
            yref="paper",
            text=stats_text,
            showarrow=False,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#000",
            borderwidth=1,
            font=dict(size=10)
        )
        
        # Enhanced layout
        fig.update_layout(
            title=dict(
                text=f"Cup and Handle Pattern #{i} - {start_ts.strftime('%Y-%m-%d %H:%M')} to {breakout_ts.strftime('%Y-%m-%d %H:%M')}",
                x=0.5,
                font=dict(size=16, color="#2E3440")
            ),
            xaxis=dict(
                title="Time",
                rangeslider=dict(visible=False),
                gridcolor="rgba(128, 128, 128, 0.2)"
            ),
            yaxis=dict(
                title="Price (USDT)",
                gridcolor="rgba(128, 128, 128, 0.2)",
                side="left"
            ),
            yaxis2=dict(
                title="Volume",
                overlaying="y",
                side="right",
                showgrid=False,
                gridcolor="rgba(128, 128, 128, 0.2)"
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="#000",
                borderwidth=1
            ),
            plot_bgcolor="white",
            paper_bgcolor="white",
            width=1200,
            height=800,
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Save image using kaleido if requested
        if save_images:
            try:
                filename = f"cup_handle_{i}.png"
                filepath = os.path.join(output_dir, filename)
                fig.write_image(filepath, engine="kaleido")
                print(f"Saved pattern image: {filepath}")
            except Exception as e:
                print(f"Error saving image: {e}")
                # Fallback to show the plot
                fig.show()
        else:
            fig.show()

def create_pattern_summary_plot(intervals_df, save_image=True, output_dir="patterns"):
    """
    Create a summary plot showing all detected patterns
    """
    fig = go.Figure()
    
    # Add pattern markers
    for i, row in intervals_df.iterrows():
        start_time = row[START_TIME]
        end_time = row[END_TIME]
        breakout_time = row[BREAKOUT_TIME]
        r2 = row[R2]
        
        # Color based on R² score
        if r2 >= 0.9:
            color = "#4CAF50"  # Green for high quality
        elif r2 >= 0.85:
            color = "#FF9800"  # Orange for medium quality
        else:
            color = "#F44336"  # Red for low quality
        
        # Add pattern timeline
        fig.add_trace(go.Scatter(
            x=[start_time, end_time, breakout_time],
            y=[i, i, i],
            mode="lines+markers",
            line=dict(color=color, width=3),
            marker=dict(size=8, color=color),
            name=f"Pattern {i} (R²={r2:.3f})",
            showlegend=False
        ))
    
    fig.update_layout(
        title="Cup and Handle Patterns Timeline",
        xaxis_title="Time",
        yaxis_title="Pattern Index",
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=1200,
        height=600
    )
    
    if save_image:
        try:
            filepath = os.path.join(output_dir, "patterns_summary.png")
            fig.write_image(filepath, engine="kaleido")
            print(f"Saved summary plot: {filepath}")
        except Exception as e:
            print(f"Error saving summary plot: {e}")
    
    return fig

