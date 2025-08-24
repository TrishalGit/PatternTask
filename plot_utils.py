import pandas as pd
import plotly.graph_objects as go
from numpy.polynomial.polynomial import polyfit
import numpy as np
from sklearn.metrics import r2_score
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --------------------------
# Plot function for each interval
# --------------------------
def plot_interval(ohlcv, intervals, start_idx, end_idx):
    # # Ensure timestamp is datetime
    # ohlcv["timestamp"] = pd.to_datetime(ohlcv["timestamp"])

    # # Set datetime index
    # ohlcv.set_index("timestamp", inplace=True)
    for i in range(start_idx, end_idx):
        start_ts = intervals.loc[i, "Start time"]
        end_ts = intervals.loc[i, "Breakout time"]
        curve_end_ts = intervals.loc[i, "End time"]
        # Add -15 min and +15 min buffer
        plot_start = start_ts - pd.Timedelta(minutes=15)
        plot_end = end_ts + pd.Timedelta(minutes=15)
        curve_end = curve_end_ts + pd.Timedelta(minutes=1)
        
        # Slice OHLCV data
        # print(start_ts, end_ts)
        # print(plot_start, plot_end)
        # return 0
        data = ohlcv.loc[plot_start:plot_end]
        cup_df = ohlcv.loc[start_ts:curve_end_ts]
        reg_df = data.loc[curve_end:end_ts]
        min_idx = reg_df['low'].idxmin()
        reg_df = reg_df.loc[:min_idx]
        y = cup_df["close"].values
        x = np.arange(len(cup_df))
        # print(x)
        coefs = polyfit(x, y, 2)
        c, b, a = coefs
        # print("Parabola plot", a, b, c)
        y_fit = np.polyval(coefs[::-1], x)
        
        r2 = r2_score(y, y_fit)
        # print("R2", r2, cup_df.head(1), cup_df.tail(1))


        fig_fit = go.Scatter(
                x=cup_df.index,
                y=y_fit,
                mode="lines",
                name=f"Parabolic Fit (R²={r2:.2f})",
                line=dict(color="red", width=2, dash="dot")
            )

        # Create candlestick figure
        fig = go.Figure()

        # Candlestick
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            increasing_line_color="green",
            decreasing_line_color="red",
            name="Candles"
        ))

        # Add volume as bar chart (secondary y-axis)
        fig.add_trace(go.Bar(
            x=data.index,
            y=data["volume"],
            name="Volume",
            marker_color="blue",
            opacity=0.3,
            yaxis="y2"
        ))

        x = np.arange(len(reg_df)).reshape(-1, 1)
        reg_high = LinearRegression().fit(x, reg_df["high"].values)
        y_high_fit = reg_high.predict(x)

        fig_high_line = go.Scatter(
            x=reg_df.index,
            y=y_high_fit,
            mode="lines",
            name="Handle Highs Trend",
            line=dict(color="orange", width=2, dash="dash")
        )

        # Linear regression on lows
        reg_low = LinearRegression().fit(x, reg_df["low"].values)
        y_low_fit = reg_low.predict(x)

        fig_low_line = go.Scatter(
            x=reg_df.index,
            y=y_low_fit,
            mode="lines",
            name="Handle Lows Trend",
            line=dict(color="blue", width=2, dash="dash")
        )

        breakout_ts = end_ts   # assuming breakout is at end timestamp (adjust if you have another column)
        breakout_price = ohlcv.loc[breakout_ts, "close"]

        fig.add_trace(go.Scatter(
            x=[breakout_ts],
            y=[breakout_price],
            mode="markers+text",
            marker=dict(symbol="star", size=15, color="gold"),
            text=["Breakout"],
            textposition="top center",
            name="Breakout"
        ))

        # Add parabola fit if available
        if fig_fit:
            fig.add_trace(fig_fit)

        if fig_high_line:
            fig.add_trace(fig_high_line)
        if fig_low_line:
            fig.add_trace(fig_low_line)

        # Layout settings
        fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=False)),
            yaxis=dict(title="Price"),
            yaxis2=dict(title="Volume", overlaying="y", side="right", showgrid=False),
            title=f"Candlestick with Volume ({start_ts} → {end_ts})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # fig.write_image("patterns/cup_handle_" + str(i) + ".png")
        fig.show()

