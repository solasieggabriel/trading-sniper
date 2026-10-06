import os
import sys
from pathlib import Path
import requests
import numpy as np
import pandas as pd
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor

# ==========================================
# 1. CONFIGURATION & TELEGRAM CREDENTIALS
# ==========================================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

DATA_DIR = Path(__file__).resolve().parent
PARQUET_FILE = DATA_DIR / "market_data_15m.parquet"
CHART_FILE = DATA_DIR / "forecast_chart.png"

TICKER_GOLD = "GC=F"
TICKER_SILVER = "SI=F"

# ==========================================
# 2. DATA LAKE & MARKET INGESTION
# ==========================================
def save_parquet_safe(df, filepath):
    """Saves dataframe to parquet cleanly using native Arrow table."""
    reset_df = df.reset_index()
    first_col = reset_df.columns[0]
    reset_df.rename(columns={first_col: 'Timestamp'}, inplace=True)
    reset_df['Timestamp'] = pd.to_datetime(reset_df['Timestamp'], utc=True).dt.strftime('%Y-%m-%d %H:%M:%S%z')
    for col in reset_df.columns:
        if col != 'Timestamp':
            reset_df[col] = reset_df[col].astype('float64')
    arrow_arrays = [pa.array(reset_df[col]) for col in reset_df.columns]
    table = pa.Table.from_arrays(arrow_arrays, names=list(reset_df.columns))
    pq.write_table(table, filepath)

def load_parquet_safe(filepath):
    """Loads parquet into pandas with UTC DatetimeIndex."""
    table = pq.read_table(filepath)
    df = table.to_pandas()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)
    df = df.set_index('Timestamp').sort_index()
    return df

def fetch_clean_ticker(ticker, period="60d", interval="15m", prefix=""):
    print(f"--> Fetching {ticker} ({period}, {interval})...")
    raw_df = yf.download(ticker, period=period, interval=interval, progress=False)
    if isinstance(raw_df.columns, pd.MultiIndex):
        raw_df.columns = raw_df.columns.get_level_values(0)
    core_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    clean_df = raw_df[[c for c in core_cols if c in raw_df.columns]].copy().dropna()
    if prefix:
        clean_df = clean_df.add_prefix(prefix)
    return clean_df

def get_market_data():
    """Fetches, updates, and persists 15m historical data for Gold & Silver."""
    if PARQUET_FILE.exists():
        print(f"--> Found existing data lake at: {PARQUET_FILE}")
        existing_df = load_parquet_safe(PARQUET_FILE)
        new_gold = fetch_clean_ticker(TICKER_GOLD, period="5d", interval="15m", prefix="Gold_")
        new_silver = fetch_clean_ticker(TICKER_SILVER, period="5d", interval="15m", prefix="Silver_")
        new_merged = pd.merge(new_gold, new_silver, left_index=True, right_index=True, how='inner')
        if new_merged.index.tz is None:
            new_merged.index = new_merged.index.tz_localize('UTC')
        else:
            new_merged.index = new_merged.index.tz_convert('UTC')
        combined = pd.concat([existing_df, new_merged])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        save_parquet_safe(combined, PARQUET_FILE)
        print(f"--> Data lake updated! Total rows: {len(combined)}")
        return combined
    else:
        print("--> Initializing data lake from 60d download...")
        gold_df = fetch_clean_ticker(TICKER_GOLD, period="60d", interval="15m", prefix="Gold_")
        silver_df = fetch_clean_ticker(TICKER_SILVER, period="60d", interval="15m", prefix="Silver_")
        merged = pd.merge(gold_df, silver_df, left_index=True, right_index=True, how='inner').sort_index()
        if merged.index.tz is None:
            merged.index = merged.index.tz_localize('UTC')
        else:
            merged.index = merged.index.tz_convert('UTC')
        save_parquet_safe(merged, PARQUET_FILE)
        print(f"--> Data lake created! Total rows: {len(merged)}")
        return merged

# ==========================================
# 3. FEATURE & TARGET FACTORY
# ==========================================
def build_features_and_targets(df_raw, max_lags=16, max_horizon=16):
    df = df_raw.copy().sort_index()

    # Base Stationary Transforms (d=1)
    df['Gold_Close_diff1'] = df['Gold_Close'].diff(1)
    df['Gold_High_diff1']  = df['Gold_High'].diff(1)
    df['Gold_Low_diff1']   = df['Gold_Low'].diff(1)
    df['Gold_Open_diff1']  = df['Gold_Open'].diff(1)

    df['Silver_Close_diff1'] = df['Silver_Close'].diff(1)
    df['Silver_High_diff1']  = df['Silver_High'].diff(1)
    df['Silver_Low_diff1']   = df['Silver_Low'].diff(1)
    df['Silver_Open_diff1']  = df['Silver_Open'].diff(1)

    # Volatility / Spreads
    df['Gold_Range']   = df['Gold_High'] - df['Gold_Low']
    df['Silver_Range'] = df['Silver_High'] - df['Silver_Low']
    df['Gold_Log_Volume']   = np.log1p(df['Gold_Volume'])
    df['Silver_Log_Volume'] = np.log1p(df['Silver_Volume'])
    df['Gold_Vol_Ratio']   = df['Gold_Volume'] / (df['Gold_Volume'].rolling(20).mean() + 1e-6)
    df['Silver_Vol_Ratio'] = df['Silver_Volume'] / (df['Silver_Volume'].rolling(20).mean() + 1e-6)

    # Momentum
    df['Gold_Silver_Spread_diff'] = df['Gold_Close_diff1'] - df['Silver_Close_diff1']
    df['Gold_Ret_4']  = df['Gold_Close'].pct_change(4)
    df['Gold_Ret_8']  = df['Gold_Close'].pct_change(8)
    df['Gold_Ret_16'] = df['Gold_Close'].pct_change(16)
    df['Silver_Ret_4']  = df['Silver_Close'].pct_change(4)
    df['Silver_Ret_8']  = df['Silver_Close'].pct_change(8)
    df['Silver_Ret_16'] = df['Silver_Close'].pct_change(16)

    # 16 Lags
    lag_cols = ['Gold_Close_diff1', 'Gold_High_diff1', 'Gold_Low_diff1', 'Gold_Range', 'Gold_Vol_Ratio',
                'Silver_Close_diff1', 'Silver_High_diff1', 'Silver_Low_diff1', 'Silver_Range', 'Silver_Vol_Ratio']
    for lag in range(1, max_lags + 1):
        for col in lag_cols:
            df[f'{col}_lag{lag}'] = df[col].shift(lag)

    # Time Encodings
    if isinstance(df.index, pd.DatetimeIndex):
        minute_of_day = df.index.hour * 60 + df.index.minute
        df['Sin_Time'] = np.sin(2 * np.pi * minute_of_day / 1440)
        df['Cos_Time'] = np.cos(2 * np.pi * minute_of_day / 1440)
        df['DayOfWeek'] = df.index.dayofweek

    # Targets: 16 Steps & 4 Envelopes
    target_cols = []
    for step in range(1, max_horizon + 1):
        df[f'Gold_High_step_{step}']  = df['Gold_High'].shift(-step) - df['Gold_Close']
        df[f'Gold_Low_step_{step}']   = df['Gold_Low'].shift(-step) - df['Gold_Close']
        df[f'Gold_Close_step_{step}'] = df['Gold_Close'].shift(-step) - df['Gold_Close']
        df[f'Silver_High_step_{step}']  = df['Silver_High'].shift(-step) - df['Silver_Close']
        df[f'Silver_Low_step_{step}']   = df['Silver_Low'].shift(-step) - df['Silver_Close']
        df[f'Silver_Close_step_{step}'] = df['Silver_Close'].shift(-step) - df['Silver_Close']
        target_cols.extend([f'Gold_High_step_{step}', f'Gold_Low_step_{step}', f'Gold_Close_step_{step}',
                            f'Silver_High_step_{step}', f'Silver_Low_step_{step}', f'Silver_Close_step_{step}'])

    for h in [4, 8, 12, 16]:
        forward_g_high = [df['Gold_High'].shift(-i) for i in range(1, h + 1)]
        forward_g_low  = [df['Gold_Low'].shift(-i) for i in range(1, h + 1)]
        forward_s_high = [df['Silver_High'].shift(-i) for i in range(1, h + 1)]
        forward_s_low  = [df['Silver_Low'].shift(-i) for i in range(1, h + 1)]

        df[f'Gold_MFE_{h}'] = pd.concat(forward_g_high, axis=1).max(axis=1) - df['Gold_Close']
        df[f'Gold_MAE_{h}'] = pd.concat(forward_g_low, axis=1).min(axis=1) - df['Gold_Close']
        df[f'Silver_MFE_{h}'] = pd.concat(forward_s_high, axis=1).max(axis=1) - df['Silver_Close']
        df[f'Silver_MAE_{h}'] = pd.concat(forward_s_low, axis=1).min(axis=1) - df['Silver_Close']
        target_cols.extend([f'Gold_MFE_{h}', f'Gold_MAE_{h}', f'Silver_MFE_{h}', f'Silver_MAE_{h}'])

    raw_leaks = ['Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close', 'Gold_Volume',
                 'Silver_Open', 'Silver_High', 'Silver_Low', 'Silver_Close', 'Silver_Volume']
    feat_names = [c for c in df.columns if c not in target_cols and c not in raw_leaks]

    labeled_data = df.dropna(subset=feat_names + target_cols).copy()
    live_row = df.dropna(subset=feat_names).iloc[[-1]][feat_names].copy()

    return labeled_data, feat_names, live_row, df

# ==========================================
# 4. TRAINING & INFERENCE PIPELINE
# ==========================================
def run_sniper_engine():
    print("--> Starting Sniper Engine...")
    raw_market = get_market_data()
    labeled_data, feat_names, live_row, master_df = build_features_and_targets(raw_market)

    X_train = labeled_data[feat_names]
    y_gold_path = labeled_data[[f'Gold_High_step_{i}' for i in range(1, 17)] + 
                               [f'Gold_Low_step_{i}' for i in range(1, 17)] + 
                               [f'Gold_Close_step_{i}' for i in range(1, 17)]]
    y_gold_env = labeled_data[['Gold_MFE_4', 'Gold_MAE_4', 'Gold_MFE_8', 'Gold_MAE_8', 
                               'Gold_MFE_12', 'Gold_MAE_12', 'Gold_MFE_16', 'Gold_MAE_16']]

    print(f"--> Training on {len(labeled_data)} labeled historical candles...")
    lgbm_p = dict(n_estimators=120, learning_rate=0.04, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    xgb_p  = dict(n_estimators=100, learning_rate=0.04, max_depth=5, random_state=42, n_jobs=-1)

    m_lgbm_path = MultiOutputRegressor(LGBMRegressor(**lgbm_p)).fit(X_train, y_gold_path)
    m_xgb_path  = MultiOutputRegressor(XGBRegressor(**xgb_p)).fit(X_train, y_gold_path)
    m_lgbm_env  = MultiOutputRegressor(LGBMRegressor(**lgbm_p)).fit(X_train, y_gold_env)
    m_xgb_env   = MultiOutputRegressor(XGBRegressor(**xgb_p)).fit(X_train, y_gold_env)

    # Live Predictions
    latest_feat = live_row[feat_names]
    curr_gold_close = master_df['Gold_Close'].iloc[-1]
    curr_silver_close = master_df['Silver_Close'].iloc[-1]
    
    # Time in GMT+8
    if master_df.index.tz is None:
        idx_gmt8 = master_df.index.tz_localize('UTC').tz_convert('Asia/Singapore')
    else:
        idx_gmt8 = master_df.index.tz_convert('Asia/Singapore')
    curr_time_gmt8 = idx_gmt8[-1].strftime('%Y-%m-%d %H:%M [GMT+8]')

    # Gold Forecasts
    lgbm_path = m_lgbm_path.predict(latest_feat)[0]
    xgb_path  = m_xgb_path.predict(latest_feat)[0]
    lgbm_env  = m_lgbm_env.predict(latest_feat)[0]
    xgb_env   = m_xgb_env.predict(latest_feat)[0]

    # Generate Chart
    generate_forecast_chart(master_df, idx_gmt8, curr_gold_close, lgbm_path, xgb_path, CHART_FILE)

    # 2h Metrics for Gold
    lgbm_down_2h = abs(lgbm_env[3])
    lgbm_risk_2h = max(lgbm_env[2], 0.8)
    lgbm_rr_2h   = lgbm_down_2h / lgbm_risk_2h

    xgb_down_2h = abs(xgb_env[3])
    xgb_risk_2h = max(xgb_env[2], 0.8)
    xgb_rr_2h   = xgb_down_2h / xgb_risk_2h

    agree = (lgbm_rr_2h >= 2.5) and (xgb_rr_2h >= 2.5)
    verdict = "STRICT AGREEMENT (Both Models Trigger)" if agree else "DIVERGENCE / CAUTION"
    action = "SELL / SHORT" if agree else "STAND DOWN (Wait for Confluence)"

    hard_sl = curr_gold_close + max(lgbm_risk_2h, xgb_risk_2h) + 0.50
    tp1 = curr_gold_close - abs((lgbm_env[1] + xgb_env[1]) / 2.0)
    tp2 = curr_gold_close - min(lgbm_down_2h, xgb_down_2h)

    # Build Telegram Message
    message = (
        f"🎯 <b>[DUAL-KEY SNIPER SESSION REPORT]</b>\n"
        f"⏰ <b>Session Time:</b> {curr_time_gmt8}\n\n"
        f"📌 <b>Trading Instrument:</b> GOLD (<code>GC=F</code>)\n"
        f"💵 <b>Current Gold Price:</b> <code>${curr_gold_close:.2f}</code>\n"
        f"💵 <b>Current Silver Price:</b> <code>${curr_silver_close:.3f}</code>\n\n"
        f"📊 <b>2-Hour Forecast (Bar #8):</b>\n"
        f"• <b>LightGBM:</b> Down -${lgbm_down_2h:.2f} | Risk +${lgbm_risk_2h:.2f} | <b>{lgbm_rr_2h:.1f}:1 R:R</b>\n"
        f"• <b>XGBoost:</b> Down -${xgb_down_2h:.2f} | Risk +${xgb_risk_2h:.2f} | <b>{xgb_rr_2h:.1f}:1 R:R</b>\n"
        f"🚦 <b>Consensus:</b> <b>{verdict}</b>\n"
        f"──────────────────────────────\n"
        f"🛒 <b>Suggested Action:</b> <b>{action}</b>\n"
        f"🔴 <b>Hard Stop Loss:</b> <code>${hard_sl:.2f}</code> (Risk: ${hard_sl - curr_gold_close:.2f})\n"
        f"🟢 <b>Take Profit 1 (1h):</b> <code>${tp1:.2f}</code>\n"
        f"🟢 <b>Take Profit 2 (2h):</b> <code>${tp2:.2f}</code>\n"
        f"⏱️ <b>Time-Stop:</b> Bar #8 (+120 min)\n"
        f"──────────────────────────────\n"
        f"<i>Check the attached chart for full 16-candle trajectory and all 8 extreme boundaries.</i>"
    )

    send_telegram_notification(message, CHART_FILE)

# ==========================================
# 5. CHART GENERATOR
# ==========================================
def generate_forecast_chart(master_df, idx_gmt8, curr_close, lgbm_path, xgb_path, save_path):
    hist_len = 16
    hist_df = master_df.tail(hist_len).copy()
    future_indices = list(range(hist_len, hist_len + 16))

    lgbm_high  = [curr_close + lgbm_path[i]      for i in range(16)]
    lgbm_low   = [curr_close + lgbm_path[16 + i] for i in range(16)]
    lgbm_close = [curr_close + lgbm_path[32 + i] for i in range(16)]

    xgb_high  = [curr_close + xgb_path[i]      for i in range(16)]
    xgb_low   = [curr_close + xgb_path[16 + i] for i in range(16)]
    xgb_close = [curr_close + xgb_path[32 + i] for i in range(16)]

    x_conn = [hist_len - 1] + future_indices
    y_lgbm_c = [curr_close] + lgbm_close
    y_xgb_c  = [curr_close] + xgb_close

    fig, ax = plt.subplots(figsize=(22, 10), facecolor='#121212')
    ax.set_facecolor('#1a1a1a')

    # Historical
    bar_width = 0.65
    for i in range(hist_len):
        row = hist_df.iloc[i]
        c = '#26a69a' if row['Gold_Close'] >= row['Gold_Open'] else '#ef5350'
        ax.vlines(x=i, ymin=row['Gold_Low'], ymax=row['Gold_High'], color=c, linewidth=1.5)
        rect = patches.Rectangle((i - bar_width/2, min(row['Gold_Open'], row['Gold_Close'])), 
                                 bar_width, max(abs(row['Gold_Close'] - row['Gold_Open']), 0.1),
                                 facecolor=c, edgecolor=c, alpha=0.9)
        ax.add_patch(rect)

    # NOW Line
    ax.axvline(x=hist_len - 1, color='#ffffff', linestyle=':', linewidth=2.5, alpha=0.95)
    ax.text(hist_len - 0.85, curr_close + 10, ' NOW', color='#ffffff', fontsize=13, fontweight='bold')

    # Models
    ax.plot(x_conn, y_lgbm_c, color='#00e5ff', linewidth=3.0, marker='o', markersize=6, label='LightGBM Close')
    ax.fill_between(future_indices, lgbm_low, lgbm_high, color='#00e5ff', alpha=0.14, label='LightGBM Envelope')

    ax.plot(x_conn, y_xgb_c, color='#ffab00', linewidth=3.0, marker='s', markersize=6, label='XGBoost Close')
    ax.fill_between(future_indices, xgb_low, xgb_high, color='#ffab00', alpha=0.14, label='XGBoost Envelope')

    # Time-Stop
    ax.axvline(x=hist_len + 7, color='#ff5252', linestyle='--', linewidth=2.0, alpha=0.85)
    ax.text(hist_len + 7.25, curr_close - 4.5, '2h Time-Stop (+120m)', color='#ff5252', fontsize=12, fontweight='bold', rotation=90)
    ax.axhline(y=curr_close, color='#bdbdbd', linestyle='-.', linewidth=1.4, alpha=0.7, label=f'Current (${curr_close:.2f})')

    # Annotate Extremes (8 points)
    def callout(x, y, text, fc, ec, offset):
        ax.scatter(x, y, color=ec, s=75, zorder=6, edgecolor='#ffffff', linewidth=1.2)
        ax.annotate(text, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.5),
                    arrowprops=dict(arrowstyle="->", color=ec, lw=1.4),
                    color="#ffffff", fontsize=9.5, fontweight="bold", ha='center')

    # LGBM Extremes
    callout(future_indices[np.argmax(lgbm_high)], max(lgbm_high), f"LGBM Max: ${max(lgbm_high):.2f}", "#002b33", "#00e5ff", (-1.5, 3.5))
    callout(future_indices[np.argmin(lgbm_low)],  min(lgbm_low),  f"LGBM Min: ${min(lgbm_low):.2f}",  "#002b33", "#00e5ff", (-1.5, -4.0))
    # XGB Extremes
    callout(future_indices[np.argmax(xgb_high)],  max(xgb_high),  f"XGB Max: ${max(xgb_high):.2f}",  "#332200", "#ffab00", (1.5, 3.5))
    callout(future_indices[np.argmin(xgb_low)],   min(xgb_low),   f"XGB Min: ${min(xgb_low):.2f}",   "#332200", "#ffab00", (1.5, -4.0))

    # Ticks
    total_bars = hist_len + 16
    ax.set_xlim(-0.8, total_bars + 0.5)
    hist_ticks = [0, 4, 8, 12, 15]
    hist_gmt8_tail = idx_gmt8[-hist_len:]
    hist_labels = [hist_gmt8_tail[idx].strftime('%H:%M') for idx in hist_ticks]
    future_ticks = list(range(hist_len, total_bars))
    future_labels = [f"+{(s + 1) * 15}m" for s in range(16)]

    ax.set_xticks(hist_ticks + future_ticks)
    ax.set_xticklabels(hist_labels + future_labels, color='#e0e0e0', fontsize=11.5, fontweight='bold', rotation=45, ha='right')
    ax.tick_params(axis='y', colors='#e0e0e0', labelsize=13)

    for tick, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        if tick >= hist_len:
            label.set_color('#80d8ff')

    ax.set_title("Dual-Key Sniper Forecast: 16 Past Candles vs. 16 Projected Future Candles (+15m to +240m)", 
                 color='#ffffff', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Gold Price ($)", color='#ffffff', fontsize=15, fontweight='bold', labelpad=14)
    ax.set_xlabel("Timeline (Historical HH:MM [GMT+8] on Left | +Minutes Projected on Right)", 
                  color='#b0bec5', fontsize=14, fontweight='bold', labelpad=14)
    ax.grid(True, color='#2c2c2c', linestyle='--', linewidth=0.8, alpha=0.75)
    ax.legend(facecolor='#212121', edgecolor='#424242', labelcolor='#e0e0e0', loc='upper left', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=130, facecolor='#121212')
    plt.close()
    print(f"--> Forecast chart saved at: {save_path}")

# ==========================================
# 6. TELEGRAM MESSENGER
# ==========================================
def send_telegram_notification(text, image_path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("\n[WARNING] Telegram credentials not set. Printing report locally:")
        print(text)
        return

    print("--> Sending Telegram notification...")
    # Send Text
    text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    res_text = requests.post(text_url, json=payload, timeout=15)
    
    if res_text.status_code == 200:
        print("--> Text report sent successfully!")
    else:
        print(f"[ERROR] Failed to send text: {res_text.text}")

    # Send Photo
    if image_path.exists():
        photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
        with open(image_path, "rb") as f:
            res_photo = requests.post(photo_url, data={"chat_id": TELEGRAM_CHAT_ID}, files={"photo": f}, timeout=30)
            if res_photo.status_code == 200:
                print("--> Forecast chart image sent successfully!")
            else:
                print(f"[ERROR] Failed to send photo: {res_photo.text}")

if __name__ == "__main__":
    run_sniper_engine()
