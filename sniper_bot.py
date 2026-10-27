import os
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
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
# 3. MULTI-SCALE POI FEATURE FACTORY
# ==========================================
def build_features_and_targets(df_raw, max_lags=16, max_horizon=16):
    df = df_raw.copy().sort_index()

    # Time Encodings in GMT+8
    if isinstance(df.index, pd.DatetimeIndex):
        dt_gmt8 = df.index.tz_convert('Asia/Singapore') if df.index.tz is not None else df.index.tz_localize('UTC').tz_convert('Asia/Singapore')
    else:
        dt_gmt8 = pd.to_datetime(df.index, utc=True).tz_convert('Asia/Singapore')

    df['Date_GMT8'] = dt_gmt8.date
    df['Hour_GMT8'] = dt_gmt8.hour
    minute_of_day = dt_gmt8.hour * 60 + dt_gmt8.minute
    df['Sin_Time'] = np.sin(2 * np.pi * minute_of_day / 1440)
    df['Cos_Time'] = np.cos(2 * np.pi * minute_of_day / 1440)
    df['DayOfWeek'] = dt_gmt8.dayofweek

    # -------------------------------------------------------------
    # TIER 1: MACRO & WEEKLY CONTEXT (Up to 480 bars = 5 Days / 120h)
    # -------------------------------------------------------------
    # 5-Day Rolling Extremes & Weekly Range Exhaustion
    df['Gold_5d_High'] = df['Gold_High'].rolling(480, min_periods=96).max()
    df['Gold_5d_Low']  = df['Gold_Low'].rolling(480, min_periods=96).min()
    df['Gold_Dist_5d_High'] = (df['Gold_Close'] - df['Gold_5d_High']) / (df['Gold_Close'] + 1e-6)
    df['Gold_Dist_5d_Low']  = (df['Gold_Close'] - df['Gold_5d_Low']) / (df['Gold_Close'] + 1e-6)
    df['Gold_Weekly_Range_Pct'] = (df['Gold_Close'] - df['Gold_5d_Low']) / (df['Gold_5d_High'] - df['Gold_5d_Low'] + 1e-6)

    # Multi-Day Trend Regimes (50 EMA = 12.5h, 200 EMA = 50h)
    df['Gold_EMA_50']  = df['Gold_Close'].ewm(span=50, adjust=False).mean()
    df['Gold_EMA_200'] = df['Gold_Close'].ewm(span=200, adjust=False).mean()
    df['Gold_EMA_Slope']   = (df['Gold_EMA_50'] - df['Gold_EMA_200']) / (df['Gold_Close'] + 1e-6)
    df['Gold_Dist_EMA200'] = (df['Gold_Close'] - df['Gold_EMA_200']) / (df['Gold_Close'] + 1e-6)

    # Silver Macro
    df['Silver_5d_High'] = df['Silver_High'].rolling(480, min_periods=96).max()
    df['Silver_5d_Low']  = df['Silver_Low'].rolling(480, min_periods=96).min()
    df['Silver_Dist_5d_High'] = (df['Silver_Close'] - df['Silver_5d_High']) / (df['Silver_Close'] + 1e-6)
    df['Silver_Dist_5d_Low']  = (df['Silver_Close'] - df['Silver_5d_Low']) / (df['Silver_Close'] + 1e-6)
    df['Silver_Weekly_Range_Pct'] = (df['Silver_Close'] - df['Silver_5d_Low']) / (df['Silver_5d_High'] - df['Silver_5d_Low'] + 1e-6)
    df['Silver_EMA_50']  = df['Silver_Close'].ewm(span=50, adjust=False).mean()
    df['Silver_EMA_200'] = df['Silver_Close'].ewm(span=200, adjust=False).mean()
    df['Silver_EMA_Slope'] = (df['Silver_EMA_50'] - df['Silver_EMA_200']) / (df['Silver_Close'] + 1e-6)

    # -------------------------------------------------------------
    # TIER 2: SESSION & DAILY POIs (96 bars = 24 Hours)
    # -------------------------------------------------------------
    # Daily Calendar Anchors (PDH, PDL, PDC, Daily Open)
    daily_stats = df.groupby('Date_GMT8').agg(
        Day_High=('Gold_High', 'max'),
        Day_Low=('Gold_Low', 'min'),
        Day_Open=('Gold_Open', 'first'),
        Day_Close=('Gold_Close', 'last')
    )
    daily_stats['PDH'] = daily_stats['Day_High'].shift(1)
    daily_stats['PDL'] = daily_stats['Day_Low'].shift(1)
    daily_stats['PDC'] = daily_stats['Day_Close'].shift(1)
    daily_stats['Daily_Open'] = daily_stats['Day_Open']

    df = df.join(daily_stats[['PDH', 'PDL', 'PDC', 'Daily_Open']], on='Date_GMT8')
    df['PDH'] = df['PDH'].ffill()
    df['PDL'] = df['PDL'].ffill()
    df['PDC'] = df['PDC'].ffill()
    df['Daily_Open'] = df['Daily_Open'].ffill()

    # Asian Session POIs (08:00 - 15:59 GMT+8)
    asian_mask = (df['Hour_GMT8'] >= 8) & (df['Hour_GMT8'] < 16)
    asian_stats = df[asian_mask].groupby('Date_GMT8').agg(
        ASH=('Gold_High', 'max'),
        ASL=('Gold_Low', 'min')
    )
    df = df.join(asian_stats, on='Date_GMT8')
    df['ASH'] = df['ASH'].ffill()
    df['ASL'] = df['ASL'].ffill()

    # Distances to POIs
    df['Dist_PDH'] = (df['Gold_Close'] - df['PDH']) / (df['Gold_Close'] + 1e-6)
    df['Dist_PDL'] = (df['Gold_Close'] - df['PDL']) / (df['Gold_Close'] + 1e-6)
    df['Dist_Daily_Open'] = (df['Gold_Close'] - df['Daily_Open']) / (df['Gold_Close'] + 1e-6)
    df['Dist_ASH'] = (df['Gold_Close'] - df['ASH']) / (df['Gold_Close'] + 1e-6)
    df['Dist_ASL'] = (df['Gold_Close'] - df['ASL']) / (df['Gold_Close'] + 1e-6)

    # 24h Rolling Volume-Weighted Average Price (VWAP)
    rolling_pv = (df['Gold_Close'] * df['Gold_Volume']).rolling(96).sum()
    rolling_v  = df['Gold_Volume'].rolling(96).sum()
    df['Gold_VWAP_24h'] = rolling_pv / (rolling_v + 1e-6)
    df['Dist_VWAP_24h'] = (df['Gold_Close'] - df['Gold_VWAP_24h']) / (df['Gold_Close'] + 1e-6)

    # 24h ATR & Volatility Squeeze (Range Compression)
    df['Gold_Range']   = df['Gold_High'] - df['Gold_Low']
    df['Silver_Range'] = df['Silver_High'] - df['Silver_Low']
    df['Gold_ATR_24h'] = df['Gold_Range'].rolling(96).mean()
    df['Range_Compression'] = df['Gold_Range'].rolling(16).mean() / (df['Gold_ATR_24h'] + 1e-6)

    # -------------------------------------------------------------
    # TIER 3: MICRO DYNAMICS & STATIONARY DIFFERENCING (d=1, 4 Hours)
    # -------------------------------------------------------------
    df['Gold_Close_diff1'] = df['Gold_Close'].diff(1)
    df['Gold_High_diff1']  = df['Gold_High'].diff(1)
    df['Gold_Low_diff1']   = df['Gold_Low'].diff(1)
    df['Gold_Open_diff1']  = df['Gold_Open'].diff(1)

    df['Silver_Close_diff1'] = df['Silver_Close'].diff(1)
    df['Silver_High_diff1']  = df['Silver_High'].diff(1)
    df['Silver_Low_diff1']   = df['Silver_Low'].diff(1)
    df['Silver_Open_diff1']  = df['Silver_Open'].diff(1)

    df['Gold_Log_Volume']   = np.log1p(df['Gold_Volume'])
    df['Silver_Log_Volume'] = np.log1p(df['Silver_Volume'])
    df['Gold_Vol_Ratio']   = df['Gold_Volume'] / (df['Gold_Volume'].rolling(20).mean() + 1e-6)
    df['Silver_Vol_Ratio'] = df['Silver_Volume'] / (df['Silver_Volume'].rolling(20).mean() + 1e-6)

    # Cross-Asset Momentum & Spillovers
    df['Gold_Silver_Spread_diff'] = df['Gold_Close_diff1'] - df['Silver_Close_diff1']
    df['Gold_Ret_4']  = df['Gold_Close'].pct_change(4)
    df['Gold_Ret_8']  = df['Gold_Close'].pct_change(8)
    df['Gold_Ret_16'] = df['Gold_Close'].pct_change(16)
    df['Silver_Ret_4']  = df['Silver_Close'].pct_change(4)
    df['Silver_Ret_8']  = df['Silver_Close'].pct_change(8)
    df['Silver_Ret_16'] = df['Silver_Close'].pct_change(16)

    # 16 Lags & Forward Targets (Batch Concatenation to avoid fragmentation)
    extra_cols = {}
    lag_cols = ['Gold_Close_diff1', 'Gold_High_diff1', 'Gold_Low_diff1', 'Gold_Range', 'Gold_Vol_Ratio',
                'Silver_Close_diff1', 'Silver_High_diff1', 'Silver_Low_diff1', 'Silver_Range', 'Silver_Vol_Ratio']
    for lag in range(1, max_lags + 1):
        for col in lag_cols:
            extra_cols[f'{col}_lag{lag}'] = df[col].shift(lag)

    target_cols = []
    for step in range(1, max_horizon + 1):
        extra_cols[f'Gold_High_step_{step}']  = df['Gold_High'].shift(-step) - df['Gold_Close']
        extra_cols[f'Gold_Low_step_{step}']   = df['Gold_Low'].shift(-step) - df['Gold_Close']
        extra_cols[f'Gold_Close_step_{step}'] = df['Gold_Close'].shift(-step) - df['Gold_Close']
        extra_cols[f'Silver_High_step_{step}']  = df['Silver_High'].shift(-step) - df['Silver_Close']
        extra_cols[f'Silver_Low_step_{step}']   = df['Silver_Low'].shift(-step) - df['Silver_Close']
        extra_cols[f'Silver_Close_step_{step}'] = df['Silver_Close'].shift(-step) - df['Silver_Close']
        target_cols.extend([f'Gold_High_step_{step}', f'Gold_Low_step_{step}', f'Gold_Close_step_{step}',
                            f'Silver_High_step_{step}', f'Silver_Low_step_{step}', f'Silver_Close_step_{step}'])

    for h in [4, 8, 12, 16]:
        forward_g_high = [df['Gold_High'].shift(-i) for i in range(1, h + 1)]
        forward_g_low  = [df['Gold_Low'].shift(-i) for i in range(1, h + 1)]
        forward_s_high = [df['Silver_High'].shift(-i) for i in range(1, h + 1)]
        forward_s_low  = [df['Silver_Low'].shift(-i) for i in range(1, h + 1)]

        extra_cols[f'Gold_MFE_{h}'] = pd.concat(forward_g_high, axis=1).max(axis=1) - df['Gold_Close']
        extra_cols[f'Gold_MAE_{h}'] = pd.concat(forward_g_low, axis=1).min(axis=1) - df['Gold_Close']
        extra_cols[f'Silver_MFE_{h}'] = pd.concat(forward_s_high, axis=1).max(axis=1) - df['Silver_Close']
        extra_cols[f'Silver_MAE_{h}'] = pd.concat(forward_s_low, axis=1).min(axis=1) - df['Silver_Close']
        target_cols.extend([f'Gold_MFE_{h}', f'Gold_MAE_{h}', f'Silver_MFE_{h}', f'Silver_MAE_{h}'])

    df = pd.concat([df, pd.DataFrame(extra_cols, index=df.index)], axis=1)

    raw_leaks = ['Gold_Open', 'Gold_High', 'Gold_Low', 'Gold_Close', 'Gold_Volume',
                 'Silver_Open', 'Silver_High', 'Silver_Low', 'Silver_Close', 'Silver_Volume',
                 'Date_GMT8', 'PDH', 'PDL', 'PDC', 'Daily_Open', 'ASH', 'ASL',
                 'Gold_5d_High', 'Gold_5d_Low', 'Silver_5d_High', 'Silver_5d_Low',
                 'Gold_EMA_50', 'Gold_EMA_200', 'Silver_EMA_50', 'Silver_EMA_200',
                 'Gold_VWAP_24h']
    feat_names = [c for c in df.columns if c not in target_cols and c not in raw_leaks]

    labeled_data = df.dropna(subset=feat_names + target_cols).copy()
    live_row = df.dropna(subset=feat_names).iloc[[-1]][feat_names].copy()

    return labeled_data, feat_names, live_row, df

# ==========================================
# 4. TRAINING & ASYMMETRIC POI ENGINE
# ==========================================
def run_sniper_engine():
    print("--> Starting Multi-Scale POI Sniper Engine...")
    raw_market = get_market_data()
    labeled_data, feat_names, live_row, master_df = build_features_and_targets(raw_market)

    X_train = labeled_data[feat_names]
    y_gold_path = labeled_data[[f'Gold_High_step_{i}' for i in range(1, 17)] + 
                               [f'Gold_Low_step_{i}' for i in range(1, 17)] + 
                               [f'Gold_Close_step_{i}' for i in range(1, 17)]]
    y_gold_env = labeled_data[['Gold_MFE_4', 'Gold_MAE_4', 'Gold_MFE_8', 'Gold_MAE_8', 
                               'Gold_MFE_12', 'Gold_MAE_12', 'Gold_MFE_16', 'Gold_MAE_16']]

    print(f"--> Training on {len(labeled_data)} labeled candles across 1-week macro hierarchy ({len(feat_names)} features)...")
    lgbm_env_p  = dict(n_estimators=45, learning_rate=0.06, num_leaves=15, max_bin=63, random_state=42, n_jobs=-1, verbose=-1)
    xgb_env_p   = dict(n_estimators=40, learning_rate=0.06, max_depth=3, tree_method='hist', random_state=42, n_jobs=-1)
    
    lgbm_path_p = dict(n_estimators=25, learning_rate=0.08, num_leaves=15, max_bin=63, random_state=42, n_jobs=-1, verbose=-1)
    xgb_path_p  = dict(n_estimators=20, learning_rate=0.08, max_depth=3, tree_method='hist', random_state=42, n_jobs=-1)

    print("--> Fitting LightGBM Path & Envelope Engines...")
    m_lgbm_path = MultiOutputRegressor(LGBMRegressor(**lgbm_path_p)).fit(X_train, y_gold_path)
    m_lgbm_env  = MultiOutputRegressor(LGBMRegressor(**lgbm_env_p)).fit(X_train, y_gold_env)

    print("--> Fitting XGBoost Path & Envelope Engines...")
    m_xgb_path  = MultiOutputRegressor(XGBRegressor(**xgb_path_p)).fit(X_train, y_gold_path)
    m_xgb_env   = MultiOutputRegressor(XGBRegressor(**xgb_env_p)).fit(X_train, y_gold_env)

    # Live Predictions
    latest_feat = live_row[feat_names]
    curr_gold_close = master_df['Gold_Close'].iloc[-1]
    curr_silver_close = master_df['Silver_Close'].iloc[-1]
    
    # Time in GMT+8
    if master_df.index.tz is None:
        idx_gmt8 = master_df.index.tz_localize('UTC').tz_convert('Asia/Singapore')
    else:
        idx_gmt8 = master_df.index.tz_convert('Asia/Singapore')
    
    latest_dt = idx_gmt8[-1]
    curr_time_gmt8 = latest_dt.strftime('%Y-%m-%d %H:%M [GMT+8]')
    curr_hour = latest_dt.hour

    # Session Context Tagging (09:00, 17:00, 21:00 GMT+8 targets)
    if curr_hour == 9:
        session_title = "ASIAN SESSION (Post-Open Initial Balance Sweep)"
    elif curr_hour == 17:
        session_title = "LONDON SESSION (Post-Open Judas Swing Resolution)"
    elif curr_hour == 21:
        session_title = "NEW YORK SESSION (Post-Open US Cash Expansion)"
    else:
        session_title = f"INTRADAY SESSION ({latest_dt.strftime('%H:%M')} GMT+8)"

    # Live Model Forecasts
    lgbm_path = m_lgbm_path.predict(latest_feat)[0]
    xgb_path  = m_xgb_path.predict(latest_feat)[0]
    lgbm_env  = m_lgbm_env.predict(latest_feat)[0]
    xgb_env   = m_xgb_env.predict(latest_feat)[0]

    # Model Predicted Expansions
    lgbm_up_2h   = lgbm_env[2]
    lgbm_down_2h = abs(lgbm_env[3])
    xgb_up_2h    = xgb_env[2]
    xgb_down_2h  = abs(xgb_env[3])

    lgbm_up_4h   = lgbm_env[6]
    lgbm_down_4h = abs(lgbm_env[7])
    xgb_up_4h    = xgb_env[6]
    xgb_down_4h  = abs(xgb_env[7])

    # -------------------------------------------------------------
    # EXTRACT LIVE STRUCTURAL POIs
    # -------------------------------------------------------------
    curr_pdh = master_df['PDH'].iloc[-1]
    curr_pdl = master_df['PDL'].iloc[-1]
    curr_ash = master_df['ASH'].iloc[-1]
    curr_asl = master_df['ASL'].iloc[-1]
    curr_do  = master_df['Daily_Open'].iloc[-1]
    curr_5d_high = master_df['Gold_5d_High'].iloc[-1]
    curr_5d_low  = master_df['Gold_5d_Low'].iloc[-1]
    curr_week_pct = master_df['Gold_Weekly_Range_Pct'].iloc[-1] * 100
    curr_ema_slope = master_df['Gold_EMA_Slope'].iloc[-1]

    swing_low_8  = master_df['Gold_Low'].iloc[-8:].min()
    swing_high_8 = master_df['Gold_High'].iloc[-8:].max()

    # Direction Bias Evaluation
    bull_score = (lgbm_up_2h > lgbm_down_2h) + (xgb_up_2h > xgb_down_2h) + (curr_ema_slope > 0)
    bear_score = (lgbm_down_2h > lgbm_up_2h) + (xgb_down_2h > xgb_up_2h) + (curr_ema_slope < 0)

    # -------------------------------------------------------------
    # ASYMMETRIC POI-ANCHORED RISK & TARGET DEFINITION
    # -------------------------------------------------------------
    entry_price = curr_gold_close

    if (lgbm_down_2h > lgbm_up_2h) and (xgb_down_2h > xgb_up_2h):
        direction = "BEARISH (SELL / SHORT)"
        trigger_ref = "SILVER Breakdown -> Leads Gold Low (F=11.28, p<0.0008)"
        action = "SELL / SHORT GOLD"
        instrument = "GOLD (GC=F)"
        
        # Invalidation ceiling: Above local 8-bar high or Asian High
        structural_ceiling = max(swing_high_8, curr_ash if entry_price < curr_ash else swing_high_8)
        raw_risk = (structural_ceiling + 0.50) - entry_price
        risk_dist = float(np.clip(raw_risk, 2.50, 5.50))
        hard_sl = entry_price + risk_dist
        
        # Asymmetric Take Profits
        # TP1: 1h partial target (>= 2.2:1 R:R)
        tp1 = entry_price - max(risk_dist * 2.2, (abs(lgbm_env[1]) + abs(xgb_env[1])) / 2.0)
        
        # TP2: 2h-4h macro target (>= 3.8:1 to 5.0+:1 R:R), targeting PDL/5d Low if nearby
        base_tp2_dist = max(risk_dist * 3.8, min(lgbm_down_4h, xgb_down_4h))
        tp2 = entry_price - base_tp2_dist
        if curr_pdl < entry_price - (risk_dist * 2.5):
            tp2 = min(tp2, curr_pdl)
        
        rr_tp1 = abs(entry_price - tp1) / risk_dist
        rr_tp2 = abs(entry_price - tp2) / risk_dist
        verdict = f"STRICT AGREEMENT (Both Models Trigger SHORT | {rr_tp2:.1f}:1 R:R)"
        poi_invalidation_str = f"Above 8-Bar Pivot / ASH (${structural_ceiling:.2f} + $0.50)"

    elif (lgbm_up_2h > lgbm_down_2h) and (xgb_up_2h > xgb_down_2h):
        direction = "BULLISH (BUY / LONG)"
        trigger_ref = "GOLD Breakout -> Leads Silver High (F=4.12, High Beta)"
        action = "BUY / LONG SILVER (or Gold)"
        instrument = "SILVER (SI=F) / GOLD (GC=F)"
        
        # Invalidation floor: Below local 8-bar low or Asian Low
        structural_floor = min(swing_low_8, curr_asl if entry_price > curr_asl else swing_low_8)
        raw_risk = entry_price - (structural_floor - 0.50)
        risk_dist = float(np.clip(raw_risk, 2.50, 5.50))
        hard_sl = entry_price - risk_dist
        
        # Asymmetric Take Profits
        # TP1: 1h partial target (>= 2.2:1 R:R)
        tp1 = entry_price + max(risk_dist * 2.2, (lgbm_env[0] + xgb_env[0]) / 2.0)
        
        # TP2: 2h-4h macro target (>= 3.8:1 to 5.0+:1 R:R), targeting PDH/5d High if nearby
        base_tp2_dist = max(risk_dist * 3.8, min(lgbm_up_4h, xgb_up_4h))
        tp2 = entry_price + base_tp2_dist
        if curr_pdh > entry_price + (risk_dist * 2.5):
            tp2 = max(tp2, curr_pdh)
            
        rr_tp1 = abs(tp1 - entry_price) / risk_dist
        rr_tp2 = abs(tp2 - entry_price) / risk_dist
        verdict = f"STRICT AGREEMENT (Both Models Trigger LONG | {rr_tp2:.1f}:1 R:R)"
        poi_invalidation_str = f"Below 8-Bar Pivot / ASL (${structural_floor:.2f} - $0.50)"

    else:
        direction = "NEUTRAL / CHOP"
        trigger_ref = "Cross-Asset Range Compression / Model Divergence"
        action = "STAND DOWN (Wait for Session POI Sweep)"
        instrument = "NONE (Capital Protected)"
        risk_dist = 0.0
        hard_sl = entry_price
        tp1 = entry_price
        tp2 = entry_price
        rr_tp1 = 0.0
        rr_tp2 = 0.0
        verdict = "DIVERGENCE / RANGE CHOP (No Clean Asymmetric Setup)"
        poi_invalidation_str = "N/A (Chop Regime)"

    # Compute 8 Extreme Points (4-Hour Window)
    l_h = [entry_price + lgbm_path[i] for i in range(16)]
    l_l = [entry_price + lgbm_path[16 + i] for i in range(16)]
    l_c = [entry_price + lgbm_path[32 + i] for i in range(16)]

    x_h = [entry_price + xgb_path[i] for i in range(16)]
    x_l = [entry_price + xgb_path[16 + i] for i in range(16)]
    x_c = [entry_price + xgb_path[32 + i] for i in range(16)]

    l_max_h = (int(np.argmax(l_h)) + 1) * 15, max(l_h)
    l_min_l = (int(np.argmin(l_l)) + 1) * 15, min(l_l)
    l_max_c = (int(np.argmax(l_c)) + 1) * 15, max(l_c)
    l_min_c = (int(np.argmin(l_c)) + 1) * 15, min(l_c)

    x_max_h = (int(np.argmax(x_h)) + 1) * 15, max(x_h)
    x_min_l = (int(np.argmin(x_l)) + 1) * 15, min(x_l)
    x_max_c = (int(np.argmax(x_c)) + 1) * 15, max(x_c)
    x_min_c = (int(np.argmin(x_c)) + 1) * 15, min(x_c)

    # -------------------------------------------------------------
    # GENERATE POI-ENRICHED FORECAST CHART
    # -------------------------------------------------------------
    poi_dict = {
        'PDH': curr_pdh, 'PDL': curr_pdl,
        'ASH': curr_ash, 'ASL': curr_asl,
        'Entry': entry_price, 'SL': hard_sl, 'TP1': tp1, 'TP2': tp2,
        'Action': action, 'Direction': direction, 'Risk': risk_dist,
        'RR_TP1': rr_tp1, 'RR_TP2': rr_tp2
    }
    generate_forecast_chart(master_df, idx_gmt8, entry_price, lgbm_path, xgb_path, poi_dict, CHART_FILE)

    # -------------------------------------------------------------
    # BUILD TELEGRAM REPORT
    # -------------------------------------------------------------
    trend_regime_str = "BULLISH (EMA50 > EMA200)" if curr_ema_slope > 0 else "BEARISH (EMA50 < EMA200)"
    
    message = (
        f"🎯 <b>[MULTI-SCALE POI SNIPER REPORT]</b>\n"
        f"🏛️ <b>Session:</b> <code>{session_title}</code>\n"
        f"⏰ <b>Time:</b> {curr_time_gmt8}\n\n"
        f"📌 <b>Trading Asset:</b> <code>{instrument}</code>\n"
        f"🔗 <b>Causality Driver:</b> {trigger_ref}\n"
        f"🧭 <b>1-Week Macro Regime:</b> <code>{trend_regime_str}</code> (Range: {curr_week_pct:.1f}%)\n"
        f"💵 <b>Current Gold Close:</b> <code>${entry_price:.2f}</code>\n"
        f"💵 <b>Current Silver Close:</b> <code>${curr_silver_close:.3f}</code>\n\n"
        f"📍 <b>Key Market Structure POIs:</b>\n"
        f"• <b>PDH / PDL:</b> <code>${curr_pdh:.2f}</code> / <code>${curr_pdl:.2f}</code>\n"
        f"• <b>Asian Range (ASH/ASL):</b> <code>${curr_ash:.2f}</code> / <code>${curr_asl:.2f}</code>\n"
        f"• <b>5-Day High / Low:</b> <code>${curr_5d_high:.2f}</code> / <code>${curr_5d_low:.2f}</code>\n\n"
        f"📊 <b>Model Consensus & Asymmetric Setup:</b>\n"
        f"• <b>Direction Bias:</b> <b>{direction}</b>\n"
        f"• <b>LightGBM 2h Target:</b> Upside +${lgbm_up_2h:.2f} | Down -${lgbm_down_2h:.2f}\n"
        f"• <b>XGBoost  2h Target:</b> Upside +${xgb_up_2h:.2f} | Down -${xgb_down_2h:.2f}\n"
        f"🚦 <b>Verdict:</b> <b>{verdict}</b>\n"
        f"──────────────────────────────\n"
        f"🛒 <b>Action:</b> <b>{action}</b>\n"
        f"🔴 <b>Structural Stop Loss:</b> <code>${hard_sl:.2f}</code> (Risk: <b>${risk_dist:.2f}</b>)\n"
        f"   └ <i>Invalidation: {poi_invalidation_str}</i>\n"
        f"🟢 <b>Take Profit 1 (1h):</b> <code>${tp1:.2f}</code> (<b>{rr_tp1:.1f}:1 R:R</b>)\n"
        f"🟢 <b>Take Profit 2 (2h-4h POI):</b> <code>${tp2:.2f}</code> (<b>{rr_tp2:.1f}:1 R:R</b>)\n"
        f"⏱️ <b>Time-Stop:</b> Bar #8 (+120 min) - Scratch if no expansion\n"
        f"──────────────────────────────\n"
        f"📍 <b>8 Forecast Extremes (Next 4 Hours):</b>\n"
        f"🔵 <b>LightGBM:</b> High <code>${l_max_h[1]:.2f}</code> (+{l_max_h[0]}m) | Low <code>${l_min_l[1]:.2f}</code> (+{l_min_l[0]}m)\n"
        f"🟠 <b>XGBoost:</b>  High <code>${x_max_h[1]:.2f}</code> (+{x_max_h[0]}m) | Low <code>${x_min_l[1]:.2f}</code> (+{x_min_l[0]}m)\n"
        f"──────────────────────────────\n"
        f"<i>Inspect the attached chart for POI levels, entry, SL, and 16-candle projection.</i>"
    )

    send_telegram_notification(message, CHART_FILE)

# ==========================================
# 5. POI-ENRICHED CHART GENERATOR
# ==========================================
def generate_forecast_chart(master_df, idx_gmt8, curr_close, lgbm_path, xgb_path, poi_dict, save_path):
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

    fig, ax = plt.subplots(figsize=(24, 11), facecolor='#121212')
    ax.set_facecolor('#181818')

    # Historical Candlesticks
    bar_width = 0.65
    for i in range(hist_len):
        row = hist_df.iloc[i]
        c = '#26a69a' if row['Gold_Close'] >= row['Gold_Open'] else '#ef5350'
        ax.vlines(x=i, ymin=row['Gold_Low'], ymax=row['Gold_High'], color=c, linewidth=1.5)
        rect = patches.Rectangle((i - bar_width/2, min(row['Gold_Open'], row['Gold_Close'])), 
                                 bar_width, max(abs(row['Gold_Close'] - row['Gold_Open']), 0.1),
                                 facecolor=c, edgecolor=c, alpha=0.9)
        ax.add_patch(rect)

    # NOW Vertical Line
    ax.axvline(x=hist_len - 1, color='#ffffff', linestyle=':', linewidth=2.5, alpha=0.95)
    ax.text(hist_len - 0.85, curr_close + 8, ' NOW', color='#ffffff', fontsize=13, fontweight='bold')

    # Models Projections
    ax.plot(x_conn, y_lgbm_c, color='#00e5ff', linewidth=3.0, marker='o', markersize=6, label='LightGBM Projected Path')
    ax.fill_between(future_indices, lgbm_low, lgbm_high, color='#00e5ff', alpha=0.12, label='LightGBM Envelope')

    ax.plot(x_conn, y_xgb_c, color='#ffab00', linewidth=3.0, marker='s', markersize=6, label='XGBoost Projected Path')
    ax.fill_between(future_indices, xgb_low, xgb_high, color='#ffab00', alpha=0.12, label='XGBoost Envelope')

    # 2h Time-Stop Line
    ax.axvline(x=hist_len + 7, color='#ff5252', linestyle='--', linewidth=2.0, alpha=0.8)
    ax.text(hist_len + 7.25, curr_close - 5.0, '2h Time-Stop (+120m)', color='#ff5252', fontsize=12, fontweight='bold', rotation=90)

    # -------------------------------------------------------------
    # STRUCTURAL POIs & EXECUTION TARGET OVERLAYS
    # -------------------------------------------------------------
    total_bars = hist_len + 16
    
    # PDH & PDL (Previous Day Extremes)
    if not np.isnan(poi_dict['PDH']):
        ax.axhline(y=poi_dict['PDH'], color='#ffd54f', linestyle='--', linewidth=1.4, alpha=0.75)
        ax.text(0.5, poi_dict['PDH'] + 0.6, f"PDH: ${poi_dict['PDH']:.2f}", color='#ffd54f', fontsize=11, fontweight='bold')
    
    if not np.isnan(poi_dict['PDL']):
        ax.axhline(y=poi_dict['PDL'], color='#ffd54f', linestyle='--', linewidth=1.4, alpha=0.75)
        ax.text(0.5, poi_dict['PDL'] - 1.2, f"PDL: ${poi_dict['PDL']:.2f}", color='#ffd54f', fontsize=11, fontweight='bold')

    # Asian Range (ASH / ASL)
    if not np.isnan(poi_dict['ASH']):
        ax.axhline(y=poi_dict['ASH'], color='#80deea', linestyle=':', linewidth=1.2, alpha=0.65)
        ax.text(hist_len * 0.45, poi_dict['ASH'] + 0.6, f"ASH: ${poi_dict['ASH']:.2f}", color='#80deea', fontsize=10.5)

    if not np.isnan(poi_dict['ASL']):
        ax.axhline(y=poi_dict['ASL'], color='#80deea', linestyle=':', linewidth=1.2, alpha=0.65)
        ax.text(hist_len * 0.45, poi_dict['ASL'] - 1.2, f"ASL: ${poi_dict['ASL']:.2f}", color='#80deea', fontsize=10.5)

    # Execution Setup Levels (If Active)
    if poi_dict['Risk'] > 0:
        # Stop Loss
        ax.axhline(y=poi_dict['SL'], color='#ff1744', linestyle='-', linewidth=2.2, alpha=0.95)
        ax.text(total_bars - 4.5, poi_dict['SL'] + 0.6, f"HARD SL: ${poi_dict['SL']:.2f} (Risk: ${poi_dict['Risk']:.2f})", 
                color='#ff1744', fontsize=11.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#2a0000", ec="#ff1744", lw=1.2))

        # Take Profit 1
        ax.axhline(y=poi_dict['TP1'], color='#69f0ae', linestyle='-.', linewidth=2.0, alpha=0.9)
        ax.text(total_bars - 4.5, poi_dict['TP1'] + 0.6, f"TP1 (1h): ${poi_dict['TP1']:.2f} ({poi_dict['RR_TP1']:.1f}:1 R:R)", 
                color='#69f0ae', fontsize=11, fontweight='bold', bbox=dict(boxstyle="round,pad=0.2", fc="#002a11", ec="#69f0ae", lw=1.2))

        # Take Profit 2
        ax.axhline(y=poi_dict['TP2'], color='#00e676', linestyle='-', linewidth=2.4, alpha=0.95)
        ax.text(total_bars - 4.5, poi_dict['TP2'] + 0.6, f"TP2 (POI): ${poi_dict['TP2']:.2f} ({poi_dict['RR_TP2']:.1f}:1 R:R)", 
                color='#00e676', fontsize=12, fontweight='bold', bbox=dict(boxstyle="round,pad=0.25", fc="#003314", ec="#00e676", lw=1.5))

    # Current Price Line
    ax.axhline(y=curr_close, color='#ffffff', linestyle='-.', linewidth=1.4, alpha=0.8, label=f'Entry (${curr_close:.2f})')

    # Annotate Forecast Extremes
    def callout(x, y, text, fc, ec, offset):
        ax.scatter(x, y, color=ec, s=75, zorder=6, edgecolor='#ffffff', linewidth=1.2)
        ax.annotate(text, xy=(x, y), xytext=(x + offset[0], y + offset[1]),
                    bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=1.5),
                    arrowprops=dict(arrowstyle="->", color=ec, lw=1.4),
                    color="#ffffff", fontsize=9.5, fontweight="bold", ha='center')

    callout(future_indices[np.argmax(lgbm_high)], max(lgbm_high), f"LGBM Max: ${max(lgbm_high):.2f}", "#002b33", "#00e5ff", (-1.5, 3.5))
    callout(future_indices[np.argmin(lgbm_low)],  min(lgbm_low),  f"LGBM Min: ${min(lgbm_low):.2f}",  "#002b33", "#00e5ff", (-1.5, -4.0))
    callout(future_indices[np.argmax(xgb_high)],  max(xgb_high),  f"XGB Max: ${max(xgb_high):.2f}",  "#332200", "#ffab00", (1.5, 3.5))
    callout(future_indices[np.argmin(xgb_low)],   min(xgb_low),   f"XGB Min: ${min(xgb_low):.2f}",   "#332200", "#ffab00", (1.5, -4.0))

    # Ticks & Formatting
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

    ax.set_title(f"Multi-Scale POI Sniper Forecast: 16 Past Candles vs. 16 Projected Future Candles (+15m to +240m)", 
                 color='#ffffff', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel("Gold Price ($)", color='#ffffff', fontsize=15, fontweight='bold', labelpad=14)
    ax.set_xlabel("Timeline (Historical HH:MM [GMT+8] on Left | +Minutes Projected on Right)", 
                  color='#b0bec5', fontsize=14, fontweight='bold', labelpad=14)
    ax.grid(True, color='#282828', linestyle='--', linewidth=0.8, alpha=0.75)
    ax.legend(facecolor='#212121', edgecolor='#424242', labelcolor='#e0e0e0', loc='upper left', fontsize=11.5)

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
        try:
            print(text)
        except Exception:
            print(text.encode('ascii', errors='ignore').decode('ascii'))
        return

    print("--> Sending Telegram notification...")
    text_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "HTML"}
    res_text = requests.post(text_url, json=payload, timeout=15)
    
    if res_text.status_code == 200:
        print("--> Text report sent successfully!")
    else:
        print(f"[ERROR] Failed to send text: {res_text.text}")

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
