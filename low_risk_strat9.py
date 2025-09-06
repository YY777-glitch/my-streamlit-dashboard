import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from tradingview_datafeed import TradingViewDatafeed, Interval  # Updated import
from datetime import datetime, timedelta
import time

# --- Config ---
st.set_page_config(layout="wide", page_title="PCA Trading System")
st.title("🔥 Enhanced PCA Trading Dashboard")

# --- Constants ---
CORRELATION_HISTORY_START = datetime(2020, 1, 1)  # Start of correlation history

# --- Helper Functions ---
def calculate_rolling_correlations(pca_df, window=50):
    """Calculate rolling correlations between components with enhanced stability"""
    corr_df = pca_df[['PC1', 'PC2', 'PC3']].rolling(window, min_periods=int(window*0.8)).corr().dropna()
    # Add metadata about the calculation window
    corr_df = corr_df.assign(
        calculation_window=window,
        last_updated=datetime.now()
    )
    return corr_df

def calculate_correlation_regimes(corr_data):
    """Identify historical correlation regimes"""
    regimes = []
    for pair in [('PC1', 'PC2'), ('PC2', 'PC3'), ('PC1', 'PC3')]:
        col = f"{pair,[object Object],}-{pair,[object Object],}"
        rolling_mean = corr_data[col].rolling(90).mean()
        
        # Define regimes
        conditions = [
            (rolling_mean > 0.6),
            (rolling_mean > 0.3),
            (rolling_mean > -0.3),
            (rolling_mean > -0.6),
            (rolling_mean <= -0.6)
        ]
        choices = [
            f"Strong Positive {col}",
            f"Positive {col}",
            f"Neutral {col}",
            f"Negative {col}",
            f"Strong Negative {col}"
        ]
        
        regime = pd.Series(np.select(conditions, choices, default='Other'), index=corr_data.index)
        regimes.append(pd.DataFrame({f"{col}_regime": regime}))
    
    return pd.concat(regimes, axis=1)

# --- Enhanced Data Fetching ---
@st.cache_data(ttl=3600, show_spinner="Fetching live market data...")
def fetch_data(shift_audjpy=True):
    tv = TradingViewDatafeed()  # Updated class name
    assets = {
        'AMEX': ['gld', 'kweb', 'uso', 'bito', 'uup'],
        'NASDAQ': ['qqq', 'ief'],
        'OANDA': ['audjpy'],
        'TVC': ['vix']
    }
    
    max_retries = 3
    retry_delay = 2
    raw_dfs = []
    
    # 1. Fetch all data
    for exchange, symbols in assets.items():
        for symbol in symbols:
            for attempt in range(max_retries):
                try:
                    data = tv.get_hist(
                        symbol=symbol,
                        exchange=exchange,
                        interval=Interval.in_4_hour,
                        n_bars=5000
                    )
                    if not data.empty:
                        df = data[['close']].copy()
                        df.columns = [f"{exchange.lower()}:{symbol}"]
                        raw_dfs.append(df)  # Changed from `raw_dfs.append(df)` to `raw_dfs.append(df)`
                    break
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.error(f"Failed to fetch {exchange}:{symbol} after {max_retries} attempts: {str(e)}")
                        continue
                    time.sleep(retry_delay)
    
    try:
        # 2. Initial merge (keep all data)
        vix_df = next((df for df in raw_dfs if 'tvc:vix' in df.columns), None)
        other_dfs = [df for df in raw_dfs if 'tvc:vix' not in df.columns]
        merged_df = pd.concat(other_dfs, axis=1)
        
        # And that the VIX merge is happening:
        if vix_df is not None:
            merged_df = pd.merge_asof(
                merged_df.sort_index(),
                vix_df.sort_index(),
                left_index=True,
                right_index=True,
                direction='nearest',
                tolerance=pd.Timedelta('2h')
            )

        # Apply transformations and filtering
        if 'amex:uup' in merged_df.columns:
            merged_df['amex:uup'] = 1 / merged_df['amex:uup']
        
        if shift_audjpy and 'oanda:audjpy' in merged_df.columns:
            merged_df['oanda:audjpy'] = merged_df['oanda:audjpy'].shift(1)
            merged_df['tvc:vix'] = merged_df['tvc:vix'].shift(1)

        # Filter to target hours
        def is_target_time(ts):
            hour, minute = ts.hour, ts.minute
            return ((19 <= hour <= 23 and minute <= 55) or  # 21:00-22:30 window
                    (1 <= hour <= 6 and minute <= 55))      # 05:00-06:30 window
        
        target_mask = merged_df.index.map(is_target_time)
        filtered_df = merged_df[target_mask].copy()

        # 8. Final clean-up
        final_df = filtered_df.dropna()
        
        return final_df
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return pd.DataFrame()

# --- Enhanced PCA Analysis ---
@st.cache_data(ttl=1800, show_spinner="Running PCA analysis...")
def rolling_pca(df, window=125):
    # Ensure we're only using numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df = df[numerical_cols]
    
    scaler = StandardScaler()
    results = {
        'Time': [], 'PC1': [], 'PC2': [], 'PC3': [],
        'residual_norm': [], 'sum_abs_residuals': [], 'sum_residuals': [],
        'explained_var': [], 'window_size': window
    }
    residuals_df = pd.DataFrame(columns=df.columns)
    component_loadings = []
    
    for i in range(window, len(df)):
        window_data = df.iloc[i-window:i]
        scaled_data = scaler.fit_transform(window_data)
        
        pca = PCA(n_components=3)
        transformed = pca.fit_transform(scaled_data)
        
        # Enforce sign convention
        bond_col = 'ief' if 'ief' in df.columns else next((c for c in df.columns if 'bond' in c.lower()), None)
        risk_col = 'qqq' if 'qqq' in df.columns else next((c for c in df.columns if 'spy' in c.lower()), None)
        usd_col = 'uup' if 'uup' in df.columns else next((c for c in df.columns if 'dxy' in c.lower()), None)
        
        # Flip PC2 if bonds load negatively
        if bond_col and pca.components_[1, df.columns.get_loc(bond_col)] < 0:
            transformed[:, 1] *= -1
            pca.components_[1, :] *= -1
        
        # Flip PC1 if risk asset loads negatively
        if risk_col and pca.components_[0, df.columns.get_loc(risk_col)] < 0:
            transformed[:, 0] *= -1
            pca.components_[0, :] *= -1
        
        # Flip PC3 if USD loads negatively
        if usd_col and pca.components_[2, df.columns.get_loc(usd_col)] < 0:
            transformed[:, 2] *= -1
            pca.components_[2, :] *= -1

        reconstruction = np.dot(transformed, pca.components_)
        residuals = scaled_data - reconstruction
        
        results['Time'].append(df.index[i])
        results['PC1'].append(transformed[-1, 0])
        results['PC2'].append(transformed[-1, 1])
        results['PC3'].append(transformed[-1, 2])
        results['residual_norm'].append(np.linalg.norm(residuals[-1]))
        results['sum_abs_residuals'].append(np.sum(np.abs(residuals[-1])))
        results['sum_residuals'].append(np.sum(residuals[-1]))
        results['explained_var'].append(pca.explained_variance_ratio_)
        residuals_df.loc[df.index[i]] = residuals[-1]
        component_loadings.append(pca.components_)
    
    pca_df = pd.DataFrame(results).set_index('Time')
    
    # Add metadata and diagnostics
    pca_df = pca_df.assign(
        analysis_date=datetime.now(),
        pca_window=window,
        pc1_90th=np.percentile(pca_df['PC1'].rolling(90).mean(), 90),
        pc2_90th=np.percentile(pca_df['PC2'].rolling(90).mean(), 90),
        residual_90th=np.percentile(pca_df['residual_norm'].rolling(90).mean(), 90)
    )
    
    # Store component loadings history
    loadings_history = pd.DataFrame(
        np.array(component_loadings).reshape(len(component_loadings), -1),
        index=pca_df.index,
        columns=[f"loading_{i}_{j}" for i in range(3) for j in range(len(df.columns))]
    )
    
    return pca_df, residuals_df, loadings_history

# ... (rest of your code remains unchanged) ...

if __name__ == "__main__":
    main()