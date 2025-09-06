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
    regimes = {}
    
    # Define pairs of components to analyze
    pairs = [('PC1', 'PC2'), ('PC2', 'PC3'), ('PC1', 'PC3')]
    
    for pc1, pc2 in pairs:
        col = f"{pc1}-{pc2}"
        rolling_mean = corr_data[col].rolling(90).mean()
        
        # Define regimes with clearer logic
        regime_conditions = [
            (rolling_mean > 0.6, f"Strong Positive {col}"),
            (rolling_mean > 0.3, f"Positive {col}"),
            (rolling_mean > -0.3, f"Neutral {col}"),
            (rolling_mean > -0.6, f"Negative {col}"),
            (rolling_mean <= -0.6, f"Strong Negative {col}")
        ]
        
        # Build regime series
        regime = pd.Series('Other', index=corr_data.index)
        for condition, label in regime_conditions:
            regime = regime.where(~condition, label)
        
        regimes[f"{col}_regime"] = regime
    
    return pd.DataFrame(regimes)

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
                        raw_dfs.append(df)  # Fixed: removed duplicate line
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
        bond_col = 'amex:ief' if 'amex:ief' in df.columns else next((c for c in df.columns if 'bond' in c.lower() or 'ief' in c), None)
        risk_col = 'nasdaq:qqq' if 'nasdaq:qqq' in df.columns else next((c for c in df.columns if 'qqq' in c or 'spy' in c.lower()), None)
        usd_col = 'amex:uup' if 'amex:uup' in df.columns else next((c for c in df.columns if 'uup' in c or 'dxy' in c.lower()), None)
        
        # Flip PC2 if bonds load negatively
        if bond_col and bond_col in df.columns and pca.components_[1, df.columns.get_loc(bond_col)] < 0:
            transformed[:, 1] *= -1
            pca.components_[1, :] *= -1
        
        # Flip PC1 if risk asset loads negatively
        if risk_col and risk_col in df.columns and pca.components_[0, df.columns.get_loc(risk_col)] < 0:
            transformed[:, 0] *= -1
            pca.components_[0, :] *= -1
        
        # Flip PC3 if USD loads negatively
        if usd_col and usd_col in df.columns and pca.components_[2, df.columns.get_loc(usd_col)] < 0:
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
        columns=[f"loading_pc{pc+1}_{col}" for pc in range(3) for col in df.columns]
    )
    
    return pca_df, residuals_df, loadings_history

def check_pca_stability(pca_df, lookback=20):
    """Calculate stability metrics for PCA components using percentile-based approach"""
    # Calculate absolute daily changes
    pc1_changes = pca_df['PC1'].diff().abs().dropna()
    pc2_changes = pca_df['PC2'].diff().abs().dropna()
    
    # Calculate stability percentiles (lower change = more stable)
    pc1_percentile = np.mean(pc1_changes.rolling(lookback).mean() < pc1_changes.rolling(lookback).mean().iloc[-1]) * 0.8
    pc2_percentile = np.mean(pc2_changes.rolling(lookback).mean() < pc2_changes.rolling(lookback).mean().iloc[-1]) * 0.8
    
    # Convert to stability scores (higher = more stable)
    # pc1_stability = 1 - pc1_percentile
    # pc2_stability = 1 - pc2_percentile
    
    stats = {
        'PC1': {
            'day_to_day_chg': pc1_changes.mean(),
            'weekly_vol': pc1_changes.rolling(5).mean().iloc[-1],
            'stability_score': min(1.0, 1 - pc1_percentile + 0.2),  # Add buffer
            'percentile': pc1_percentile,
            'current_avg_change': pc1_changes.rolling(lookback).mean().iloc[-1]
        },
        'PC2': {
            'day_to_day_chg': pc2_changes.mean(),
            'weekly_vol': pc2_changes.rolling(5).mean().iloc[-1],
            'stability_score': min(1.0, 1 - pc2_percentile + 0.2),
            'percentile': pc2_percentile,
            'current_avg_change': pc2_changes.rolling(lookback).mean().iloc[-1]
        }
    }
    return stats

def classify_pca_move(current, previous):
    """Classify the magnitude of PCA moves"""
    pc1_chg = abs(current['PC1'] - previous['PC1'])
    pc2_chg = abs(current['PC2'] - previous['PC2'])
    
    move_size = {
        'PC1': 'Small' if pc1_chg < 0.15 else 
               'Medium' if pc1_chg < 0.30 else
               'Big' if pc1_chg < 0.50 else 'Extreme',
        'PC2': 'Small' if pc2_chg < 0.20 else 
               'Medium' if pc2_chg < 0.40 else
               'Big' if pc2_chg < 0.60 else 'Extreme'
    }
    return move_size


def generate_signals(df, pca_df, residuals_df, z_threshold=1.5):
    """Generate trading signals with pair trading thresholds at half of normal z_threshold"""
    signals = []
    
    # ===== Helper Functions =====
    def detect_regimes(pca_values, vix):
        """Classify current market regime with enforced sign convention"""
        return {
            'risk_on': pca_values['PC1'] > 0.5,
            'risk_off': pca_values['PC1'] < -0.5,
            'high_vol': vix > 20 if vix else False,
            'pca_extreme': (abs(pca_values['PC1']) > 1.5) or (abs(pca_values['PC2']) > 1.5),
            'bonds_rising': pca_values['PC2'] > 0.3,
            'bonds_falling': pca_values['PC2'] < -0.3,
            'usd_strong': 'PC3' in pca_values and pca_values['PC3'] > 0.3,
            'usd_weak': 'PC3' in pca_values and pca_values['PC3'] < -0.3
        }

    def get_z_score(series, window=20, current_price=None):
        """Calculate z-score of current price relative to rolling window"""
        if len(series) < window:
            return 0
        rolling = series.rolling(window)
        mean = rolling.mean().iloc[-1]
        std = rolling.std().iloc[-1]
        current = current_price if current_price is not None else series.iloc[-1]
        return (current - mean) / std if std > 0 else 0

    # ===== Data Validation =====
    try:
        available_assets = set(df.columns) & {'qqq', 'ief', 'gld', 'uso', 'bito', 'kweb', 'audjpy', 'vix', 'uup'}
        numeric_df = df[list(available_assets)].apply(pd.to_numeric, errors='coerce').ffill().bfill()
        current = numeric_df.iloc[-1]
        pca_current = pca_df.iloc[-1]
        
        regimes = detect_regimes(pca_current, current.get('vix', 16))
        pca_stability = check_pca_stability(pca_df) if 'check_pca_stability' in globals() else {
            'PC1': {'stability_score': 1.0}, 
            'PC2': {'stability_score': 1.0}
        }
        
    except Exception as e:
        return [{
            'strategy': 'Data Error',
            'ticker': 'ERROR',
            'action': 'NO ACTION',
            'size': 0,
            'condition': f"Data validation failed: {str(e)}",
            'stability_req': "None"
        }]

    # ===== Signal Generation =====
    def add_signal(strategy, ticker, action, base_size, condition, stability_req, z_score=None):
        """Helper function to consistently add signals with risk adjustments"""
        size = base_size
        
        # Apply z-score scaling if provided
        if z_score is not None:
            if action == 'LONG' and z_score > 0:
                size *= max(0.1, 1 - (abs(z_score)/z_threshold))
            elif action == 'SHORT' and z_score < 0:
                size *= max(0.1, 1 - (abs(z_score)/z_threshold))
        
        # Apply stability scaling
        if stability_req != "None":
            stability_score = min(
                pca_stability['PC1']['stability_score'],
                pca_stability['PC2']['stability_score']
            )
            size *= stability_score
        
        # Volatility scaling
        if 'vix' in current and current['vix'] > 22:
            size *= max(0.3, 22/current['vix'])
            
        # Final size constraints
        size = max(0.1, min(1.0, size))
        
        signals.append({
            'strategy': strategy,
            'ticker': ticker,
            'action': action,
            'size': round(size, 2),
            'condition': condition,
            'stability_req': stability_req,
            'z_score': z_score
        })

    # 1. Risk-On Momentum (QQQ)
    if 'qqq' in available_assets:
        qqq_z = get_z_score(df['qqq'], current_price=current['qqq'])
        if (pca_stability['PC1']['stability_score'] > 0.3 and
            regimes['risk_on'] and 
            current['qqq'] > df['qqq'].rolling(20).mean().iloc[-1] and
            qqq_z < z_threshold):
            
            add_signal(
                strategy='Risk-On Momentum',
                ticker='QQQ',
                action='LONG',
                base_size=1.0,
                condition=f"PC1={pca_current['PC1']:.2f}>0.5, QQQ>MA20, Z={qqq_z:.2f}<{z_threshold}",
                stability_req="PC1>30%",
                z_score=qqq_z
            )

    # 2. Risk-Off Hedge (IEF)
    if 'ief' in available_assets:
        ief_z = get_z_score(df['ief'], current_price=current['ief'])
        if (pca_stability['PC2']['stability_score'] > 0.25 and
            regimes['risk_off'] and 
            regimes.get('high_vol', False) and
            ief_z < z_threshold):
            
            add_signal(
                strategy='Risk-Off Hedge',
                ticker='IEF',
                action='LONG',
                base_size=0.8,
                condition=f"PC1={pca_current['PC1']:.2f}<-0.5, VIX>20, Z={ief_z:.2f}<{z_threshold}",
                stability_req="PC2>25%",
                z_score=ief_z
            )

    # 3. Commodity Rotation (USO/GLD) - Using HALF z_threshold for pairs
    if all(a in available_assets for a in ['gld','uso']) and regimes['bonds_falling']:
        uso_z = get_z_score(df['uso'], current_price=current['uso'])
        gld_z = get_z_score(df['gld'], current_price=current['gld'])
        pair_threshold = z_threshold / 2  # Special threshold for pairs
        
        if uso_z < pair_threshold and gld_z > -pair_threshold:
            add_signal(
                strategy='Commodity Rotation',
                ticker='USO/GLD',
                action='PAIR_LONG_USO_SHORT_GLD',
                base_size=0.7,
                condition=f"PC2={pca_current['PC2']:.2f}<-0.3, USO_Z={uso_z:.2f}<{pair_threshold:.1f}, GLD_Z={gld_z:.2f}>-{pair_threshold:.1f}",
                stability_req="PC2<-0.3"
            )

    # 4. Residual Mean-Reversion
    for asset in set(residuals_df.columns) & available_assets:
        resid_z = (residuals_df[asset].iloc[-1] - residuals_df[asset].mean()) / residuals_df[asset].std()
        if abs(resid_z) > z_threshold:
            add_signal(
                strategy='Residual MR',
                ticker=asset.upper(),
                action='SHORT' if resid_z > 0 else 'LONG',
                base_size=min(0.5, abs(resid_z)/3),
                condition=f"Z={resid_z:.2f}",
                stability_req="None"
            )

    # 5. USD Carry (AUDJPY)
    if ('audjpy' in available_assets and 
        regimes.get('usd_weak', False) and
        'PC3' in pca_df.columns):
        
        audjpy_z = get_z_score(df['audjpy'], current_price=current['audjpy'])
        if audjpy_z < z_threshold:
            add_signal(
                strategy='USD Carry',
                ticker='AUDJPY',
                action='LONG',
                base_size=0.6,
                condition=f"PC3={pca_current['PC3']:.2f}<-0.3, Z={audjpy_z:.2f}<{z_threshold}",
                stability_req="PC3 exists",
                z_score=audjpy_z
            )

    # 6. Volatility Management
    if regimes.get('high_vol', False) and 'vix' in available_assets:
        add_signal(
            strategy='Volatility Management',
            ticker='ALL',
            action='REDUCE_EXPOSURE_30%',
            base_size=0.3,
            condition=f"VIX={current.get('vix', 'N/A')}>20",
            stability_req="None"
        )

    # 7. Crypto Hedge (BITO)
    if 'bito' in available_assets:
        bito_z = get_z_score(df['bito'], current_price=current['bito'])
        if (pca_stability['PC1']['stability_score'] > 0.25 and
            regimes['risk_on'] and
            current['bito'] < df['bito'].rolling(20).mean().iloc[-1] and
            bito_z > -z_threshold):
            
            add_signal(
                strategy='Crypto Hedge',
                ticker='BITO',
                action='LONG',
                base_size=0.4,
                condition=f"PC1={pca_current['PC1']:.2f}>0, BITO<MA20, Z={bito_z:.2f}>-{z_threshold}",
                stability_req="PC1>25%",
                z_score=bito_z
            )

    # 8. China Rotation (KWEB)
    if 'kweb' in available_assets:
        kweb_z = get_z_score(df['kweb'], current_price=current['kweb'])
        if (regimes['risk_on'] and
            pca_current['PC2'] < 0 and
            kweb_z < z_threshold):
            
            add_signal(
                strategy='China Rotation',
                ticker='KWEB',
                action='LONG',
                base_size=0.5,
                condition=f"PC1>0, PC2={pca_current['PC2']:.2f}<0, Z={kweb_z:.2f}<{z_threshold}",
                stability_req="None",
                z_score=kweb_z
            )

    # 9. Gold Hedge (GLD)
    if 'gld' in available_assets:
        gld_z = get_z_score(df['gld'], current_price=current['gld'])
        if (regimes['risk_off'] and 
            pca_current['PC2'] > 0.3 and
            gld_z < z_threshold):
            
            add_signal(
                strategy='Gold Hedge',
                ticker='GLD',
                action='LONG',
                base_size=0.6,
                condition=f"PC1<-0.5, PC2={pca_current['PC2']:.2f}>0.3, Z={gld_z:.2f}<{z_threshold}",
                stability_req="PC2>0.3",
                z_score=gld_z
            )

    # 10. Extreme Reversion
    if regimes['pca_extreme']:
        add_signal(
            strategy='Extreme Reversion',
            ticker='ALL',
            action='PREPARE_REVERSION',
            base_size=0.2,
            condition=f"PC1={pca_current['PC1']:.2f} or PC2={pca_current['PC2']:.2f} extremes",
            stability_req="None"
        )

    # 11. QQQ-IEF Negative Correlation Pair - Using HALF z_threshold
    if ('qqq' in available_assets and 'ief' in available_assets):
        qqq_ief_corr = df['qqq'].rolling(20).corr(df['ief']).iloc[-1]
        if qqq_ief_corr < -0.3:
            qqq_z = get_z_score(df['qqq'], current_price=current['qqq'])
            ief_z = get_z_score(df['ief'], current_price=current['ief'])
            pair_threshold = z_threshold / 2
            
            if qqq_z < pair_threshold and ief_z > -pair_threshold:
                add_signal(
                    strategy='Equity-Bond Divergence',
                    ticker='QQQ/IEF',
                    action='PAIR_LONG_QQQ_SHORT_IEF',
                    base_size=0.6,
                    condition=f"QQQ-IEF corr={qqq_ief_corr:.2f}<-0.3, QQQ_Z={qqq_z:.2f}<{pair_threshold:.1f}, IEF_Z={ief_z:.2f}>-{pair_threshold:.1f}",
                    stability_req="None"
                )

    # 12. USD-Gold Divergence - Using HALF z_threshold
    if ('uup' in available_assets and 'gld' in available_assets):
        usd_gld_corr = df['uup'].rolling(20).corr(df['gld']).iloc[-1]
        if usd_gld_corr < -0.4:
            uup_z = get_z_score(df['uup'], current_price=current['uup'])
            gld_z = get_z_score(df['gld'], current_price=current['gld'])
            pair_threshold = z_threshold / 2
            
            if uup_z < pair_threshold and gld_z > -pair_threshold:
                add_signal(
                    strategy='USD-Gold Divergence',
                    ticker='GLD/UUP',
                    action='PAIR_LONG_GLD_SHORT_UUP',
                    base_size=0.5,
                    condition=f"USD-Gold corr={usd_gld_corr:.2f}<-0.4, UUP_Z={uup_z:.2f}<{pair_threshold:.1f}, GLD_Z={gld_z:.2f}>-{pair_threshold:.1f}",
                    stability_req="None"
                )

    if not signals:
        signals.append({
            'strategy': 'No Signals',
            'ticker': 'DEBUG',
            'action': 'MONITOR',
            'size': 0,
            'condition': "No conditions met with current thresholds",
            'stability_req': "None"
        })
    
    return signals

# --- Enhanced Dashboard Layout ---
def main():
    # --- Data Loading ---
    with st.spinner("Loading market data..."):
        df = fetch_data()
        if df.empty:
            st.error("Failed to load data. Please check your internet connection and try again.")
            return
    # --- Helper: Parse action string ---
    def parse_trade_action(action: str):
        try:
            parts = action.strip().split()
            if len(parts) == 2:
                return parts[0].upper(), parts[1].upper()
        except Exception:
            pass
        return "Unknown", "Unknown"
    
    # --- Helper: Trade execution suggestions ---
    execution_lookup = {
        "QQQ": "Buy QQQ ETF or NASDAQ futures/options",
        "GLD": "Buy GLD ETF or gold futures/options",
        "IEF": "Buy IEF or Treasury futures",
        "TLT": "Buy TLT or long bond futures",
        "UUP": "Buy UUP ETF or USD futures",
        "CPER": "Buy CPER ETF or copper futures",
        "DBA": "Buy DBA ETF or ag futures",
        "USO": "Buy USO or crude oil futures",
        "KWEB": "Buy KWEB ETF or China Tech exposure",
        "BITO": "Buy BITO or BTC futures",
        "AUDJPY": "Buy AUD/JPY spot or futures"
    }
    # --- Sidebar ---
    st.sidebar.header("⚙️ Parameters")
    window = st.sidebar.slider("PCA Window", 30, 300, 125)
    z_threshold = st.sidebar.slider("Z-Score Threshold", 1.5, 3.0, 1.5, 0.1)
    residual_alert_threshold = st.sidebar.slider("Residual Alert Threshold", 1.0, 3.0, 1.5, 0.1)
    
    # Add correlation history date range selector
    min_date = CORRELATION_HISTORY_START
    max_date = datetime.now()
    start_date = st.sidebar.date_input(
        "Correlation History Start",
        value=min_date,
        min_value=min_date,
        max_value=max_date
    )
    # Usage in your main function:
  
    # --- PCA Analysis ---
    with st.spinner("Running PCA analysis..."):
        pca_df, residuals_df, loadings_history = rolling_pca(df, window)
        pca_stability = check_pca_stability(pca_df)
    
    # --- Enhanced Signals Section ---
    st.subheader("🚦 Actionable Trading Signals")
    
    # Generate all possible signals (including those that might need filtering)
    all_signals = generate_signals(df, pca_df, residuals_df, z_threshold)
    
    if not all_signals:
        st.warning("No trading signals generated with current market conditions")
    else:
        # Categorize signals by type
        categorized_signals = {
            'Trend Following': [],
            'Mean Reversion': [],
            'Hedge Strategies': [],
            'Special Situations': []
        }
        
        for signal in all_signals:
            # Skip if signal is not a dictionary (debug messages etc)
            if not isinstance(signal, dict):
                continue
                
            st.write("Processing signal:", signal)  # Debug output
            
            # Categorize signals based on strategy name
            strategy_name = signal.get('strategy', '')
            
            if 'Momentum' in strategy_name or 'Trend' in strategy_name:
                categorized_signals['Trend Following'].append(signal)
            elif 'MR' in strategy_name or 'Reversion' in strategy_name:
                categorized_signals['Mean Reversion'].append(signal)
            elif 'Hedge' in strategy_name or 'Defensive' in strategy_name:
                categorized_signals['Hedge Strategies'].append(signal)
            else:
                categorized_signals['Special Situations'].append(signal)
        
        # Display each category in its own expandable section
        for category, signals in categorized_signals.items():
            if signals:
                with st.expander(f"{category} ({len(signals)} signals)", expanded=True):
                    for signal in signals:
                        # Create columns for better layout
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            # Determine position size based on signal strength
                            size = signal.get('size', 0)
                            if size > 0.7:
                                size_label = "Large (3-5%)"
                                color = "green"
                            elif size > 0.4:
                                size_label = "Medium (1-3%)"
                                color = "blue"
                            else:
                                size_label = "Small (0.5-1%)"
                                color = "orange"
                            
                            st.markdown(f"""
                            <div style='
                                border-left: 4px solid {color};
                                padding: 8px;
                                background-color: #f8f9fa;
                                margin-bottom: 10px;
                            '>
                            <h4>{signal.get('action', '')}</h4>
                            <p><b>Size:</b> {size_label}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            # Enhanced reasoning section
                            st.markdown(f"""
                            **{signal.get('strategy', '')}**  
                            *Rationale*:  
                            {signal.get('condition', '')}  
                            
                            *Current Market Conditions*:  
                            - PC1 (Risk): {pca_df['PC1'].iloc[-1]:.2f}  
                            - PC2 (Rates): {pca_df['PC2'].iloc[-1]:.2f}  
                            - VIX: {df.get('vix', ['N/A'])[-1]}  
                            """)
                            
                            # Parse and suggest trade execution
                            ticker = signal.get('ticker', '').upper()
                            if ticker in execution_lookup:
                                st.info(f"**Trade Execution**: {execution_lookup[ticker]}")
                            else:
                                st.warning("**Action Plan**: Signal detected but trade execution unclear - review manually")

    # Visual stability indicator
    stability_score = (pca_stability['PC1']['stability_score'] + 
                      pca_stability['PC2']['stability_score'])/2
    
    # Enhanced stability description
    stability_text = (
        f"Regime Stability: {stability_score:.0%}\n"
        f"PC1: {pca_stability['PC1']['current_avg_change']:.2f}/day "
        f"(better than {pca_stability['PC1']['percentile']:.0%} history)\n"
        f"PC2: {pca_stability['PC2']['current_avg_change']:.2f}/day "
        f"(better than {pca_stability['PC2']['percentile']:.0%} history)"
    )
    
    st.progress(
        stability_score,
        text=stability_text
    )
    
    # Full metrics table (collapsible)
    with st.expander("Detailed PCA Stability Metrics", expanded=False):
        st.write(pd.DataFrame(pca_stability).T)
        st.caption("""
        - **day_to_day_chg**: Avg absolute daily change
        - **weekly_vol**: Avg 5-day change  
        - **stability_score**: 1 = Perfect stability
        """)
        
    # In the section where you create the 'current' dictionary (around line 785):
    current = {
        'PC1': pca_df['PC1'].iloc[-1],
        'PC2': pca_df['PC2'].iloc[-1],
        'PC3': pca_df['PC3'].iloc[-1],
        'residual_norm': pca_df['residual_norm'].iloc[-1],
        'sum_abs': pca_df['sum_abs_residuals'].iloc[-1],
        '90th_percentile': np.percentile(pca_df['sum_abs_residuals'].rolling(100).mean(), 90),
        'residual_status': "Normal",
        'vix': df['vix'].iloc[-1] if 'vix' in df.columns else np.nan  # Safely add VIX if available
    }

    # Enhanced residual status
    if current['residual_norm'] > residual_alert_threshold * current['90th_percentile']:
        current['residual_status'] = "ALERT: Elevated Residuals"
    elif current['residual_norm'] > 2 * residual_alert_threshold * current['90th_percentile']:
        current['residual_status'] = "WARNING: Extreme Residuals"
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("PC1 (Risk)", f"{current['PC1']:.2f}",
                "Risk-On" if current['PC1'] > 0.5 else "Risk-Off")
    with col2:
        st.metric("PC2 (Bond Price)", f"{current['PC2']:.2f}",
                "Bond Rising" if current['PC2'] > 0.5 else "Bond Falling")
    with col3:
        st.metric("PC3 (USD)", f"{current['PC3']:.2f}",
                "USD Strong" if current['PC3'] > 0 else "USD Weak")
    with col4:
        st.metric("Residual Status", current['residual_status'],
                f"Norm: {current['residual_norm']:.2f} (90%: {current['90th_percentile']:.2f})")
    


    # --- Market Conditions Summary ---
    with st.expander("📊 Current Market Conditions Summary", expanded=True):
        cols = st.columns(3)
        with cols[0]:
            st.metric("Risk Regime (PC1)", 
                    f"{pca_df['PC1'].iloc[-1]:.2f}", 
                    "Risk-On" if pca_df['PC1'].iloc[-1] > 0.5 else "Risk-Off")
            st.write(f"20-day MA: {pca_df['PC1'].rolling(20).mean().iloc[-1]:.2f}")
        
        with cols[1]:
            st.metric("Rates Regime (PC2)", 
                    f"{pca_df['PC2'].iloc[-1]:.2f}", 
                    "Bond Price Rising" if pca_df['PC2'].iloc[-1] > 0.5 else "Bond Price Falling")
            st.write(f"20-day MA: {pca_df['PC2'].rolling(20).mean().iloc[-1]:.2f}")
        
        with cols[2]:
            vix_status = "High" if current.get('vix', 0) > 25 else "Normal" if current.get('vix', 0) > 15 else "Low"
            st.metric("Market Volatility (VIX)", 
                    f"{current.get('vix', 'N/A')}", 
                    vix_status)
            if 'vix' in current:
                st.write(f"20-day MA: {df['vix'].rolling(20).mean().iloc[-1]:.1f}")

    # --- Key Asset Levels ---
    st.subheader("📈 Key Asset Levels")
    assets_to_display = ['qqq', 'ief', 'gld', 'uso', 'audjpy']
    asset_cols = st.columns(len(assets_to_display))
    
    for i, asset in enumerate(assets_to_display):
        if asset in df.columns:
            with asset_cols[i]:
                current_price = df[asset].iloc[-1]
                ma20 = df[asset].rolling(20).mean().iloc[-1]
                ma50 = df[asset].rolling(50).mean().iloc[-1]
                
                st.metric(asset.upper(), 
                         f"{current_price:.2f}",
                         f"{'Above' if current_price > ma20 else 'Below'} 20MA")
                st.progress(min(1.0, abs(current_price - ma20)/ma20))
                st.caption(f"20MA: {ma20:.2f} | 50MA: {ma50:.2f}")

    
    # --- Diagnostics ---
    st.subheader("📊 PCA Diagnostics")
    tab1, tab2, tab3 = st.tabs(["Components & Correlations", "Residuals Analysis", "Asset Breakdown"])
    
    with tab1:
        # Enhanced Regime Classification
        current_pc1 = pca_df['PC1'].iloc[-1]
        current_pc2 = pca_df['PC2'].iloc[-1]
        regime = (
            "Extreme Risk-On" if current_pc1 > 1.5 else
            "Risk-On" if current_pc1 > 0.5 else
            "Neutral" if current_pc1 > -0.5 else
            "Risk-Off" if current_pc1 > -1.5 else
            "Extreme Risk-Off"
        )
        
        # Add rates regime
        rates_regime = (
            "Bonds Rally - Flight to Safety" if current_pc2 > 1.0 else
            "Bonds Rising" if current_pc2 > 0.3 else
            "Neutral" if current_pc2 > -0.3 else
            "Bonds Selling Off" if current_pc2 > -1.0 else
            "Rates Crash"
        )
        
        st.markdown(f"""
        <div style='
            border-radius: 5px;
            padding: 10px;
            background-color: {"#4CAF50" if current_pc1 > 0.5 else 
                              "#FF9800" if current_pc1 > -0.5 else 
                              "#F44336"};
            color: white;
            margin-bottom: 10px;
        '>
        <h4>Current Regime: {regime} (PC1 = {current_pc1:.2f}) | {rates_regime} (PC2 = {current_pc2:.2f})</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Components Chart with enhanced annotations
        fig = go.Figure()
        colors = ['#FF5722', '#2196F3', '#9C27B0']
        
        for i, col in enumerate(['PC1', 'PC2', 'PC3']):
            fig.add_trace(go.Scatter(
                x=pca_df.index,
                y=pca_df[col],
                name=col,
                line=dict(color=colors[i], width=2),
                hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>"
            ))
        
        # Add regime zones with enhanced descriptions
        fig.add_hrect(x0="min", x1="max", y0=0.5, y1=2.0, fillcolor="green", opacity=0.1,
                      annotation_text="Risk-On Zone: Favor equities, commodities")
        
        fig.add_hrect(x0="min", x1="max", y0=-2.0, y1=-0.5, fillcolor="red", opacity=0.1,
                      annotation_text="Risk-Off Zone: Favor bonds, gold")
        
        fig.add_hrect(x0="min", x1="max", y0=0.3, y1=1.0, fillcolor="blue", opacity=0.05, 
                      annotation_text="Flight-to-Safety: Bonds up")
        
        fig.add_hrect(x0="min", x1="max", y0=-1.0, y1=-0.3, fillcolor="purple", opacity=0.05, 
                      annotation_text="Risk-On in Bonds: Bonds down")

        
        # Add significant events
        significant_moves = pca_df[(abs(pca_df['PC1'].diff()) > 0.5) | 
                                  (abs(pca_df['PC2'].diff()) > 0.5)].index
        
        for event in significant_moves:
            # Convert pandas Timestamp to datetime if needed
            event_dt = event.to_pydatetime() if hasattr(event, 'to_pydatetime') else event
            
            # Add vertical line with proper datetime handling
            fig.add_vline(
                x=event_dt.timestamp() * 1000,  # Convert to milliseconds since epoch
                line_width=1,
                line_dash="dash",
                line_color="yellow",
                opacity=0.5,
                annotation_text=f"Regime Shift {event_dt.strftime('%Y-%m-%d')}"
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Enhanced Correlation Analysis
        st.subheader("Component Relationships")
        component_corr = pca_df[['PC1', 'PC2', 'PC3']].corr()
        
        # Rolling Correlations with history preservation
        st.subheader("🔄 Correlation History (Preserved Since 2020)")
        rolling_corrs = calculate_rolling_correlations(pca_df)
        
        # Calculate correlation pairs
        corr_pairs = pd.DataFrame({
            'PC1-PC2': pca_df['PC1'].rolling(90).corr(pca_df['PC2']),
            'PC2-PC3': pca_df['PC2'].rolling(90).corr(pca_df['PC3']),
            'PC1-PC3': pca_df['PC1'].rolling(90).corr(pca_df['PC3'])
        }).dropna()
        
        # Add correlation regimes
        corr_regimes = calculate_correlation_regimes(corr_pairs)
        corr_pairs = corr_pairs.join(corr_regimes)
        
        # Plot correlation history
        fig_corr = go.Figure()
        pairs = [
            ('PC1-PC2', '#FF5722', 'Risk vs Bond Px'),
            ('PC2-PC3', '#2196F3', 'Bond Px vs USD'), 
            ('PC1-PC3', '#9C27B0', 'Risk vs USD')
        ]
        
        for col, color, name in pairs:
            fig_corr.add_trace(go.Scatter(
                x=corr_pairs.index,
                y=corr_pairs[col],
                name=name,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{name}</b>: %{{y:.2f}}<br>Regime: %{{text}}",
                text=corr_pairs[f"{col}_regime"]
            ))
        
        # Add correlation thresholds with enhanced explanations
        fig_corr.add_hline(y=0.7, line_dash="dot", line_color="green",
                          annotation_text="Strong Positive: Assets moving together")
        fig_corr.add_hline(y=0.3, line_dash="dot", line_color="lightgreen",
                          annotation_text="Positive: Some co-movement")
        fig_corr.add_hline(y=-0.3, line_dash="dot", line_color="lightcoral",
                          annotation_text="Negative: Some divergence")
        fig_corr.add_hline(y=-0.7, line_dash="dot", line_color="red",
                          annotation_text="Strong Negative: Assets moving opposite")
        
        fig_corr.update_layout(
            yaxis_range=[-1,1], 
            hovermode="x unified",
            title="90-Day Rolling Correlations Between Principal Components"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("🚦 Current Correlation Signals")
        current_corrs = {
            'PC1-PC2': corr_pairs['PC1-PC2'].iloc[-1],
            'PC2-PC3': corr_pairs['PC2-PC3'].iloc[-1],
            'PC1-PC3': corr_pairs['PC1-PC3'].iloc[-1]
        }
        
        # Define alert thresholds and messages
        correlation_alerts = {
            'PC1-PC2': [
                (0.7, "success", "STRONG RISK/RATES LINK",
                 "Risk and Rates moving together strongly - Trade directional strategies"),
                (0.3, "info", "Risk/Rates aligned",
                 "Moderate positive correlation - Fade extreme moves"),
                (-0.4, "warning", "Risk/Rates divergent",
                 "Negative correlation - Hedge positions accordingly"),
                (-0.7, "error", "STRONG DIVERGENCE",
                 "Strong negative correlation - Pairs trading favored")
            ],
            'PC2-PC3': [
                (0.7, "success", "VERY STRONG RATES-USD LINK",
                 "Strong relationship between Rates and USD - Trade currencies"),
                (0.5, "info", "Strong Rates-USD link",
                 "Moderate positive correlation - Currency strategies favored"),
                (-0.25, "warning", "Unusual divergence",
                 "Negative correlation - Caution needed in currency positions"),
                (-0.5, "error", "EXTREME DIVERGENCE",
                 "Strong negative correlation - Potential regime shift")
            ],
            'PC1-PC3': [
                (0.4, "success", "USD strengthens with risk",
                 "Positive correlation - Check commodity exposures"),
                (0.25, "info", "USD/risk co-movement",
                 "Moderate positive correlation - Adjust hedges"),
                (-0.2, "warning", "USD weakens as risk rises",
                 "Negative correlation - Favor EM assets"),
                (-0.35, "error", "STRONG RISK-USD INVERSE",
                 "Strong negative correlation - Re-evaluate hedges")
            ]
        }
        
        # Display alerts for each pair
        for pair, value in current_corrs.items():
            for threshold, alert_type, title, message in correlation_alerts[pair]:
                if (threshold > 0 and value > threshold) or (threshold < 0 and value < threshold):
                    # Determine comparison direction
                    direction = ">" if threshold > 0 else "<"
                    
                    # Get the appropriate Streamlit alert method
                    alert_method = {
                        "success": st.success,
                        "info": st.info,
                        "warning": st.warning,
                        "error": st.error
                    }[alert_type]
                    
                    # Display the alert
                    with st.expander(f"{title} ({pair} {direction} {abs(threshold):.1f})", expanded=True):
                        alert_method(f"""
                        **Current {pair} Correlation: {value:.2f}**  
                        **Interpretation:**  
                        {message}  
                        
                        **Current Regime:** {corr_pairs[f'{pair}_regime'].iloc[-1]}  
                        **Historical Percentile:** {np.mean(corr_pairs[pair] < value)*100:.1f}%  
                        (Higher means more extreme correlation)  
                        
                        **Suggested Actions:**  
                        - {message.split(' - ')[-1]}  
                        - Monitor for confirmation of trend  
                        """)
                    break  # Only show the most significant alert
                    
        # NEW: NSE Alert for combined conditions
        if (current_corrs['PC1-PC2'] > 0.2 and 
            current_corrs['PC2-PC3'] < -0.2 and 
            current_corrs['PC1-PC3'] < -0.2):
            
            with st.expander("🚨 NSE Alert: Unusual Correlation Pattern", expanded=True):
                st.error(f"""
                **Non-Standard Event Detected**  
                Current Correlations:  
                - PC1-PC2: {current_corrs['PC1-PC2']:.2f} (Risk/Rates)  
                - PC2-PC3: {current_corrs['PC2-PC3']:.2f} (Rates/USD)  
                - PC1-PC3: {current_corrs['PC1-PC3']:.2f} (Risk/USD)  
                
                **Interpretation:**  
                - Risk and Rates moving together (PC1-PC2 > 0.2)  
                - Rates and USD moving opposite (PC2-PC3 < -0.2)  
                - Risk and USD moving opposite (PC1-PC3 < -0.2)  
                
                **Market Implications:**  
                - Potential breakdown of normal market relationships  
                - USD weakening despite risk-on environment  
                - Bonds showing unusual behavior relative to risk assets  
                
                **Recommended Actions:**  
                ✅ Reduce position sizes  
                ✅ Increase hedging activities  
                🔄 Focus on mean-reversion strategies  
                📊 Monitor for confirmation of new regime  
                """)
        
        # Correlation Heatmap with Enhanced Annotations
        st.subheader("📌 Relationship Matrix with Actionable Insights")
        
        annotation_rules = {
            ('PC1','PC2'): {
                '>0.7': "STRONG RISK/RATES LINK - Trade directional strategies",
                '>0.3': "Risk/Rates aligned - Fade extremes",
                '<-0.4': "Risk/Rates divergent - Hedge positions",
                '<-0.7': "STRONG DIVERGENCE - Pairs trading favored"
            },
            ('PC2','PC3'): {
                '>0.8': "VERY STRONG RATES-USD LINK - Trade currencies",
                '>0.6': "Strong Rates-USD link - Trade currencies",
                '<-0.3': "Unusual divergence - Caution needed",
                '<-0.6': "EXTREME DIVERGENCE - Potential regime shift"
            },
            ('PC1','PC3'): {
                '>0.5': "USD strengthens with risk - Check commodities",
                '>0.3': "USD/risk co-movement - Adjust hedges",
                '<-0.2': "USD weakens as risk rises - Favor EM",
                '<-0.4': "STRONG RISK-USD INVERSE - Re-evaluate hedges"
            }
        }
        
        hover_text = np.empty_like(component_corr, dtype=object)
        for (i, j), val in np.ndenumerate(component_corr):
            pair = (component_corr.index[i], component_corr.columns[j])
            if pair in annotation_rules:
                for threshold, text in annotation_rules[pair].items():
                    if eval(f"{val}{threshold}"):
                        hover_text[i,j] = f"{text} (Corr: {val:.2f})"
                        break
                else:
                    hover_text[i,j] = f"Normal relationship (Corr: {val:.2f})"
            else:
                hover_text[i,j] = f"Correlation: {val:.2f}"
        
        fig_heat = go.Figure(go.Heatmap(
            z=component_corr,
            x=component_corr.columns,
            y=component_corr.index,
            colorscale='RdYlGn',
            zmin=-1,
            zmax=1,
            text=hover_text,
            hoverinfo="text+z",
            colorbar=dict(title="Correlation")
        ))
        
        # Add value annotations
        for i in range(len(component_corr)):
            for j in range(len(component_corr.columns)):
                val = component_corr.iloc[i,j]
                color = "white" if abs(val) > 0.5 else "black"
                fig_heat.add_annotation(
                    x=component_corr.columns[j],
                    y=component_corr.index[i],
                    text=f"{val:.2f}",
                    font=dict(color=color),
                    showarrow=False
                )
        
        st.plotly_chart(fig_heat, use_container_width=True)
        
    # --- Regime Monitor --- 
    st.subheader("📊 Market Regime Dashboard")
    regime_cols = st.columns([1,2])  # Wider right column for visualization
    
    # Calculate stability first
    stable_regime = (pca_stability['PC1']['stability_score'] > 0.8 and  # Changed from 0.7
                    pca_stability['PC2']['stability_score'] > 0.7)      # Changed from 0.6
    
    # Replace the VIX metric code (around line 785) with:
    with regime_cols[0]:
        # Stability Metrics
        st.metric("PCA Stability Score", 
                 f"{(pca_stability['PC1']['stability_score'] + pca_stability['PC2']['stability_score'])/2:.0%}",
                 "Stable" if stable_regime else "Unstable")
        
        # Core Regime Indicators
        st.write("#### Core Components")
        st.metric("Risk (PC1)", f"{current['PC1']:.2f}", 
                 "Risk-On" if current['PC1'] > 0.5 else "Risk-Off")
        st.metric("Rates (PC2)", f"{current['PC2']:.2f}",
                 "Rising" if current['PC2'] > 0.5 else "Falling")
        
        # Quick Status Checks
        st.write("#### Market Status")
        if not np.isnan(current['vix']):
            st.metric("VIX Level", f"{current['vix']:.1f}",
                     "Elevated" if current['vix'] > 22 else "Normal")
        else:
            st.metric("VIX Level", "N/A", "Data not available")
        st.metric("Residuals", current['residual_status'])
    
    with regime_cols[1]:
        # Visual Regime Tracker
        # Highlight last 10 points with larger markers and annotations
        fig = px.scatter(
            pca_df.tail(150),  # Still show 100 points for context
            x='PC1',
            y='PC2',
            color=pca_df.tail(150).index,
            title="Risk/Rates Regime (Last 100 Periods) - Last 10 Points Highlighted",
            labels={'PC1': 'Risk Component', 'PC2': 'Rates Component'},
            opacity=0.7  # Make older points semi-transparent
        )
        
        # Add highlighted last 10 points
        last_150 = pca_df.tail(45)
        fig.add_trace(
            go.Scatter(
                x=last_150['PC1'],
                y=last_150['PC2'],
                mode='markers+text',
                marker=dict(
                    color='red',
                    size=12,
                    line=dict(width=2, color='black')
                ),
                text=[f"t-{44-i}" for i in range(45)],  # Labels: t-0 (latest) to t-9
                textposition='top center',
                name='Last60 Moves',
                hoverinfo='text+x+y',
                hovertext=last_150.index.strftime('%Y-%m-%d %H:%M')
            )
        )
        
        # Add vertical bands for PC1 (Risk)
        fig.add_vrect(
            x0=0.5, x1=2.0,  # Risk-On zone (PC1 > 0.5)
            fillcolor="green", opacity=0.1, line_width=0,
            annotation_text="Risk-On", annotation_position="top left"
        )
        fig.add_vrect(
            x0=-2.0, x1=-0.5,  # Risk-Off zone (PC1 < -0.5)
            fillcolor="red", opacity=0.1, line_width=0,
            annotation_text="Risk-Off", annotation_position="top left"
        )
        
        # Existing horizontal bands for PC2 (Rates)
        fig.add_hrect(
            y0=0.3, y1=1.0, fillcolor="blue", opacity=0.1, line_width=0,
            annotation_text="Rates Rising", annotation_position="top right"
        )
        fig.add_hrect(
            y0=-1.0, y1=-0.3, fillcolor="purple", opacity=0.1, line_width=0,
            annotation_text="Rates Falling", annotation_position="bottom right"
        )
        # Add arrows to show movement direction (optional)
        for i in range(1, len(last_150)):
            fig.add_annotation(
                x=last_150['PC1'].iloc[i],
                y=last_150['PC2'].iloc[i],
                ax=last_150['PC1'].iloc[i-1],
                ay=last_150['PC2'].iloc[i-1],
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='black'
            )
        
        fig.update_layout(
            coloraxis_showscale=False,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Enhanced Residuals Analysis
        st.subheader("🔍 Residual Diagnostics")
        
        # Current Residual Status
        current_resid = pca_df[['residual_norm', 'sum_abs_residuals']].iloc[-1]
        resid_percentile = np.mean(pca_df['residual_norm'] < current_resid['residual_norm'])
        
        if resid_percentile > 0.9:
            st.error(f"""
            ⚠️ **Extreme Residuals Detected**  
            Current: {current_resid['residual_norm']:.2f}  
            Percentile: {resid_percentile*100:.1f}%  
            Model fit is deteriorating - consider:  
            - Reducing position sizes  
            - Adding more assets to PCA  
            - Increasing PCA window  
            """)
        elif resid_percentile > 0.75:
            st.warning(f"""
            ⚠️ **Elevated Residuals**  
            Current: {current_resid['residual_norm']:.2f}  
            Percentile: {resid_percentile*100:.1f}%  
            Model fit is weakening - monitor closely  
            """)
        else:
            st.success(f"""
            ✅ **Normal Residuals**  
            Current: {current_resid['residual_norm']:.2f}  
            Percentile: {resid_percentile*100:.1f}%  
            Model fit is good  
            """)
        
        # Residuals Over Time
        fig_resid = go.Figure()
        fig_resid.add_trace(go.Scatter(
            x=pca_df.index,
            y=pca_df['residual_norm'],
            name='Residual Norm',
            line=dict(color='#FF5722')
        ))
        fig_resid.add_trace(go.Scatter(
            x=pca_df.index,
            y=pca_df['sum_abs_residuals'],
            name='Sum Abs Residuals',
            line=dict(color='#2196F3')
        ))
        
        # Add thresholds
        fig_resid.add_hrect(
            y0=pca_df['residual_90th'].iloc[-1],
            y1=pca_df['residual_norm'].max(),
            fillcolor="red",
            opacity=0.2,
            annotation_text="Elevated Residual Zone"
        )
        
        st.plotly_chart(fig_resid, use_container_width=True)
        
        # Asset-Specific Residuals
        st.subheader("📌 Asset-Specific Residual Alerts")
        current_residuals = residuals_df.iloc[-1]
        resid_z = (current_residuals - residuals_df.mean()) / residuals_df.std()
        
        alert_assets = resid_z[abs(resid_z) > z_threshold].sort_values()
        if not alert_assets.empty:
            cols = st.columns(2)
            for i, (asset, z) in enumerate(alert_assets.items()):
                with cols[i % 2]:
                    direction = "overpriced" if z > 0 else "underpriced"
                    st.warning(f"""
                    **{asset.upper()}**  
                    Z-score: {z:.2f}  
                    Model suggests asset is {direction}  
                    Last residual: {current_residuals[asset]:.2f}  
                    Mean: {residuals_df[asset].mean():.2f}  
                    Std: {residuals_df[asset].std():.2f}  
                    """)
        else:
            st.info("No assets with significant residual deviations")
        
        # Residual Distribution by Asset
        st.subheader("📊 Residual Distributions (Latest Point Highlighted)")
 
        # Create the box plot using px.box (not Scatter)
        fig_box = px.box(
             residuals_df.melt(var_name='Asset', value_name='Residual'),
             x='Asset',
             y='Residual',
             color='Asset',
             points="all",  # This shows all points
             hover_data={'Residual': ':.2f'}
         )
         
         # Add highlighted points for the latest residuals
        latest_residuals = residuals_df.iloc[-1]
        for asset in residuals_df.columns:
             fig_box.add_trace(
                 go.Scatter(
                     x=[asset],
                     y=[latest_residuals[asset]],
                     mode='markers',
                     marker=dict(
                         color='black',
                         size=12,
                         line=dict(width=2, color='yellow')
                     ),
                     name='Latest',
                     showlegend=False,
                     hovertext=[f"Latest: {latest_residuals[asset]:.2f}"],
                     hoverinfo='text+x+y'
                 )
             )
         
         # Update box plot style
        fig_box.update_traces(
             boxpoints='all',  # Show all points
             jitter=0.3,       # Add some jitter for better visibility
             pointpos=0,       # Position points around the box
             marker=dict(size=4, opacity=0.5),
             selector={'type': 'box'}  # Only apply to box traces
         )
         
        st.plotly_chart(fig_box, use_container_width=True)
    
        with tab3:
            # --- Enhanced Debugging ---
            st.write("## 🔍 PCA Loadings Diagnostics")
            st.write("Number of assets:", len(df.columns))
            st.write("Loadings history shape:", loadings_history.shape)
            st.write("Expected loadings shape:", (3, len(df.columns)))
            
            # --- Safely reshape loadings ---
            try:
                # Get the most recent loadings (last row)
                last_loadings = loadings_history.iloc[-1]
                
                # Convert to numpy array and reshape (3 components × n_assets)
                loadings_array = last_loadings.values.reshape(3, -1)
                
                # Create DataFrame with proper labels
                loadings_df = pd.DataFrame(
                    loadings_array,
                    columns=df.columns,  # Asset names
                    index=['PC1', 'PC2', 'PC3']  # Component names
                )
                
                st.success("✅ Loadings successfully processed!")
                
            except Exception as e:
                st.error(f"❌ Loadings processing failed: {str(e)}")
                loadings_df = pd.DataFrame(
                    np.zeros((3, len(df.columns))),
                    columns=df.columns,
                    index=['PC1', 'PC2', 'PC3']
                )
        
            # --- Normalize for visualization ---
            norm_loadings = loadings_df.apply(lambda x: x / np.linalg.norm(x), axis=1)
            
            # --- Heatmap Visualization ---
            st.subheader("📊 Asset Contributions to Components")
            fig = px.imshow(
                norm_loadings.T,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                labels=dict(x="Principal Component", y="Asset", color="Loading"),
                aspect="auto"
            )
            fig.update_layout(
                title="Normalized PCA Loadings (Last Observation)",
                xaxis_title="Principal Component",
                yaxis_title="Asset"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- Contribution Alerts ---
            st.subheader("🚨 Significant Loadings")
            significant_threshold = 0.5
            significant_loadings = loadings_df[(loadings_df.abs() > significant_threshold).any(axis=1)]
            
            if not significant_loadings.empty:
                st.write("Assets with strong component contributions (>0.5 absolute loading):")
                st.dataframe(significant_loadings.style.background_gradient(cmap='RdBu', vmin=-1, vmax=1))
            else:
                st.info("No assets exceed loading threshold (|loading| > 0.5)")
        
            # --- Loadings Change Analysis ---
            st.subheader("📈 Loadings Trend Analysis")
            try:
                # Calculate 30d vs 90d changes
                recent_mean = loadings_history.iloc[-30:].mean()
                older_mean = loadings_history.iloc[-90:-60].mean()
                changes = (recent_mean - older_mean).abs()
                
                # Reshape changes to match components
                changes_array = changes.values.reshape(3, -1)
                changes_df = pd.DataFrame(
                    changes_array,
                    columns=df.columns,
                    index=['PC1', 'PC2', 'PC3']
                )
                
                # Highlight significant changes
                change_threshold = 0.1
                significant_changes = changes_df[changes_df > change_threshold].dropna(how='all')
                
                if not significant_changes.empty:
                    st.warning(f"Significant loading changes (> {change_threshold} absolute difference):")
                    st.dataframe(significant_changes.style.background_gradient(cmap='YlOrRd', vmin=0))
                else:
                    st.info(f"No significant changes in loadings (> {change_threshold} absolute difference)")
                    
            except Exception as e:
                st.error(f"Could not analyze loading changes: {str(e)}")
        
            # --- Asset Correlation Matrix ---
            st.subheader("🔄 Asset Correlation Matrix")
            asset_corr = df.corr()
            fig_corr = px.imshow(
                asset_corr,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                labels=dict(x="", y="", color="Correlation"),
                aspect="auto"
            )
            fig_corr.update_layout(title="90-Day Asset Correlations")
            st.plotly_chart(fig_corr, use_container_width=True)
        
            # --- Performance Metrics ---
            st.subheader("📈 Asset Performance Metrics")
            returns = df.pct_change().iloc[-90:]  # Last 90 periods
            perf_metrics = pd.DataFrame({
                'Return (90d)': returns.mean() * 90,
                'Volatility (90d)': returns.std() * np.sqrt(90),
                'Sharpe Ratio': returns.mean() / returns.std() * np.sqrt(90),
                'Max Drawdown': (1 - (df / df.rolling(90).max()).iloc[-90:].min())
            })
            
            st.dataframe(
                perf_metrics.style
                .background_gradient(cmap='RdYlGn', subset=['Return (90d)'])
                .background_gradient(cmap='YlOrRd', subset=['Volatility (90d)'])
                .background_gradient(cmap='RdYlGn', subset=['Sharpe Ratio'])
                .background_gradient(cmap='YlOrRd_r', subset=['Max Drawdown']),
                use_container_width=True
            )
if __name__ == "__main__":
    main()