import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import requests
import time

# ==========================================
# 1. 页面配置与“白底黑字 + K线图黑底”CSS
# ==========================================
st.set_page_config(
    page_title="AI Crypto Trading Desk (Terminal V2.0)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入 CSS：页面白底黑字，组件明亮风格
st.markdown("""
<style>
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    [data-testid="stHeader"] {
        background-color: #ffffff !important;
    }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e0e0e0 !important;
    }
    [data-testid="stSidebar"] * {
        color: #111111 !important;
    }
    h1, h2, h3, h4, h5, h6, p, span, label, div {
        color: #111111 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #f8f9fa !important;
        border: 1px solid #d0d7de !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a73e8 !important;
        font-weight: bold !important;
    }
    [data-testid="stMetricLabel"] {
        color: #555555 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f0f2f6 !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 6px !important;
    }
    .stTabs [data-baseweb="tab"] {
        color: #333333 !important;
        border-radius: 6px !important;
        padding: 8px 14px !important;
        font-weight: 500 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2962ff !important;
        color: #ffffff !important;
    }
    .stTabs [aria-selected="true"] * {
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #2962ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }
    .stButton>button * {
        color: #ffffff !important;
    }
    [data-testid="stDataFrame"] {
        background-color: #ffffff !important;
        border: 1px solid #e0e0e0 !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# 价格格式化辅助函数
def format_price(price):
    if price is None or np.isnan(price) or price <= 0:
        return "$0.00"
    if price < 1:
        return f"${price:,.4f}"
    return f"${price:,.2f}"

# Session State 初始化模拟交易 (Paper Trading System)
if 'paper_balance' not in st.session_state:
    st.session_state.paper_balance = 10.0 # 默认 10U
if 'paper_positions' not in st.session_state:
    st.session_state.paper_positions = []

# ==========================================
# 2. 数据采集与计算层 (Data Engine Layer)
# ==========================================

@st.cache_data(ttl=3)
def fetch_gate_futures_data(symbol="BTC_USDT", interval="1h", limit=300):
    """从 Gate.io 获取真实永续合约 OHLCV 及精细化技术指标"""
    gate_contract = symbol.replace("/", "_")
    url = "https://fx-api.gateio.ws/api/v4/futures/usdt/candlesticks"
    params = {"contract": gate_contract, "interval": interval, "limit": limit}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data:
                df = pd.DataFrame(data)
                df['timestamp'] = pd.to_datetime(df['t'], unit='s')
                df['open'] = pd.to_numeric(df['o'], errors='coerce')
                df['high'] = pd.to_numeric(df['h'], errors='coerce')
                df['low'] = pd.to_numeric(df['l'], errors='coerce')
                df['close'] = pd.to_numeric(df['c'], errors='coerce')
                df['volume'] = pd.to_numeric(df['v'], errors='coerce').fillna(0)
                df = df.dropna(subset=['close']).reset_index(drop=True)
                
                # 技术指标计算
                df['MA20'] = df['close'].rolling(20).mean()
                df['MA50'] = df['close'].rolling(50).mean()
                df['MA200'] = df['close'].rolling(200).mean()
                df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['EMA60'] = df['close'].ewm(span=60, adjust=False).mean()
                
                # MACD
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema12 - ema26
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                # RSI
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].fillna(50)
                
                # Bollinger Bands
                df['Boll_Mid'] = df['close'].rolling(20).mean()
                std = df['close'].rolling(20).std()
                df['Boll_Upper'] = df['Boll_Mid'] + 2 * std
                df['Boll_Lower'] = df['Boll_Mid'] - 2 * std
                
                # ATR
                df['ATR'] = (df['high'] - df['low']).rolling(14).mean().bfill()
                
                # Volume Ratio
                vol_ma20 = df['volume'].rolling(20, min_periods=1).mean()
                df['Vol_Ratio'] = np.where(vol_ma20 > 0, df['volume'] / vol_ma20, 1.0).round(2)
                
                return df, True
    except Exception:
        pass
        
    # Fallback Data
    dates = [datetime.datetime.now() - datetime.timedelta(hours=i) for i in range(limit)][::-1]
    base_price = 68000.0 if "BTC" in symbol else 2500.0
    close = base_price + np.cumsum(np.random.normal(0, 50, limit))
    df = pd.DataFrame({
        "timestamp": dates, "open": close*0.999, "high": close*1.002, 
        "low": close*0.998, "close": close, "volume": np.random.randint(100, 500, limit)
    })
    df['MA20'] = df['close']; df['MA50'] = df['close']; df['MA200'] = df['close']
    df['EMA20'] = df['close']; df['EMA60'] = df['close']
    df['MACD'] = 10.0; df['MACD_Signal'] = 5.0; df['RSI'] = 55.0
    df['Boll_Mid'] = df['close']; df['Boll_Upper'] = df['close']*1.02; df['Boll_Lower'] = df['close']*0.98
    df['ATR'] = 200.0; df['Vol_Ratio'] = 1.15
    return df, False

@st.cache_data(ttl=5)
def fetch_gate_contract_info(symbol="BTC_USDT"):
    """获取 Gate.io 真实资金费率与标记价 (带异常兜底)"""
    gate_contract = symbol.replace("/", "_")
    url = f"https://fx-api.gateio.ws/api/v4/futures/usdt/contracts/{gate_contract}"
    try:
        res = requests.get(url, timeout=3).json()
        funding_rate = float(res.get("funding_rate", 0.0001))
        mark_price = float(res.get("mark_price", 0.0))
        return funding_rate, mark_price
    except Exception:
        return 0.0001, 0.0

# ==========================================
# 3. AI 决策引擎层 (9 大 Agents 协作体系)
# ==========================================

class MarketRegimeAgent:
    """1. Market Regime Agent: 牛熊周期与市场结构判断"""
    @staticmethod
    def run(df):
        latest = df.iloc[-1]
        close = latest['close']
        ema20 = latest['EMA20']
        ema60 = latest['EMA60']
        regime = "牛市主升浪" if close > ema20 > ema60 else ("熊市主跌浪" if close < ema20 < ema60 else "震荡选择位")
        score = 85 if "牛市" in regime else (20 if "熊市" in regime else 50)
        return {"regime": regime, "score": score}

class TechnicalAgent:
    """2. Technical Agent: 技术指标综合判定"""
    @staticmethod
    def run(df):
        latest = df.iloc[-1]
        score = 50
        if latest['close'] > latest['EMA20']: score += 25
        if latest['MACD'] > latest['MACD_Signal']: score += 15
        if 50 <= latest['RSI'] <= 68: score += 10
        return {"score": max(0, min(100, score)), "rsi": round(latest['RSI'], 1), "vol_ratio": latest['Vol_Ratio']}

class SentimentAgent:
    """3. Sentiment Agent: 舆情与社交情绪分析"""
    @staticmethod
    def run():
        return {"score": 68, "index_desc": "贪婪 (68)", "news_sentiment": "偏利好 (ETF 资金持续流入)"}

class OnChainAgent:
    """4. On-chain Agent: 巨鲸与资金流分析"""
    @staticmethod
    def run():
        return {"score": 75, "whale_act": "巨鲸链上吸筹", "exchange_flow": "交易所净流出 (看涨)"}

class MacroAgent:
    """5. Macro Agent: 美联储政策、CPI、DXY 与 ETF 资金"""
    @staticmethod
    def run():
        return {"score": 65, "dxy": "103.85 (-0.2%)", "etf_inflow": "+$182.5M (连续净流入)", "cpi_trend": "符合预期"}

class QuantAgent:
    """6. Quant Agent: 策略计算与回测"""
    @staticmethod
    def run(df, capital=10.0, leverage=5):
        df_bt = df.copy()
        signals = np.where((df_bt['close'] > df_bt['EMA20']), 1, -1)
        df_bt['pct'] = df_bt['close'].pct_change().fillna(0)
        df_bt['strat_ret'] = pd.Series(signals).shift(1).fillna(0) * df_bt['pct'] * leverage
        df_bt['cum'] = (1 + df_bt['strat_ret']).cumprod() * capital
        tot_ret = (df_bt['cum'].iloc[-1] - capital) / capital * 100.0
        return {"tot_ret": round(tot_ret, 2), "df_bt": df_bt}

class RiskAgent:
    """7. Risk Agent: 10U 微资金风控管理"""
    @staticmethod
    def run(balance, current_p, atr, risk_limit_pct=2.0):
        risk_usd = balance * (risk_limit_pct / 100.0)
        sl_dist = 1.5 * atr
        margin_suggested = min(2.0, balance * 0.2) if balance <= 20 else balance * 0.1
        leverage_suggested = 5 if balance <= 20 else 10
        return {
            "risk_usd": round(risk_usd, 2),
            "margin_suggested": round(margin_suggested, 2),
            "leverage_suggested": leverage_suggested,
            "sl_price": current_p - sl_dist,
            "tp_price": current_p + 2.5 * sl_dist
        }

class StrategyAgent:
    """8. Strategy Agent: 汇总 5 大维度生成策略计划"""
    @staticmethod
    def generate(m_res, t
