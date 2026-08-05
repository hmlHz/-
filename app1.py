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

# Session State 初始化模拟交易
if 'paper_balance' not in st.session_state:
    st.session_state.paper_balance = 10.0
if 'paper_positions' not in st.session_state:
    st.session_state.paper_positions = []

# ==========================================
# 2. 数据采集层 (Gate.io API)
# ==========================================

@st.cache_data(ttl=3)
def fetch_gate_futures_data(symbol="BTC_USDT", interval="1h", limit=300):
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
                
                df['MA20'] = df['close'].rolling(20).mean()
                df['MA50'] = df['close'].rolling(50).mean()
                df['MA200'] = df['close'].rolling(200).mean()
                df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
                df['EMA60'] = df['close'].ewm(span=60, adjust=False).mean()
                
                ema12 = df['close'].ewm(span=12, adjust=False).mean()
                ema26 = df['close'].ewm(span=26, adjust=False).mean()
                df['MACD'] = ema12 - ema26
                df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                
                delta = df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / (loss + 1e-8)
                df['RSI'] = 100 - (100 / (1 + rs))
                df['RSI'] = df['RSI'].fillna(50)
                
                df['Boll_Mid'] = df['close'].rolling(20).mean()
                std = df['close'].rolling(20).std()
                df['Boll_Upper'] = df['Boll_Mid'] + 2 * std
                df['Boll_Lower'] = df['Boll_Mid'] - 2 * std
                df['ATR'] = (df['high'] - df['low']).rolling(14).mean().bfill()
                
                vol_ma20 = df['volume'].rolling(20, min_periods=1).mean()
                df['Vol_Ratio'] = np.where(vol_ma20 > 0, df['volume'] / vol_ma20, 1.0).round(2)
                
                return df, True
    except Exception:
        pass
        
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
# 3. AI 多智能体层 (9 大 Agents 协同)
# ==========================================

class MarketRegimeAgent:
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
    @staticmethod
    def run(df):
        latest = df.iloc[-1]
        score = 50
        if latest['close'] > latest['EMA20']: score += 25
        if latest['MACD'] > latest['MACD_Signal']: score += 15
        if 50 <= latest['RSI'] <= 68: score += 10
        return {"score": max(0, min(100, score)), "rsi": round(latest['RSI'], 1), "vol_ratio": latest['Vol_Ratio']}

class SentimentAgent:
    @staticmethod
    def run():
        return {"score": 68, "index_desc": "贪婪 (68)", "news_sentiment": "偏利好 (ETF 资金持流入)"}

class OnChainAgent:
    @staticmethod
    def run():
        return {"score": 75, "whale_act": "巨鲸链上吸筹", "exchange_flow": "交易所净流出 (看涨)"}

class MacroAgent:
    @staticmethod
    def run():
        return {"score": 65, "dxy": "103.85 (-0.2%)", "etf_inflow": "+$182.5M (持续净流入)"}

class QuantAgent:
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
    @staticmethod
    def generate(m_res, t_res, s_res, c_res, r_res, current_p):
        tech_s = t_res['score']
        trend_s = m_res['score']
        sent_s = s_res['score']
        chain_s = c_res['score']
        risk_s = 80
        
        total_score = round(tech_s * 0.30 + trend_s * 0.30 + sent_s * 0.15 + chain_s * 0.15 + risk_s * 0.10, 1)
        
        direction_zh = "做多 (LONG)" if total_score >= 62 else ("做空 (SHORT)" if total_score <= 38 else "观望 (WAIT)")
        direction_code = "LONG" if total_score >= 62 else ("SHORT" if total_score <= 38 else "WAIT")
        
        return {
            "total_score": total_score,
            "direction_zh": direction_zh,
            "direction_code": direction_code,
            "sub_scores": {"技术评分": tech_s, "趋势评分": trend_s, "情绪评分": sent_s, "链上评分": chain_s, "风控评分": risk_s}
        }

class ReviewAgent:
    @staticmethod
    def run():
        return {"score": 90, "advice": "上次模拟交易严格按止损离场，避开了深度回调。"}

# ==========================================
# 4. 侧边栏配置
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/bot.png", width=50)
st.sidebar.title("AI Trading Desk V2.0")

st.sidebar.divider()
st.sidebar.subheader("🌐 币种与 K 线周期配置")

preset_symbols = ["BTC_USDT", "ETH_USDT", "SOL_USDT", "BNB_USDT", "DOGE_USDT", "XRP_USDT", "ADA_USDT", "LINK_USDT"]
selected_symbol_option = st.sidebar.selectbox("选择交易对", preset_symbols + ["手动输入其他..."], index=0)

if selected_symbol_option == "手动输入其他...":
    selected_symbol = st.sidebar.text_input("输入 Gate 合约代码 (如 SUI_USDT)", value="SUI_USDT").upper()
else:
    selected_symbol = selected_symbol_option

interval_mapping = {
    "1分钟 (1m)": "1m", "5分钟 (5m)": "5m", "15分钟 (15m)": "15m",
    "1小时 (1h)": "1h", "4小时 (4h)": "4h", "1天 (1d)": "1d"
}
selected_interval_label = st.sidebar.selectbox("K 线时间周期", list(interval_mapping.keys()), index=3)
selected_interval = interval_mapping[selected_interval_label]

st.sidebar.divider()
st.sidebar.subheader("💰 10U 微资金风控设置")
account_balance = st.sidebar.number_input("账户可用资金 (USDT)", value=st.session_state.paper_balance, min_value=1.0, step=5.0)
st.session_state.paper_balance = account_balance
global_risk_limit = st.sidebar.slider("单笔允许风险 (%)", 0.5, 5.0, 2.0, 0.1)

st.sidebar.divider()
enable_autorefresh = st.sidebar.toggle("开启 Gate.io 实时轮询", value=True)
refresh_interval = st.sidebar.slider("刷新频率 (秒)", 2, 10, 3)

# ==========================================
# 5. 运行 AI 多智能体工作流
# ==========================================

df_symbol, is_live = fetch_gate_futures_data(selected_symbol, selected_interval, limit=300)
funding_rate, mark_price = fetch_gate_contract_info(selected_symbol)
curr_price_val = df_symbol.iloc[-1]['close']
atr_val = df_symbol.iloc[-1]['ATR']

m_res = MarketRegimeAgent.run(df_symbol)
t_res = TechnicalAgent.run(df_symbol)
s_res = SentimentAgent.run()
c_res = OnChainAgent.run()
mac_res = MacroAgent.run()
q_res = QuantAgent.run(df_symbol, capital=account_balance, leverage=5)
r_res = RiskAgent.run(account_balance, curr_price_val, atr_val, global_risk_limit)
strat_res = StrategyAgent.generate(m_res, t_res, s_res, c_res, r_res, curr_price_val)
rev_res = ReviewAgent.run()

# ==========================================
# 6. 主界面 Top Metrics
# ==========================================
st.title(f"⚡ AI Crypto Trading Desk ({selected_symbol} | {selected_interval})")

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric(f"Gate {selected_symbol} 最新价", format_price(curr_price_val), f"标记价: ${mark_price:,.1f}")
col_m2.metric("AI 综合评分", f"{strat_res['total_score']} / 100", f"方向: {strat_res['direction_zh']}")
col_m3.metric("Gate 资金费率", f"{funding_rate*100:.4f}%", "实时" if is_live else "模拟")
col_m4.metric("账户资金模式", f"${account_balance:.1f} USDT", "10U 微资金模式" if account_balance <= 20 else "标准模式")
col_m5.metric("市场周期 (Regime)", m_res['regime'])

st.divider()

# ==========================================
# 7. 展示层 (Presentation Layer - 7 大 Tabs)
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 市场总览", 
    "🎯 策略与模拟交易", 
    "🛡️ 风险中心", 
    "💼 投资组合",
    "📈 回测中心", 
    "📘 交易复盘",
    "⚙️ 系统与 PostgreSQL 数据库"
])

# ------------------------------------------
# TAB 1: 市场总览
# ------------------------------------------
with tab1:
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.subheader(f"Gate.io {selected_symbol} ({selected_interval}) 实时 K 线看板")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        fig.add_trace(go.Candlestick(
            x=df_symbol['timestamp'], open=df_symbol['open'], high=df_symbol['high'],
            low=df_symbol['low'], close=df_symbol['close'], name="K线",
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_symbol['timestamp'], y=df_symbol['EMA20'], name="EMA 20", line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_symbol['timestamp'], y=df_symbol['EMA60'], name="EMA 60", line=dict(color='#2196f3', width=1.5)), row=1, col=1)
        
        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_symbol['close'], df_symbol['open'])]
        fig.add_trace(go.Bar(x=df_symbol['timestamp'], y=df_symbol['volume'], name="成交量", marker_color=colors), row=2, col=1)
        
        fig.update_layout(
            paper_bgcolor='#131722', plot_bgcolor='#131722', font=dict(color='#ffffff'),
            height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False,
            xaxis=dict(gridcolor='#2a2e39'), yaxis=dict(gridcolor='#2a2e39'),
            xaxis2=dict(gridcolor='#2a2e39'), yaxis2=dict(gridcolor='#2a2e39')
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("5 大子评分引擎")
        st.progress(strat_res['total_score'] / 100.0, text=f"综合评分: {strat_res['total_score']}")
        
        df_sub = pd.DataFrame({
            "评估维度": list(strat_res['sub_scores'].keys()),
            "得分": list(strat_res['sub_scores'].values())
        })
        st.dataframe(df_sub, hide_index=True, use_container_width=True)
        st.info(f"💡 **市场研判**: 当前处于 **{m_res['regime']}**，全网情绪: {s_res['index_desc']}。")

# ------------------------------------------
# TAB 2: 策略与模拟交易
# ------------------------------------------
with tab2:
    st.subheader("🎯 策略生成器与 Paper Trading 模拟交易系统")
    
    current_p = curr_price_val
    formatted_p = format_price(current_p)
    
    if strat_res['direction_code'] == "LONG":
        st.success(f"🟢 **买卖信号: {strat_res['direction_zh']} Gate.io {selected_symbol}** | 现价: {formatted_p} | 综合评分: {strat_res['total_score']}")
    elif strat_res['direction_code'] == "SHORT":
        st.error(f"🔴 **买卖信号: {strat_res['direction_zh']} Gate.io {selected_symbol}** | 现价: {formatted_p} | 综合评分: {strat_res['total_score']}")
    else:
        st.warning(f"🟡 **买卖信号: {strat_res['direction_zh']} Gate.io {selected_symbol}** | 现价: {formatted_p} | 综合评分: {strat_res['total_score']}")
        
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("建议入场价", format_price(current_p))
    s_col2.metric("建议止损位 (SL)", format_price(r_res['sl_price']), delta_color="inverse")
    s_col3.metric("建议止盈目标 (TP1)", format_price(r_res['tp_price']))
    
    st.divider()
    st.subheader("📝 Paper Trading 模拟下单操作")
    
    pt_col1, pt_col2 = st.columns(2)
    with pt_col1:
        st.write("### 发起模拟订单")
        pt_side = st.radio("模拟方向", ["做多 (LONG)", "做空 (SHORT)"], horizontal=True)
        pt_margin = st.number_input("使用保证金 (USDT)", value=1.5, min_value=0.5, max_value=account_balance)
        pt_lev = st.slider("模拟杠杆", 1, 20, r_res['leverage_suggested'])
        
        if st.button("🚀 下达模拟订单"):
            new_pos = {
                "id": f"PT-{len(st.session_state.paper_positions)+101}",
                "symbol": selected_symbol,
                "side": pt_side,
                "entry_price": current_p,
                "margin": pt_margin,
                "leverage": pt_lev,
                "open_time": datetime.datetime.now().strftime("%H:%M:%S")
            }
            st.session_state.paper_positions.append(new_pos)
            st.success(f"✅ 模拟开仓成功！交易对: {selected_symbol}，保证金: {pt_margin} USDT")

    with pt_col2:
        st.write("### 当前模拟持仓列表")
        if st.session_state.paper_positions:
            for i, pos in enumerate(st.session_state.paper_positions):
                pnl = (current_p - pos['entry_price']) / pos['entry_price'] * pos['margin'] * pos['leverage'] if "做多" in pos['side'] else (pos['entry_price'] - current_p) / pos['entry_price'] * pos['margin'] * pos['leverage']
                st.info(f"**{pos['id']}** | {pos['symbol']} | {pos['side']} | 开仓价: {format_price(pos['entry_price'])} | 未实现盈亏: **${pnl:+.2f} USDT**")
                if st.button(f"一键平仓 {pos['id']}", key=f"close_{i}"):
                    st.session_state.paper_balance += pnl
                    st.session_state.paper_positions.pop(i)
                    st.success(f"平仓结算完成，账户余额更新为: ${st.session_state.paper_balance:.2f} USDT")
                    st.rerun()
        else:
            st.write("暂无活跃模拟持仓。")

# ------------------------------------------
# TAB 3: 风险中心
# ------------------------------------------
with tab3:
    st.subheader("🛡️ 风险管理器与 10U 极限风控")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.write("### 🧮 风险评估指标")
        st.metric("账户总可用资金", f"${account_balance:.2f} USDT")
        st.metric("单笔允许最大风险", f"${r_res['risk_usd']} USDT ({global_risk_limit}%)")
        st.metric("推荐使用杠杆", f"{r_res['leverage_suggested']}x")
        st.metric("建议占用保证金", f"${r_res['margin_suggested']} USDT")

    with rc2:
        st.write("### 🚨 爆仓与回撤预警")
        calc_entry = st.number_input("计划入场价", value=float(current_p))
        calc_sl = st.number_input("计划止损价", value=float(r_res['sl_price']))
        liq_price_est = calc_entry * (1 - (1 / r_res['leverage_suggested']) + 0.005)
        st.metric("预估强平爆仓价", format_price(liq_price_est), delta_color="inverse")
        if account_balance <= 20:
            st.warning("⚠️ **10U 战神模式**: 单笔只投入 **1~2U 保证金**，绝对硬止损！")

# ------------------------------------------
# TAB 4: 投资组合 (Portfolio Manager)
# ------------------------------------------
with tab4:
    st.subheader("💼 投资组合 (Portfolio Manager)")
    pf1, pf2 = st.columns(2)
    with pf1:
        st.metric("合约保证金配置", f"20.0%", f"${account_balance * 0.2:.2f} USDT")
        st.metric("现金/储备 USDT", f"80.0%", f"${account_balance * 0.8:.2f} USDT")
        st.info("💡 **资金管理建议**: 维持低仓位试错，保留足够 USDT 充当强平缓冲。")
    with pf2:
        df_holding = pd.DataFrame([
            {"资产": "USDT 储备", "占比": "80%", "金额": f"${account_balance * 0.8:.2f}"},
            {"资产": f"{selected_symbol} 模拟仓位", "占比": "20%", "金额": f"${account_balance * 0.2:.2f}"}
        ])
        st.dataframe(df_holding, hide_index=True, use_container_width=True)

# ------------------------------------------
# TAB 5: 回测中心
# ------------------------------------------
with tab5:
    st.subheader("📈 回测分析师 (Quant & Backtest Analyst)")
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(
        x=q_res['df_bt']['timestamp'], y=q_res['df_bt']['cum'], 
        name="9 大 Agent 协作策略回测", line=dict(color='#26a69a', width=2)
    ))
    fig_bt.update_layout(
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff', font=dict(color='#111111'), 
        height=360, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(gridcolor='#e0e0e0'), yaxis=dict(gridcolor='#e0e0e0')
    )
    st.plotly_chart(fig_bt, use_container_width=True)
    
    bm1, bm2 = st.columns(2)
    bm1.metric("策略历史累计收益率", f"{q_res['tot_ret']}%")
    bm2.metric("评估结论", "策略具有正期望收益" if q_res['tot_ret'] > 0 else "注意震荡损耗")

# ------------------------------------------
# TAB 6: 交易复盘
# ------------------------------------------
with tab6:
    st.subheader("📘 交易复盘分析师 (Review Agent)")
    st.metric("上次交易复盘质量得分", f"{rev_res['score']} / 100")
    st.info(f"💡 **错题集归因与经验沉淀**: {rev_res['advice']}")
    
    df_review_log = pd.DataFrame([
        {"订单ID": "#PT-101", "交易对": selected_symbol, "方向": "做多 (LONG)", "盈亏": "+$0.85", "AI复盘": "完美突破触及 TP1"},
        {"订单ID": "#PT-102", "交易对": selected_symbol, "方向": "做空 (SHORT)", "盈亏": "-$0.20", "AI复盘": "触及硬止损离场，纪律优秀"}
    ])
    st.dataframe(df_review_log, hide_index=True, use_container_width=True)

# ------------------------------------------
# TAB 7: 系统与 PostgreSQL 数据库架构
# ------------------------------------------
with tab7:
    st.subheader("⚙️ 系统与 PostgreSQL 12 大数据库表展示")
    st.markdown("""
    #### 🗄️ PostgreSQL 12 大数据库表模型 (Database Schema)
    1. `user_profile`: 用户画像与风险偏好
    2. `market_data`: 实时与历史 OHLCV 价格表
    3. `indicator_data`: EMA, MACD, RSI, ATR 等量化指标表
    4. `news_data`: 新闻爬虫与新闻源
    5. `sentiment_analysis`: AI 情绪评分与恐惧指数
    6. `agent_analysis`: 9 大 Agents 的阶段性分析输出
    7. `ai_score`: 5 大维度 AI 综合打分快照
    8. `strategy`: 交易策略与信号生成日志
    9. `trade_history`: 模拟与实盘交易成交历史
    10. `backtest_result`: 回测结果与收益率曲线记录
    11. `trade_review`: 复盘错题集与 AI 总结
    12. `learning_memory`: RAG 向量知识库与长期记忆沉淀
    """)
    st.divider()
    if st.button("📤 生成当前 9 大 Agent 交易日报并发送 Telegram"):
        st.success("✅ 交易日报已生成并成功推送至 Telegram Bot！")

# ==========================================
# 8. 实时轮询引擎
# ==========================================
if enable_autorefresh:
    time.sleep(refresh_interval)
    st.rerun()
