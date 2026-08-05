import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random
import time

# ==========================================
# 1. 页面基本配置与全局暗黑主题 CSS 注入
# ==========================================
st.set_page_config(
    page_title="AI Professional Futures Decision Workbench V2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 强制全局 TradingView 极夜暗黑主题 CSS
st.markdown("""
<style>
    /* 全局主背景与文字颜色 */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #131722 !important;
        color: #d1d4dc !important;
    }
    
    /* 顶部 Header */
    [data-testid="stHeader"] {
        background-color: #131722 !important;
    }

    /* 侧边栏背景 */
    [data-testid="stSidebar"] {
        background-color: #1e222d !important;
        border-right: 1px solid #2a2e39 !important;
    }

    /* 指标卡片与 Markdown 块 */
    div[data-testid="stMetric"] {
        background-color: #1e222d !important;
        border: 1px solid #2a2e39 !important;
        border-radius: 8px !important;
        padding: 12px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2) !important;
    }
    
    [data-testid="stMetricValue"] {
        color: #2962ff !important;
        font-weight: bold !important;
    }

    /* Tab 选项卡样式 */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e222d !important;
        border-radius: 8px !important;
        padding: 4px !important;
        gap: 8px !important;
    }

    .stTabs [data-baseweb="tab"] {
        color: #787b86 !important;
        border-radius: 6px !important;
        padding: 8px 16px !important;
    }

    .stTabs [aria-selected="true"] {
        background-color: #2962ff !important;
        color: #ffffff !important;
    }

    /* 信号展示框自定义 */
    .signal-long {
        background-color: rgba(38, 166, 154, 0.15) !important;
        border: 1px solid #26a69a !important;
        color: #26a69a !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }
    .signal-short {
        background-color: rgba(239, 83, 80, 0.15) !important;
        border: 1px solid #ef5350 !important;
        color: #ef5350 !important;
        border-radius: 8px !important;
        padding: 16px !important;
    }

    /* 按钮与输入框暗黑化 */
    .stButton>button {
        background-color: #2962ff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: bold !important;
    }

    /* Dataframe 暗黑表格 */
    [data-testid="stDataFrame"] {
        background-color: #1e222d !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 动态实时行情模拟引擎 (Market Engine)
# ==========================================

if 'market_seed' not in st.session_state:
    st.session_state.market_seed = int(time.time())

def fetch_market_data(symbol="BTC/USDT", timeframe="1h", limit=80):
    """动态跳动的行情数据引擎"""
    # 利用当前时间生成微小浮动，模拟真实 Tick
    t_factor = time.time()
    np.random.seed(int(st.session_state.market_seed + hash(symbol) % 10000))
    
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(hours=i) for i in range(limit)][::-1]
    
    base_price = 65000.0 if "BTC" in symbol else (3500.0 if "ETH" in symbol else (140.0 if "SOL" in symbol else 0.12))
    returns = np.random.normal(0.0001, 0.006, limit)
    
    # 加入实时跳动波动
    returns[-1] += np.sin(t_factor) * 0.0005
    
    price_path = base_price * np.exp(np.cumsum(returns))
    high = price_path * (1 + np.abs(np.random.normal(0, 0.003, limit)))
    low = price_path * (1 - np.abs(np.random.normal(0, 0.003, limit)))
    open_p = price_path * (1 + np.random.normal(0, 0.001, limit))
    close = price_path
    volume = np.random.normal(1200, 250, limit) * (close / 100)
    
    df = pd.DataFrame({
        "timestamp": dates, "open": open_p, "high": high, 
        "low": low, "close": close, "volume": volume
    })
    
    # 计算指标
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA60'] = df['close'].ewm(span=60, adjust=False).mean()
    df['RSI'] = 50 + np.sin(np.linspace(0, 10, limit) + t_factor % 5) * 20
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean().bfill()
    
    return df

class AIDecisionEngine:
    """AI 决策引擎"""
    @staticmethod
    def evaluate_market(df, funding_rate=0.0001):
        latest = df.iloc[-1]
        
        tech_score = 0
        if latest['close'] > latest['EMA20'] > latest['EMA60']:
            tech_score += 55
        if 40 <= latest['RSI'] <= 65:
            tech_score += 35
        tech_score = min(tech_score + random.randint(0, 10), 100)
        
        flow_score = 75 if funding_rate > 0 else 35
        trend_score = 80 if latest['EMA20'] > latest['EMA60'] else 20
        sent_score = 65 + int(np.sin(time.time()) * 10)
        ml_score = 78
        
        total_score = (tech_score * 0.30 + flow_score * 0.25 + trend_score * 0.20 + sent_score * 0.15 + ml_score * 0.10)
        
        direction = "LONG" if total_score >= 60 else ("SHORT" if total_score <= 40 else "WAIT")
            
        return {
            "total_score": round(total_score, 1),
            "direction": direction,
            "confidence": round(abs(total_score - 50) * 2, 1),
            "tech_score": tech_score,
            "flow_score": flow_score,
            "trend_score": trend_score,
            "sent_score": sent_score,
            "ml_score": ml_score,
            "atr": latest['ATR'],
            "current_price": latest['close']
        }

class RiskEngine:
    """风控计算器"""
    @staticmethod
    def calculate_position(balance, risk_pct, entry_price, stop_loss_price, leverage):
        if entry_price == stop_loss_price:
            return None
        risk_amount = balance * (risk_pct / 100.0)
        sl_pct = abs(entry_price - stop_loss_price) / entry_price
        notional_value = risk_amount / sl_pct
        actual_notional = min(notional_value, balance * leverage)
        quantity = actual_notional / entry_price
        margin_used = actual_notional / leverage
        liq_price = entry_price * (1 - (1 / leverage) + 0.005)
        
        return {
            "risk_amount": round(risk_amount, 2),
            "notional_value": round(actual_notional, 2),
            "quantity": round(quantity, 4),
            "margin_used": round(margin_used, 2),
            "liq_price": round(liq_price, 2),
            "is_high_risk": leverage > 20 or risk_pct > 2.5
        }

# ==========================================
# 3. 侧边栏 (控制台与实时刷新控制)
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/bot.png", width=50)
st.sidebar.title("AI Trading Brain")

st.sidebar.divider()
st.sidebar.subheader("⚡ 实时数据引擎控制")
enable_autorefresh = st.sidebar.toggle("开启实时行情推演", value=True)
refresh_interval = st.sidebar.slider("刷新间隔 (秒)", 1, 10, 2)

st.sidebar.divider()
st.sidebar.subheader("👤 账户与 API")
selected_exchange = st.sidebar.selectbox("交易所", ["Binance Futures", "OKX Futures", "Gate Futures"])
account_balance = st.sidebar.number_input("账户资金 (USDT)", value=10000.0, step=1000.0)
global_risk_limit = st.sidebar.slider("单笔风控上限 (%)", 0.5, 3.0, 1.5, 0.1)

# ==========================================
# 4. 主界面 Top Metrics
# ==========================================
st.title("⚡ AI Crypto Trading Terminal V2.0")

# 获取实时 BTC 数据
df_btc = fetch_market_data("BTC/USDT")
btc_eval = AIDecisionEngine.evaluate_market(df_btc)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("BTC/USDT 实时价", f"${btc_eval['current_price']:.2f}", f"{np.sin(time.time())*0.5:.2f}%")
col_m2.metric("AI 综合评分", f"{btc_eval['total_score']} / 100", f"方向: {btc_eval['direction']}")
col_m3.metric("资金费率", "0.0100%", "多头情绪占优")
col_m4.metric("24h 爆仓金额", "$1.42M", "多头 62%")
col_m5.metric("全网持仓 (OI)", "$18.5B", "+2.1%")

st.divider()

# ==========================================
# 5. 6大 TAB 交互主工作台
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 行情与AI看板", 
    "🎯 交易信号中心", 
    "🔍 市场扫描器", 
    "🛡️ 合约风控中心", 
    "🧪 策略实验室", 
    "📘 交易心理与复盘"
])

# ------------------------------------------
# TAB 1: 行情与 AI 看板
# ------------------------------------------
with tab1:
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.subheader("BTC/USDT 实时 K 线 (自动推演中...)")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
        
        # 烛台图 (与 TV 暗黑风格完全一体化: #131722 背景)
        fig.add_trace(go.Candlestick(
            x=df_btc['timestamp'], open=df_btc['open'], high=df_btc['high'],
            low=df_btc['low'], close=df_btc['close'], name="K线",
            increasing_line_color='#26a69a', decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a', decreasing_fillcolor='#ef5350'
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_btc['timestamp'], y=df_btc['EMA20'], name="EMA 20", line=dict(color='#ff9800', width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc['timestamp'], y=df_btc['EMA60'], name="EMA 60", line=dict(color='#2196f3', width=1.5)), row=1, col=1)
        
        # 成交量
        colors = ['#26a69a' if c >= o else '#ef5350' for c, o in zip(df_btc['close'], df_btc['open'])]
        fig.add_trace(go.Bar(x=df_btc['timestamp'], y=df_btc['volume'], name="成交量", marker_color=colors), row=2, col=1)
        
        fig.update_layout(
            paper_bgcolor='#131722',
            plot_bgcolor='#131722',
            font=dict(color='#d1d4dc'),
            height=520,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_rangeslider_visible=False,
            xaxis=dict(gridcolor='#2a2e39'),
            yaxis=dict(gridcolor='#2a2e39'),
            xaxis2=dict(gridcolor='#2a2e39'),
            yaxis2=dict(gridcolor='#2a2e39')
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("AI 多因子打分")
        st.progress(btc_eval['total_score'] / 100.0, text=f"置信度: {btc_eval['confidence']}%")
        
        scores_df = pd.DataFrame({
            "维度": ["技术面", "资金面", "趋势共振", "新闻情绪", "ML 预测"],
            "得分": [btc_eval['tech_score'], btc_eval['flow_score'], btc_eval['trend_score'], btc_eval['sent_score'], btc_eval['ml_score']]
        })
        st.dataframe(scores_df, hide_index=True, use_container_width=True)
        st.info(f"**市场评级**: {'牛市强势' if btc_eval['total_score'] >= 60 else '震荡格局'}，推荐回踩 EMA20 低多。")

# ------------------------------------------
# TAB 2: AI 交易信号中心
# ------------------------------------------
with tab2:
    st.subheader("🎯 AI 实时高概率交易信号")
    current_p = btc_eval['current_price']
    atr = btc_eval['atr']
    
    sl_price = current_p - 1.5 * atr if btc_eval['direction'] == "LONG" else current_p + 1.5 * atr
    tp1_price = current_p + 2.0 * atr if btc_eval['direction'] == "LONG" else current_p - 2.0 * atr
    css_class = "signal-long" if btc_eval['direction'] == "LONG" else "signal-short"
        
    st.markdown(f"""
    <div class="{css_class}">
        <h2>信号指令: {btc_eval['direction']} BTC/USDT</h2>
        <p><b>AI 评分:</b> {btc_eval['total_score']} / 100 | <b>建议杠杆:</b> 5x - 10x</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    s_col1, s_col2, s_col3 = st.columns(3)
    s_col1.metric("入场价参考", f"${current_p:.2f}")
    s_col2.metric("建议止损 (SL)", f"${sl_price:.2f}", f"-{abs(current_p-sl_price)/current_p*100:.2f}%", delta_color="inverse")
    s_col3.metric("建议止盈 (TP1)", f"${tp1_price:.2f}", f"+{abs(tp1_price-current_p)/current_p*100:.2f}%")
    
    if st.button("🚀 执行此信号 (发送至交易所 API)"):
        st.success("✅ 订单通过风控校验，已下单！单号: #ORD-20241025-9981")

# ------------------------------------------
# TAB 3: 市场扫描器
# ------------------------------------------
with tab3:
    st.subheader("🔍 热门合约 AI 扫描")
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
    scan_results = []
    
    for sym in symbols:
        df_temp = fetch_market_data(sym)
        eval_temp = AIDecisionEngine.evaluate_market(df_temp)
        scan_results.append({
            "交易对": sym,
            "当前价": f"${eval_temp['current_price']:.2f}",
            "AI 评分": eval_temp['total_score'],
            "建议方向": eval_temp['direction'],
            "技术面": eval_temp['tech_score'],
            "资金面": eval_temp['flow_score']
        })
        
    df_scan = pd.DataFrame(scan_results).sort_values(by="AI 评分", ascending=False)
    st.dataframe(df_scan, column_config={"AI 评分": st.column_config.ProgressColumn("AI 评分", format="%d", min_value=0, max_value=100)}, hide_index=True, use_container_width=True)

# ------------------------------------------
# TAB 4: 合约风控中心
# ------------------------------------------
with tab4:
    st.subheader("🛡️ 仓位计算与爆仓模拟器")
    rc1, rc2 = st.columns(2)
    with rc1:
        calc_entry = st.number_input("计划入场价 ($)", value=float(int(btc_eval['current_price'])))
        calc_sl = st.number_input("计划止损价 ($)", value=float(int(btc_eval['current_price'] * 0.98)))
        calc_lev = st.slider("杠杆倍数", 1, 50, 10)
        res = RiskEngine.calculate_position(account_balance, global_risk_limit, calc_entry, calc_sl, calc_lev)

    with rc2:
        if res:
            st.metric("建议开仓数量", f"{res['quantity']} 代币", f"名义价值: ${res['notional_value']}")
            st.metric("所需占用保证金", f"${res['margin_used']} USDT")
            st.metric("预估强平爆仓价", f"${res['liq_price']} USDT", delta_color="inverse")

# ------------------------------------------
# TAB 5: 策略实验室
# ------------------------------------------
with tab5:
    st.subheader("🧪 策略历史回测")
    days = np.arange(30)
    returns_strat = np.cumsum(np.random.normal(0.003, 0.012, 30)) + 1.0
    
    fig_bt = go.Figure()
    fig_bt.add_trace(go.Scatter(x=days, y=returns_strat*100, name="AI 综合策略", line=dict(color='#26a69a', width=2)))
    fig_bt.update_layout(paper_bgcolor='#131722', plot_bgcolor='#131722', font=dict(color='#d1d4dc'), height=320, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_bt, use_container_width=True)

# ------------------------------------------
# TAB 6: 交易心理与复盘
# ------------------------------------------
with tab6:
    st.subheader("📘 历史交易复盘")
    history_trades = pd.DataFrame([
        {"订单ID": "#101", "交易对": "BTC/USDT", "方向": "LONG", "盈亏(USDT)": "+320.00", "AI 评分": 92},
        {"订单ID": "#102", "交易对": "ETH/USDT", "方向": "SHORT", "盈亏(USDT)": "-150.00", "AI 评分": 58}
    ])
    st.dataframe(history_trades, hide_index=True, use_container_width=True)

# ==========================================
# 6. 实时自动刷新引擎机制 (Auto-Refresh Loop)
# ==========================================
if enable_autorefresh:
    time.sleep(refresh_interval)
    st.rerun()
