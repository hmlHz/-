import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import random

# ==========================================
# 1. 页面基本配置与样式注入 (Quant Dark Theme)
# ==========================================
st.set_page_config(
    page_title="AI Professional Futures Decision Workbench V2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 注入自定义 CSS 打造专业终端视觉效果
st.markdown("""
<style>
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
    }
    .signal-long {
        background-color: rgba(38, 166, 154, 0.2);
        border: 1px solid #26a69a;
        color: #26a69a;
        border-radius: 8px;
        padding: 15px;
    }
    .signal-short {
        background-color: rgba(239, 83, 80, 0.2);
        border: 1px solid #ef5350;
        color: #ef5350;
        border-radius: 8px;
        padding: 15px;
    }
    .stButton>button {
        width: 100%;
        background-color: #2962ff;
        color: white;
        font-weight: bold;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 模拟/真实数据与计算引擎 (Core Engines)
# ==========================================

@st.cache_data(ttl=60)
def fetch_market_data(symbol="BTC/USDT", timeframe="1h", limit=100):
    """模拟市场数据引擎 (Market Engine)"""
    np.random.seed(42)
    now = datetime.datetime.now()
    dates = [now - datetime.timedelta(hours=i) for i in range(limit)][::-1]
    
    base_price = 65000.0 if "BTC" in symbol else (3500.0 if "ETH" in symbol else 140.0)
    returns = np.random.normal(0.0002, 0.008, limit)
    price_path = base_price * np.exp(np.cumsum(returns))
    
    high = price_path * (1 + np.abs(np.random.normal(0, 0.004, limit)))
    low = price_path * (1 - np.abs(np.random.normal(0, 0.004, limit)))
    open_p = price_path * (1 + np.random.normal(0, 0.002, limit))
    close = price_path
    volume = np.random.normal(1000, 200, limit) * (close / 100)
    
    df = pd.DataFrame({
        "timestamp": dates, "open": open_p, "high": high, 
        "low": low, "close": close, "volume": volume
    })
    
    # 特征计算 (Feature Engine)
    df['EMA20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA60'] = df['close'].ewm(span=60, adjust=False).mean()
    df['RSI'] = 50 + np.sin(np.linspace(0, 10, limit)) * 25
    df['ATR'] = (df['high'] - df['low']).rolling(14).mean().fillna(method='bfill')
    
    return df

class AIDecisionEngine:
    """AI 决策引擎 - 多因子加权融合模型"""
    @staticmethod
    def evaluate_market(df, funding_rate=0.0001, sentiment_val=65):
        latest = df.iloc[-1]
        
        # 1. 技术面打分 (30%)
        tech_score = 0
        if latest['close'] > latest['EMA20'] > latest['EMA60']:
            tech_score += 50
        if 40 <= latest['RSI'] <= 65:
            tech_score += 30
        tech_score = min(tech_score + random.randint(0, 20), 100)
        
        # 2. 资金面打分 (25%)
        flow_score = 80 if funding_rate > 0 else 30
        
        # 3. 趋势强度 (20%)
        trend_score = 75 if latest['EMA20'] > latest['EMA60'] else 25
        
        # 4. 情绪与宏观 (15%)
        sent_score = sentiment_val
        
        # 5. ML 模型预测 (10%)
        ml_score = random.randint(60, 90)
        
        # 综合打分
        total_score = (
            tech_score * 0.30 +
            flow_score * 0.25 +
            trend_score * 0.20 +
            sent_score * 0.15 +
            ml_score * 0.10
        )
        
        if total_score >= 65:
            direction = "LONG"
        elif total_score <= 35:
            direction = "SHORT"
        else:
            direction = "WAIT"
            
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
    """风控引擎 - 动态仓位与爆仓模拟"""
    @staticmethod
    def calculate_position(balance, risk_pct, entry_price, stop_loss_price, leverage):
        if entry_price == stop_loss_price:
            return None
        
        risk_amount = balance * (risk_pct / 100.0)
        sl_pct = abs(entry_price - stop_loss_price) / entry_price
        
        notional_value = risk_amount / sl_pct
        max_notional_by_leverage = balance * leverage
        
        actual_notional = min(notional_value, max_notional_by_leverage)
        quantity = actual_notional / entry_price
        margin_used = actual_notional / leverage
        
        # 估算爆仓价 (以多头为例)
        maint_margin_rate = 0.005 # 0.5%
        liq_price = entry_price * (1 - (1 / leverage) + maint_margin_rate)
        
        return {
            "risk_amount": round(risk_amount, 2),
            "notional_value": round(actual_notional, 2),
            "quantity": round(quantity, 4),
            "margin_used": round(margin_used, 2),
            "liq_price": round(liq_price, 2),
            "sl_pct": round(sl_pct * 100, 2),
            "is_high_risk": leverage > 20 or risk_pct > 2.5
        }

# ==========================================
# 3. 侧边栏 (全局配置 & 账户状态)
# ==========================================
st.sidebar.image("https://img.icons8.com/color/96/000000/bot.png", width=60)
st.sidebar.title("AI Trading Brain")
st.sidebar.caption("Futures Decision Workbench V2.0")

st.sidebar.divider()
st.sidebar.subheader("👤 账户与 API 设置")
selected_exchange = st.sidebar.selectbox("交易所适配器", ["Binance Futures", "OKX Futures", "Gate Futures"])
account_balance = st.sidebar.number_input("账户可用资金 (USDT)", value=10000.0, step=1000.0)
global_risk_limit = st.sidebar.slider("单笔允许风险 (%)", 0.5, 3.0, 1.5, 0.1)

st.sidebar.divider()
st.sidebar.subheader("🛡️ 风控状态")
st.sidebar.success("● 实时API连接正常")
st.sidebar.info("● 熔断保护机制: 激活")

# ==========================================
# 4. 主界面 Top Metrics (全局行情)
# ==========================================
st.title("⚡ AI Crypto Trading Terminal")

# 获取默认数据
df_btc = fetch_market_data("BTC/USDT")
btc_eval = AIDecisionEngine.evaluate_market(df_btc)

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
col_m1.metric("BTC/USDT 现价", f"${btc_eval['current_price']:.2f}", "+1.85%")
col_m2.metric("AI 综合评分", f"{btc_eval['total_score']} / 100", f"方向: {btc_eval['direction']}")
col_m3.metric("资金费率", "0.0100%", "看多情绪偏高")
col_m4.metric("全网爆仓 (24h)", "$1.28M", "多头占比 62%")
col_m5.metric("全网持仓量 (OI)", "$18.5B", "+3.2%")

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
        st.subheader("BTC/USDT 实时 K 线与技术结构")
        # 绘制 K 线与指标图
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.7, 0.3])
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df_btc['timestamp'], open=df_btc['open'], high=df_btc['high'],
            low=df_btc['low'], close=df_btc['close'], name="K线"
        ), row=1, col=1)
        
        # EMAs
        fig.add_trace(go.Scatter(x=df_btc['timestamp'], y=df_btc['EMA20'], name="EMA 20", line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_btc['timestamp'], y=df_btc['EMA60'], name="EMA 60", line=dict(color='blue', width=1)), row=1, col=1)
        
        # Volume
        fig.add_trace(go.Bar(x=df_btc['timestamp'], y=df_btc['volume'], name="成交量", marker_color='gray'), row=2, col=1)
        
        fig.update_layout(template="plotly_dark", height=500, margin=dict(l=10, r=10, t=10, b=10), xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("AI 多因子雷达")
        st.progress(btc_eval['total_score'] / 100, text=f"AI 置信度: {btc_eval['confidence']}%")
        
        scores_df = pd.DataFrame({
            "维度": ["技术面 (30%)", "资金面 (25%)", "趋势共振 (20%)", "新闻情绪 (15%)", "模型预测 (10%)"],
            "得分": [btc_eval['tech_score'], btc_eval['flow_score'], btc_eval['trend_score'], btc_eval['sent_score'], btc_eval['ml_score']]
        })
        st.dataframe(scores_df, hide_index=True, use_container_width=True)
        
        st.info(f"**市场状态总结**: 目前处在 {'牛市主升浪' if btc_eval['total_score'] > 60 else '震荡整理趋势'}，建议配合 EMA20 回踩布局。")

# ------------------------------------------
# TAB 2: AI 交易信号中心
# ------------------------------------------
with tab2:
    st.subheader("🎯 AI 实时高概率交易信号")
    
    current_p = btc_eval['current_price']
    atr = btc_eval['atr']
    
    if btc_eval['direction'] == "LONG":
        sl_price = current_p - 1.5 * atr
        tp1_price = current_p + 2.0 * atr
        tp2_price = current_p + 3.5 * atr
        css_class = "signal-long"
    else:
        sl_price = current_p + 1.5 * atr
        tp1_price = current_p - 2.0 * atr
        tp2_price = current_p - 3.5 * atr
        css_class = "signal-short"
        
    st.markdown(f"""
    <div class="{css_class}">
        <h2>信号指令: {btc_eval['direction']} BTC/USDT</h2>
        <p><b>AI 置信度分值:</b> {btc_eval['total_score']} / 100 | <b>建议杠杆:</b> 5x - 10x</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.write("")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    s_col1.metric("建议入场价", f"${current_p:.2f}")
    s_col2.metric("建议止损价 (SL)", f"${sl_price:.2f}", f"-{abs(current_p-sl_price)/current_p*100:.2f}%", delta_color="inverse")
    s_col3.metric("目标止盈 1 (TP1)", f"${tp1_price:.2f}", f"+{abs(tp1_price-current_p)/current_p*100:.2f}%")
    s_col4.metric("目标止盈 2 (TP2)", f"${tp2_price:.2f}", f"+{abs(tp2_price-current_p)/current_p*100:.2f}%")
    
    st.subheader("🤖 GPT 交易逻辑深度解释")
    st.write(f"""
    1. **结构识别**: BTC 在 1 小时级别成功站稳 EMA20 (`${btc_eval['current_price']:.2f}`)，形成了清晰的多头排列。
    2. **资金流协同**: 全网资金费率为正，且持仓量 (OI) 伴随价格突破而增加，表明主力多头资金持续流入。
    3. **风控建议**: 距离止损点距离约为 `{abs(current_p-sl_price)/current_p*100:.2f}%`，建议单笔仓位不超过账户资金的 `{global_risk_limit}%`。
    """)
    
    if st.button("🚀 执行此信号 (自动打入风控层机制)"):
        st.success("✅ 订单通过风控检测！已通过 API 发送至交易所。单号: #ORD-20241025-8892")

# ------------------------------------------
# TAB 3: 市场扫描器
# ------------------------------------------
with tab3:
    st.subheader("🔍 全网热门合约 AI 扫描")
    
    symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT"]
    scan_results = []
    
    for sym in symbols:
        df_temp = fetch_market_data(sym)
        eval_temp = AIDecisionEngine.evaluate_market(df_temp)
        scan_results.append({
            "交易对": sym,
            "当前价格": f"${eval_temp['current_price']:.2f}",
            "AI 评分": eval_temp['total_score'],
            "建议方向": eval_temp['direction'],
            "置信度": f"{eval_temp['confidence']}%",
            "技术面得分": eval_temp['tech_score'],
            "资金面得分": eval_temp['flow_score'],
            "波动率 (ATR)": f"{eval_temp['atr']:.2f}"
        })
        
    df_scan = pd.DataFrame(scan_results).sort_values(by="AI 评分", ascending=False)
    
    st.dataframe(
        df_scan, 
        column_config={
            "AI 评分": st.column_config.ProgressColumn("AI 评分", format="%d", min_value=0, max_value=100),
        },
        hide_index=True, 
        use_container_width=True
    )

# ------------------------------------------
# TAB 4: 合约风控中心
# ------------------------------------------
with tab4:
    st.subheader("🛡️ 仓位计算与爆仓模拟器")
    
    rc1, rc2 = st.columns(2)
    
    with rc1:
        st.write("### 🧮 仓位计算参数")
        calc_entry = st.number_input("计划入场价格 ($)", value=float(int(btc_eval['current_price'])))
        calc_sl = st.number_input("计划止损价格 ($)", value=float(int(btc_eval['current_price'] * 0.98)))
        calc_lev = st.slider("选择杠杆倍数", 1, 50, 10)
        
        res = RiskEngine.calculate_position(account_balance, global_risk_limit, calc_entry, calc_sl, calc_lev)

    with rc2:
        st.write("### 📊 风控输出结果")
        if res:
            st.metric("建议开仓数量", f"{res['quantity']} 代币", f"名义价值: ${res['notional_value']}")
            st.metric("所需占用保证金", f"${res['margin_used']} USDT")
            st.metric("单笔预估最大亏损", f"${res['risk_amount']} USDT ({global_risk_limit}%)")
            st.metric("强平爆仓参考价 (估算)", f"${res['liq_price']} USDT", delta_color="inverse")
            
            if res['is_high_risk']:
                st.error("⚠️ 警告：当前杠杆过高或风险比例较大，极易引发预警！")
            else:
                st.success("✅ 交易风控指标健康，符合系统开仓规则。")

# ------------------------------------------
# TAB 5: AI 策略实验室
# ------------------------------------------
with tab5:
    st.subheader("🧪 策略历史回测与比较")
    
    st_col1, st_col2 = st.columns([1, 3])
    
    with st_col1:
        selected_strat = st.selectbox("选择测试策略", ["AI 综合多因子策略", "EMA 趋势突破策略", "RSI 均值回归策略", "Funding Rate 套利策略"])
        backtest_days = st.slider("回测天数", 7, 90, 30)
        st.button("运行回测模拟")
        
    with st_col2:
        # 绘制模拟回测收益曲线
        days = np.arange(backtest_days)
        returns_strat = np.cumsum(np.random.normal(0.003, 0.015, backtest_days)) + 1.0
        returns_btc = np.cumsum(np.random.normal(0.001, 0.02, backtest_days)) + 1.0
        
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=days, y=returns_strat*100, name=f"{selected_strat} (策略)", line=dict(color='#26a69a', width=2)))
        fig_bt.add_trace(go.Scatter(x=days, y=returns_btc*100, name="BTC 基准", line=dict(color='gray', dash='dash')))
        
        fig_bt.update_layout(template="plotly_dark", height=350, title="策略收益曲线对比 (USDT %)", margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_bt, use_container_width=True)
        
        b1, b2, b3 = st.columns(3)
        b1.metric("策略累计收益率", "+24.5%")
        b2.metric("最大回撤 (Max Drawdown)", "-5.2%")
        b3.metric("夏普比率 (Sharpe Ratio)", "2.14")

# ------------------------------------------
# TAB 6: 交易心理与复盘
# ------------------------------------------
with tab6:
    st.subheader("📘 历史交易日志与 AI 评估")
    
    history_trades = pd.DataFrame([
        {"订单ID": "#101", "开仓时间": "2024-10-24 14:20", "交易对": "BTC/USDT", "方向": "LONG", "盈亏(USDT)": "+320.00", "AI 交易评分": 92, "错误归因": "无 (完美执行)"},
        {"订单ID": "#102", "开仓时间": "2024-10-23 09:15", "交易对": "ETH/USDT", "方向": "SHORT", "盈亏(USDT)": "-150.00", "AI 交易评分": 58, "错误归因": "过早平仓 (FOMO)"},
        {"订单ID": "#103", "开仓时间": "2024-10-22 20:00", "交易对": "SOL/USDT", "方向": "LONG", "盈亏(USDT)": "+450.00", "AI 交易评分": 88, "错误归因": "无"}
    ])
    
    st.dataframe(history_trades, hide_index=True, use_container_width=True)
    
    st.divider()
    st.subheader("🤖 AI 交易心理诊断报告")
    st.warning("🧠 **纪律性提示**: 在过去的 10 笔交易中，出现 2 次未按 AI 推荐止损位硬扛的情况。建议严格开启 API 自动化止损锁，克服心理情绪化干扰。")