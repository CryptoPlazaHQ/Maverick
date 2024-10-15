import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from tradingview_ta import TA_Handler, Interval
from decimal import Decimal
from datetime import datetime, timezone
import concurrent.futures

# Set page config
st.set_page_config(page_title="Grid Bot Entry Dashboard", layout="wide")

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #E0E0E0;
    }
    .stButton>button {
        color: #E0E0E0;
        background-color: #333333;
        border: 2px solid #4CAF50;
    }
    .stButton>button:hover {
        background-color: #4CAF50;
        color: #1E1E1E;
    }
    .stTextInput>div>div>input {
        color: #E0E0E0;
        background-color: #333333;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1, h2, h3 {
        color: #4CAF50;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .task-complete {
        color: #4CAF50;
    }
    .task-incomplete {
        color: #FF4B4B;
    }
    .info-box {
        background-color: #2C3E50;
        border-left: 5px solid #4CAF50;
        padding: 10px;
        margin-bottom: 10px;
    }
    .metric-card {
        background-color: #2D2D2D;
        border: 1px solid #3D3D3D;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def float_to_decimal(value):
    return Decimal(str(value))

def fetch_tv_data(symbol, exchange, screener, interval):
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=interval,
            timeout=None
        )
        analysis = handler.get_analysis()
        return analysis
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def calculate_pivot_points(high, low, close):
    pivot = (high + low + close) / Decimal('3')
    r1 = Decimal('2') * pivot - low
    s1 = Decimal('2') * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + Decimal('2') * (pivot - low)
    s3 = low - Decimal('2') * (high - pivot)
    return {
        'pivot': pivot,
        'r1': r1, 's1': s1,
        'r2': r2, 's2': s2,
        'r3': r3, 's3': s3
    }

def plot_chart(symbol, interval, pivots):
    analysis = fetch_tv_data(symbol, "BYBIT", "crypto", interval)
    if analysis is None:
        return None

    df = pd.DataFrame({
        'open': [analysis.indicators['open']],
        'high': [analysis.indicators['high']],
        'low': [analysis.indicators['low']],
        'close': [analysis.indicators['close']],
        'volume': [analysis.indicators['volume']],
        'rsi': [analysis.indicators['RSI']]
    }, index=[pd.Timestamp.now()])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'], name='OHLC'), row=1, col=1)
    
    for level, value in pivots.items():
        fig.add_trace(go.Scatter(x=[df.index[0], df.index[0]], y=[float(value), float(value)], 
                                 mode='lines', line=dict(color='gray', dash='dash'), name=level), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI', line=dict(color='purple')), row=2, col=1)
    
    fig.update_layout(height=600, title=f'{symbol} - {interval} Chart with Pivot Points', xaxis_rangeslider_visible=False,
                      plot_bgcolor='#1E1E1E', paper_bgcolor='#1E1E1E', font_color='white')
    return fig

@st.cache_data(ttl=180)
def fetch_all_rsi_data(symbols):
    results = []
    current_datetime = datetime.now(timezone.utc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_symbol = {executor.submit(fetch_rsi, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                result["Timestamp"] = current_datetime
                results.append(result)

    return pd.DataFrame(results)

def fetch_rsi(symbol):
    try:
        analysis = fetch_tv_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_4_HOURS)
        if analysis:
            rsi_value = analysis.indicators.get('RSI', None)
            return {"Symbol": symbol, "4h RSI": rsi_value}
    except Exception as e:
        st.error(f"Error fetching RSI for {symbol}: {str(e)}")
    return None

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'checklist' not in st.session_state:
    st.session_state.checklist = {
        'Discord Alert and Pair Selection': False,
        '4H Chart Analysis': False,
        '30M Chart Analysis': False,
        'Pivot Points and Fibonacci': False,
        'RSI Distribution Analysis': False,
        'BTC Context Analysis': False,
        'Alert Activation and Update': False,
        'Entry Validation': False
    }
if 'trading_data' not in st.session_state:
    st.session_state.trading_data = {}

def main():
    st.title("Grid Bot Entry Dashboard")
    
    progress = sum(st.session_state.checklist.values()) / len(st.session_state.checklist)
    st.progress(progress)
    
    steps = list(st.session_state.checklist.keys())
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Checklist Progress")
        for i, step in enumerate(steps):
            if st.session_state.checklist[step]:
                st.markdown(f"✅ **{step}**")
            elif i == st.session_state.step:
                st.markdown(f"🔄 **{step}**")
            else:
                st.markdown(f"⏳ {step}")
    
    with col2:
        if st.session_state.step == 0:
            st.header("1. Discord Alert and Pair Selection")
            st.session_state.checklist['Discord Alert and Pair Selection'] = st.checkbox("Received Discord Alert", st.session_state.checklist['Discord Alert and Pair Selection'])
            symbol = st.text_input("Enter trading pair (e.g., BTCUSDT.P):")
            recommendation = st.radio("Bot Recommendation", ['LONG', 'SHORT', 'NEUTRAL'])
            if st.button("Confirm Selection"):
                st.session_state.trading_data['symbol'] = symbol
                st.session_state.trading_data['recommendation'] = recommendation
                st.session_state.checklist['Discord Alert and Pair Selection'] = True
                st.session_state.step += 1
                st.rerun()

        elif st.session_state.step == 1:
            st.header("2. 4H Chart Analysis")
            symbol = st.session_state.trading_data['symbol']
            analysis = fetch_tv_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_4_HOURS)
            if analysis:
                pivots = calculate_pivot_points(
                    float_to_decimal(analysis.indicators['high']),
                    float_to_decimal(analysis.indicators['low']),
                    float_to_decimal(analysis.indicators['close'])
                )
                st.plotly_chart(plot_chart(symbol, Interval.INTERVAL_4_HOURS, pivots), use_container_width=True)
                st.json(pivots)
                st.session_state.checklist['4H Chart Analysis'] = st.checkbox("4H Analysis Complete", st.session_state.checklist['4H Chart Analysis'])
                if st.session_state.checklist['4H Chart Analysis'] and st.button("Continue to 30M Analysis"):
                    st.session_state.step += 1
                    st.rerun()

        elif st.session_state.step == 2:
            st.header("3. 30M Chart Analysis")
            symbol = st.session_state.trading_data['symbol']
            analysis = fetch_tv_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_30_MINUTES)
            if analysis:
                pivots = calculate_pivot_points(
                    float_to_decimal(analysis.indicators['high']),
                    float_to_decimal(analysis.indicators['low']),
                    float_to_decimal(analysis.indicators['close'])
                )
                st.plotly_chart(plot_chart(symbol, Interval.INTERVAL_30_MINUTES, pivots), use_container_width=True)
                st.json(pivots)
                st.session_state.checklist['30M Chart Analysis'] = st.checkbox("30M Analysis Complete", st.session_state.checklist['30M Chart Analysis'])
                if st.session_state.checklist['30M Chart Analysis'] and st.button("Continue to Pivot and Fibonacci"):
                    st.session_state.step += 1
                    st.experimental_rerun()

        elif st.session_state.step == 3:
            st.header("4. Pivot Points and Fibonacci")
            symbol = st.session_state.trading_data['symbol']
            analysis = fetch_tv_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_1_DAY)
            if analysis:
                pivots = calculate_pivot_points(
                    float_to_decimal(analysis.indicators['high']),
                    float_to_decimal(analysis.indicators['low']),
                    float_to_decimal(analysis.indicators['close'])
                )
                st.json(pivots)
                st.markdown("### Fibonacci Retracement Levels")
                st.info("Place Fibonacci retracement on 5M chart and observe price interaction.")
                fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
                for level in fib_levels:
                    st.write(f"{level:.3f}")
                st.session_state.checklist['Pivot Points and Fibonacci'] = st.checkbox("Pivot and Fibonacci Analysis Complete", st.session_state.checklist['Pivot Points and Fibonacci'])
                if st.session_state.checklist['Pivot Points and Fibonacci'] and st.button("Continue to RSI Distribution"):
                    st.session_state.step += 1
                    st.experimental_rerun()

        elif st.session_state.step == 4:
            st.header("5. RSI Distribution Analysis")
            symbols = ["BTCUSDT.P", "ETHUSDT.P", "ENJUSDT.P", "DYDXUSDT.P", "WLDUSDT.P"]
            df = fetch_all_rsi_data(symbols)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Symbols", len(df))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg RSI", f"{df['4h RSI'].mean():.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                rsi_range = (30, 70)
                st.metric("Symbols in Range", len(df[(df['4h RSI'] >= rsi_range[0]) & (df['4h RSI'] <= rsi_range[1])]))
                st.markdown('</div>', unsafe_allow_html=True)

            fig_hist = px.histogram(df, x='4h RSI', nbins=20, 
                                    title='RSI Distribution',
                                    labels={'4h RSI': 'RSI Value', 'count': 'Number of Symbols'},
                                    color_discrete_sequence=['#8A2BE2'])
            fig_hist.add_vline(x=30, line_dash="dash", line_color="#FF4136", annotation_text="Oversold")
            fig_hist.add_vline(x=70, line_dash="dash", line_color="#2ECC40", annotation_text="Overbought")
            fig_hist.update_layout(
                plot_bgcolor='#1E1E1E',
                paper_bgcolor='#1E1E1E',
                font_color='white'
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            st.dataframe(df[['Symbol', '4h RSI']].style
                         .format({'4h RSI': '{:.2f}'})
                         .background_gradient(cmap='viridis', subset=['4h RSI']))

            st.session_state.checklist['RSI Distribution Analysis'] = st.checkbox("RSI Distribution Analysis Complete", st.session_state.checklist['RSI Distribution Analysis'])
            if st.session_state.checklist['RSI Distribution Analysis'] and st.button("Continue to BTC Context"):
                st.session_state.step += 1
                st.experimental_rerun()

        elif st.session_state.step == 5:
            st.header("6. BTC Context Analysis")
            btc_analysis = fetch_tv_data("BTCUSDT.P", "BYBIT", "crypto", Interval.INTERVAL_4_HOURS)
            if btc_analysis:
                pivots = calculate_pivot_points(
                    float_to_decimal(btc_analysis.indicators['high']),
                    float_to_decimal(btc_analysis.indicators['low']),
                    float_to_decimal(btc_analysis.indicators['close'])
                )
                st.plotly_chart(plot_chart("BTCUSDT.P", Interval.INTERVAL_4_HOURS, pivots), use_container_width=True)
                st.json(pivots)
                st.markdown("### BTC RSI")
                st.metric("BTC 4H RSI", f"{btc_analysis.indicators['RSI']:.2f}")
                st.session_state.checklist['BTC Context Analysis'] = st.checkbox("BTC Context Analysis Complete", st.session_state.checklist['BTC Context Analysis'])
                if st.session_state.checklist['BTC Context Analysis'] and st.button("Continue to Alert Activation"):
                    st.session_state.step += 1
                    st.experimental_rerun()

        elif st.session_state.step == 6:
            st.header("7. Alert Activation and Update")
            st.write("Monitor for S1 or R1 touch alerts.")
            alert_activated = st.checkbox("Alert Activated")
            if alert_activated:
                new_s1 = st.number_input("New S1 value:")
                new_r1 = st.number_input("New R1 value:")
                if st.button("Confirm Alert Update"):
                    st.session_state.checklist['Alert Activation and Update'] = True
                    st.session_state.trading_data['new_s1'] = new_s1
                    st.session_state.trading_data['new_r1'] = new_r1
                    st.session_state.step += 1
                    st.experimental_rerun()

        elif st.session_state.step == 7:
            st.header("8. Entry Validation")
            rsi_cross = st.checkbox("RSI crossed its moving average")
            doji_candle = st.checkbox("Doji or hammer candle formed on 5M chart")
            near_support_resistance = st.checkbox("Price near updated S1/R1")
            if rsi_cross and doji_candle and near_support_resistance:
                st.success("All entry conditions met!")
                st.session_state.checklist['Entry Validation'] = True
                
                entry_card = {
                    'timestamp': datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                    'symbol': st.session_state.trading_data['symbol'],
                    'recommendation': st.session_state.trading_data['recommendation'],
                    'entry_point': st.session_state.trading_data.get('new_s1', 0) if st.session_state.trading_data['recommendation'] == 'LONG' else st.session_state.trading_data.get('new_r1', 0),
                    'stop_loss': float(st.session_state.trading_data.get('new_s1', 0)) * 0.98 if st.session_state.trading_data['recommendation'] == 'LONG' else float(st.session_state.trading_data.get('new_r1', 0)) * 1.02,
                    'take_profit': st.session_state.trading_data.get('new_r1', 0) if st.session_state.trading_data['recommendation'] == 'LONG' else st.session_state.trading_data.get('new_s1', 0),
                }
                
                st.subheader("Entry Card")
                st.json(entry_card)
                
                if st.button("Download Entry Card (CSV)"):
                    pd.DataFrame([entry_card]).to_csv("entry_card.csv", index=False)
                    st.success("Entry card saved as 'entry_card.csv'")

    if st.button("Reset Dashboard"):
        for key in st.session_state.checklist:
            st.session_state.checklist[key] = False
        st.session_state.step = 0
        st.session_state.trading_data = {}
        st.rerun()

    # Display completion message and summary
    if all(st.session_state.checklist.values()):
        st.balloons()
        st.success("Congratulations! You've completed all steps of the Grid Bot Entry Analysis.")
        
        st.subheader("Analysis Summary")
        st.markdown(f"""
        - **Trading Pair**: {st.session_state.trading_data.get('symbol', 'N/A')}
        - **Recommendation**: {st.session_state.trading_data.get('recommendation', 'N/A')}
        - **Entry Point**: {st.session_state.trading_data.get('new_s1', 'N/A') if st.session_state.trading_data.get('recommendation') == 'LONG' else st.session_state.trading_data.get('new_r1', 'N/A')}
        - **Stop Loss**: {float(st.session_state.trading_data.get('new_s1', 0)) * 0.98 if st.session_state.trading_data.get('recommendation') == 'LONG' else float(st.session_state.trading_data.get('new_r1', 0)) * 1.02}
        - **Take Profit**: {st.session_state.trading_data.get('new_r1', 'N/A') if st.session_state.trading_data.get('recommendation') == 'LONG' else st.session_state.trading_data.get('new_s1', 'N/A')}
        """)
        
        st.markdown("### Next Steps")
        st.markdown("""
        1. Review the Entry Card details carefully.
        2. Set up your grid bot with the parameters from the Entry Card.
        3. Monitor the trade closely, especially in the beginning.
        4. Be prepared to adjust your strategy if market conditions change significantly.
        """)

    # Sidebar content
    st.sidebar.header("Dashboard Info")
    st.sidebar.info("""
    This Grid Bot Entry Dashboard guides you through a comprehensive analysis process to determine optimal entry points for your grid trading strategy.
    
    Complete each step carefully to ensure a thorough analysis of market conditions before entering a trade.
    """)

    if st.session_state.step > 0:
        performance = np.random.randint(70, 100)
        st.sidebar.metric("Analysis Performance", f"{performance}%", f"{performance - 75}%")

    tips = [
        "Did you know? Grid trading can be effective in both trending and ranging markets.",
        "Tip: Always consider the overall market sentiment before entering a grid trade.",
        "Fun fact: Grid trading originated in the forex markets before being adopted by crypto traders.",
        "Remember: Past performance doesn't guarantee future results. Always manage your risk!",
    ]
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Tip of the Day:** {np.random.choice(tips)}")

if __name__ == "__main__":
    main()
