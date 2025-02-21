import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import concurrent.futures
from tradingview_ta import TA_Handler, Interval

# List of symbols
symbols = [
    C
# Configuration
exchange = "BYBIT"
screener = "crypto"
interval = Interval.INTERVAL_4_HOURS

@st.cache_data(ttl=180)
def fetch_all_data():
    results = []
    current_datetime = datetime.now(timezone.utc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
        for future in concurrent.futures.as_completed(future_to_symbol):
            result = future.result()
            if result:
                result["Timestamp"] = current_datetime
                results.append(result)

    return pd.DataFrame(results)

def process_symbol(symbol):
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=interval,
            timeout=None
        )
        analysis = handler.get_analysis()
        rsi_value = analysis.indicators.get('RSI', None)
        return {"Symbol": symbol, "4h RSI": rsi_value}
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}")
        return None

def display_streamlit_app():
    st.set_page_config(page_title="Crypto Selector", layout="wide", initial_sidebar_state="expanded")

    # Custom CSS for dark mode
    st.markdown("""
    <style>
        .stApp {
            background-color: #1E1E1E;
            color: white;
        }
        .css-1d391kg {
            background-color: #2D2D2D;
        }
        .stAlert {
            background-color: #3D3D3D;
            border: 1px solid #4D4D4D;
            color: white;
        }
        .metric-card {
            background-color: #2D2D2D;
            border: 1px solid #3D3D3D;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
            color: white;
        }
        .stButton>button {
            color: white;
            background-color: #4CAF50;
            border: none;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title('📊 Crypto Selector')

    if st.button('Refresh Data'):
        st.experimental_rerun()

    df = fetch_all_data()

    # Sidebar
    st.sidebar.header('Dashboard Controls')
    st.sidebar.write(f"🕒 Last update: {df['Timestamp'].iloc[0].strftime('%Y-%m-%d %H:%M:%S %Z')}")
    
    rsi_range = st.sidebar.slider('RSI Range', 0, 100, (30, 70))
    symbols_to_show = st.sidebar.multiselect('Select Symbols', options=df['Symbol'].unique(), default=[])

    # Main dashboard area
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
        st.metric("Symbols in Range", len(df[(df['4h RSI'] >= rsi_range[0]) & (df['4h RSI'] <= rsi_range[1])]))
        st.markdown('</div>', unsafe_allow_html=True)

    # RSI Distribution
    st.subheader('📈 Distribution')
    fig_hist = px.histogram(df, x='4h RSI', nbins=50, 
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

    # Symbols in selected range
    st.subheader(f'🎯 Symbols in Range ({rsi_range[0]}-{rsi_range[1]})')
    df_in_range = df[(df['4h RSI'] >= rsi_range[0]) & (df['4h RSI'] <= rsi_range[1])].sort_values(by='4h RSI', ascending=False)
    if not df_in_range.empty:
        fig_range = px.scatter(df_in_range, x='Symbol', y='4h RSI', color='4h RSI', 
                               title=f'Symbols with RSI between {rsi_range[0]} and {rsi_range[1]}',
                               color_continuous_scale='Viridis')
        fig_range.update_traces(marker=dict(size=10))
        fig_range.update_layout(
            xaxis_tickangle=-45,
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font_color='white'
        )
        st.plotly_chart(fig_range, use_container_width=True)
        
        st.dataframe(df_in_range[['Symbol', '4h RSI']].style
                     .format({'4h RSI': '{:.2f}'})
                     .background_gradient(cmap='viridis', subset=['4h RSI']))
    else:
        st.info("No symbols found in the selected RSI range.")

    # Detailed view of selected symbols
    if symbols_to_show:
        st.subheader('🔍 Detailed Symbol View')
        df_selected = df[df['Symbol'].isin(symbols_to_show)]
        fig_selected = px.bar(df_selected, x='Symbol', y='4h RSI', color='4h RSI',
                              title='RSI Values for Selected Symbols',
                              color_continuous_scale='Viridis')
        fig_selected.add_hline(y=30, line_dash="dash", line_color="#FF4136", annotation_text="Oversold")
        fig_selected.add_hline(y=70, line_dash="dash", line_color="#2ECC40", annotation_text="Overbought")
        fig_selected.update_layout(
            plot_bgcolor='#1E1E1E',
            paper_bgcolor='#1E1E1E',
            font_color='white'
        )
        st.plotly_chart(fig_selected, use_container_width=True)

        st.dataframe(df_selected[['Symbol', '4h RSI']]
                     .style.format({'4h RSI': '{:.2f}'})
                     .background_gradient(cmap='viridis', subset=['4h RSI']))

if __name__ == "__main__":
    display_streamlit_app()
