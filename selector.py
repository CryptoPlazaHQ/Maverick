import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timezone
import time
import logging
from cachetools import TTLCache
import os
import concurrent.futures
import threading
from tradingview_ta import TA_Handler, Interval

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up caching
cache = TTLCache(maxsize=1000, ttl=300)  # Cache with 5-minute TTL

# CSV file path
CSV_FILE_PATH = "4h_rsi.csv"

# List of symbols
symbols = [
    "10000LADYSUSDT.P", "10000NFTUSDT.P", "1000BONKUSDT.P", "1000BTTUSDT.P", 
    "1000FLOKIUSDT.P", "1000LUNCUSDT.P", "1000PEPEUSDT.P", "1000XECUSDT.P", 
    "1INCHUSDT.P", "AAVEUSDT.P", "ACHUSDT.P", "ADAUSDT.P", "AGLDUSDT.P", 
    "AKROUSDT.P", "ALGOUSDT.P", "ALICEUSDT.P", "ALPACAUSDT.P", 
    "ALPHAUSDT.P", "AMBUSDT.P", "ANKRUSDT.P", "APEUSDT.P", "API3USDT.P", 
    "APTUSDT.P", "ARUSDT.P", "ARBUSDT.P", "ARKUSDT.P", 
    "ARKMUSDT.P", "ARPAUSDT.P", "ASTRUSDT.P", "ATAUSDT.P", "ATOMUSDT.P", 
    "AUCTIONUSDT.P", "AUDIOUSDT.P", "AVAXUSDT.P", "AXSUSDT.P", "BADGERUSDT.P", 
    "BAKEUSDT.P", "BALUSDT.P", "BANDUSDT.P", "BATUSDT.P", "BCHUSDT.P", 
    "BELUSDT.P", "BICOUSDT.P", "BIGTIMEUSDT.P", "BLURUSDT.P", "BLZUSDT.P",
    "BTCUSDT.P", "C98USDT.P", "CEEKUSDT.P", "CELOUSDT.P", "CELRUSDT.P", "CFXUSDT.P",
    "CHRUSDT.P", "CHZUSDT.P", "CKBUSDT.P", "COMBOUSDT.P", "COMPUSDT.P",
    "COREUSDT.P", "COTIUSDT.P", "CROUSDT.P", "CRVUSDT.P", "CTCUSDT.P",
    "CTKUSDT.P", "CTSIUSDT.P", "CVCUSDT.P", "CVXUSDT.P", "CYBERUSDT.P", "DARUSDT.P",
    "DASHUSDT.P", "DENTUSDT.P", "DGBUSDT.P", "DODOUSDT.P", "DOGEUSDT.P", "DOTUSDT.P",
    "DUSKUSDT.P", "DYDXUSDT.P", "EDUUSDT.P", "EGLDUSDT.P", "ENJUSDT.P", "ENSUSDT.P",
    "EOSUSDT.P", "ETCUSDT.P", "ETHUSDT.P", "ETHWUSDT.P", "FILUSDT.P",
    "FITFIUSDT.P", "FLOWUSDT.P", "FLRUSDT.P", "FORTHUSDT.P", "FRONTUSDT.P", "FTMUSDT.P",
    "FXSUSDT.P", "GALAUSDT.P", "GFTUSDT.P", "GLMUSDT.P",
    "GLMRUSDT.P", "GMTUSDT.P", "GMXUSDT.P", "GRTUSDT.P", "GTCUSDT.P", "HBARUSDT.P", 
    "HFTUSDT.P", "GMEUSDT.P", "HIFIUSDT.P", "HIGHUSDT.P", "HNTUSDT.P",
    "HOOKUSDT.P", "HOTUSDT.P", "ICPUSDT.P", "ICXUSDT.P", "IDUSDT.P", "IDEXUSDT.P",
    "ILVUSDT.P", "IMXUSDT.P", "INJUSDT.P", "IOSTUSDT.P", "IOTAUSDT.P", "IOTXUSDT.P",
    "JASMYUSDT.P", "JOEUSDT.P", "JSTUSDT.P", "KASUSDT.P", "KAVAUSDT.P", "KDAUSDT.P",
    "KEYUSDT.P", "KLAYUSDT.P", "KNCUSDT.P", "KSMUSDT.P", "LDOUSDT.P", "LEVERUSDT.P",
    "LINAUSDT.P", "LINKUSDT.P", "LITUSDT.P", "LOOKSUSDT.P", "LOOMUSDT.P", "LPTUSDT.P",
    "LQTYUSDT.P", "LRCUSDT.P", "LTCUSDT.P", "LUNA2USDT.P", "MAGICUSDT.P",
    "MANAUSDT.P", "MASKUSDT.P", "MATICUSDT.P", "MAVUSDT.P", "MDTUSDT.P",
    "MINAUSDT.P", "MKRUSDT.P", "MNTUSDT.P", "MTLUSDT.P", "NEARUSDT.P",
    "NEOUSDT.P", "NKNUSDT.P", "NMRUSDT.P", "NTRNUSDT.P", "OGUSDT.P",
    "OGNUSDT.P", "OMGUSDT.P", "ONEUSDT.P", "ONTUSDT.P", "OPUSDT.P", "ORBSUSDT.P",
    "ORDIUSDT.P", "OXTUSDT.P", "PAXGUSDT.P", "PENDLEUSDT.P", "PEOPLEUSDT.P", "PERPUSDT.P",
    "PHBUSDT.P", "PROMUSDT.P", "PONKEUSDT.P", "QNTUSDT.P", "QTUMUSDT.P", "RADUSDT.P", "RDNTUSDT.P", 
    "REEFUSDT.P", "RENUSDT.P", "REQUSDT.P", "RLCUSDT.P", "ROSEUSDT.P", 
    "RPLUSDT.P", "RSRUSDT.P", "RSS3USDT.P", "RUNEUSDT.P", "RVNUSDT.P",
    "SANDUSDT.P", "SCUSDT.P", "SCRTUSDT.P", "SEIUSDT.P", "SFPUSDT.P", "SHIB1000USDT.P",
    "SKLUSDT.P", "SLPUSDT.P", "SNXUSDT.P", "SOLUSDT.P", "SPELLUSDT.P", "SSVUSDT.P", 
    "STGUSDT.P", "STMXUSDT.P", "STORJUSDT.P", "STPTUSDT.P", "STXUSDT.P", "SUIUSDT.P", 
    "SUNUSDT.P", "SUSHIUSDT.P", "SWEATUSDT.P", "SXPUSDT.P",
    "TUSDT.P", "THETAUSDT.P", "TLMUSDT.P", "TOMIUSDT.P", "TONUSDT.P",
    "TRBUSDT.P", "TRUUSDT.P", "TRXUSDT.P", "TWTUSDT.P", "UMAUSDT.P", "UNFIUSDT.P",
    "UNIUSDT.P", "USDCUSDT.P", "VETUSDT.P", "VGXUSDT.P", "VRAUSDT.P",
    "WAVESUSDT.P", "WAXPUSDT.P", "WLDUSDT.P", "WOOUSDT.P", "XCNUSDT.P",
    "XEMUSDT.P", "XLMUSDT.P", "XMRUSDT.P", "XNOUSDT.P", "XRPUSDT.P", "XTZUSDT.P",
    "XVGUSDT.P", "XVSUSDT.P", "YFIUSDT.P", "YGGUSDT.P", "ZECUSDT.P", "ZENUSDT.P", "ZILUSDT.P", "ZRXUSDT.P"
]

# Configuration
exchange = "BYBIT"
screener = "crypto"
interval = Interval.INTERVAL_4_HOURS

def fetch_all_data(symbol, exchange, screener, interval):
    cache_key = f"{symbol}_{exchange}_{screener}_{interval}"
    if cache_key in cache:
        return cache[cache_key]
    
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange=exchange,
            screener=screener,
            interval=interval,
            timeout=None
        )
        analysis = handler.get_analysis()
        cache[cache_key] = analysis
        return analysis
    except Exception as e:
        logging.error(f"Error fetching data for {symbol} on {interval}: {str(e)}")
        return None

def process_symbol(symbol):
    analysis = fetch_all_data(symbol, exchange, screener, interval)
    
    if analysis:
        rsi_value = analysis.indicators.get('RSI', None)  # Fetch RSI value
        return {"Symbol": symbol, "4h RSI": rsi_value}
    return None

def update_csv():
    while True:
        try:
            current_datetime = datetime.now(timezone.utc)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                future_to_symbol = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
                results = []
                for future in concurrent.futures.as_completed(future_to_symbol):
                    result = future.result()
                    if result:
                        result["Timestamp"] = current_datetime
                        results.append(result)
            
            new_df = pd.DataFrame(results)
            
            # Reorder columns to ensure '4h RSI' is in the correct position
            columns = [col for col in new_df.columns if col != '4h RSI'] + ['4h RSI']
            new_df = new_df[columns]
            
            # Append the new data to the CSV file
            if os.path.exists(CSV_FILE_PATH):
                new_df.to_csv(CSV_FILE_PATH, mode='a', header=False, index=False)
            else:
                new_df.to_csv(CSV_FILE_PATH, index=False)
            
            logging.info(f"CSV updated at {current_datetime} with {len(results)} symbols.")
            
            # Sleep for 3 minutes before the next update
            time.sleep(180)
            
        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            time.sleep(600)  # Wait 10 minutes before retrying

@st.cache_data(ttl=180)
def load_and_process_data():
    df = pd.read_csv(CSV_FILE_PATH)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce', utc=True)
    df = df.dropna(subset=['Timestamp'])
    df['4h RSI'] = pd.to_numeric(df['4h RSI'], errors='coerce')
    return df

def get_latest_data(df):
    latest_timestamp = df['Timestamp'].max()
    latest_df = df[df['Timestamp'] == latest_timestamp].copy()
    latest_df['RSI Change'] = latest_df.groupby('Symbol')['4h RSI'].transform(lambda x: x.diff())
    return latest_df, latest_timestamp

def display_streamlit_app():
    st.set_page_config(page_title="Maverick", layout="wide", initial_sidebar_state="expanded")

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
        .stTextInput>div>div>input {
            color: white;
            background-color: #3D3D3D;
        }
        .stSelectbox>div>div>select {
            color: white;
            background-color: #3D3D3D;
        }
        .stMultiSelect>div>div>select {
            color: white;
            background-color: #3D3D3D;
        }
    </style>
    """, unsafe_allow_html=True)

    st.title('📊 Entry Selector')

    # Implement native Streamlit auto-refresh
    refresh_interval = 180  # 3 minutes in seconds
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()

    if time.time() - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = time.time()
        st.experimental_rerun()

    if os.path.exists(CSV_FILE_PATH):
        try:
            df = load_and_process_data()
            latest_df, latest_timestamp = get_latest_data(df)
            
            # Sidebar
            st.sidebar.header('Dashboard Controls')
            st.sidebar.write(f"🕒 Last update: {latest_timestamp.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            rsi_range = st.sidebar.slider('Range', 0, 100, (30, 70))
            symbols_to_show = st.sidebar.multiselect('Select Symbols', options=latest_df['Symbol'].unique(), default=[])

            # Main dashboard area
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Total Symbols", len(latest_df))
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Avg RSI", f"{latest_df['4h RSI'].mean():.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Symbols in Range", len(latest_df[(latest_df['4h RSI'] >= rsi_range[0]) & (latest_df['4h RSI'] <= rsi_range[1])]))
                st.markdown('</div>', unsafe_allow_html=True)

            # RSI Distribution
            st.subheader('📈 Distribution')
            fig_hist = px.histogram(latest_df, x='4h RSI', nbins=50, 
                                    title='Momentum Distribution',
                                    labels={'4h RSI': 'RSI Value', 'count': 'Number of Symbols'},
                                    color_discrete_sequence=['#8A2BE2'])  # Dark Purple
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
            df_in_range = latest_df[(latest_df['4h RSI'] >= rsi_range[0]) & (latest_df['4h RSI'] <= rsi_range[1])].sort_values(by='4h RSI', ascending=False)
            if not df_in_range.empty:
                fig_range = px.scatter(df_in_range, x='Symbol', y='4h RSI', color='4h RSI', 
                                       hover_data=['RSI Change'],
                                       title=f'Symbols in Momentum between {rsi_range[0]} and {rsi_range[1]}',
                                       color_continuous_scale='Viridis')
                fig_range.update_traces(marker=dict(size=10))
                fig_range.update_layout(
                    xaxis_tickangle=-45,
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font_color='white'
                )
                st.plotly_chart(fig_range, use_container_width=True)
                
                st.dataframe(df_in_range[['Symbol', '4h RSI', 'RSI Change']].style
                             .format({'4h RSI': '{:.2f}', 'RSI Change': '{:.2f}'})
                             .background_gradient(cmap='viridis', subset=['4h RSI']))
            else:
                st.info("No symbols found in the selected RSI range.")

            # Detailed view of selected symbols
            if symbols_to_show:
                st.subheader('🔍 Detailed Symbol View')
                df_selected = latest_df[latest_df['Symbol'].isin(symbols_to_show)]
                fig_selected = px.bar(df_selected, x='Symbol', y='4h RSI', color='4h RSI',
                                      title='Momentum Values for Selected Symbols',
                                      color_continuous_scale='Viridis')
                fig_selected.add_hline(y=30, line_dash="dash", line_color="#FF4136", annotation_text="Ready to Buy?")
                fig_selected.add_hline(y=70, line_dash="dash", line_color="#2ECC40", annotation_text="Ready to Short?")
                fig_selected.update_layout(
                    plot_bgcolor='#1E1E1E',
                    paper_bgcolor='#1E1E1E',
                    font_color='white'
                )
                st.plotly_chart(fig_selected, use_container_width=True)

                # Detailed table for selected symbols
                st.dataframe(df_selected[['Symbol', '4h RSI', 'RSI Change']]
                             .style.format({'4h RSI': '{:.2f}', 'RSI Change': '{:.2f}'})
                             .background_gradient(cmap='viridis', subset=['4h RSI']))

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("CSV file not found. Please ensure the data collection process is running.")

if __name__ == "__main__":
    # Start the CSV update thread
    threading.Thread(target=update_csv, daemon=True).start()
    
    # Run the Streamlit app
    display_streamlit_app()