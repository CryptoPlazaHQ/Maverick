import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timezone
import concurrent.futures
from tradingview_ta import TA_Handler, Interval

# List of symbols
symbols = [
    "10000000AIDOGEUSDT.P", "1000000BABYDOGEUSDT.P", "1000000CHEEMSUSDT.P", "1000000MOGUSDT.P", "1000000PEIPEIUSDT.P",
    "10000COQUSDT.P", "10000ELONUSDT.P", "10000LADYSUSDT.P", "10000QUBICUSDT.P", "10000SATSUSDT.P", "10000WENUSDT.P",
    "10000WHYUSDT.P", "1000APUUSDT.P", "1000BONKPERP.P", "1000BONKUSDT.P", "1000BTTUSDT.P", "1000CATSUSDT.P", "1000CATUSDT.P",
    "1000FLOKIUSDT.P", "1000LUNCUSDT.P", "1000MUMUUSDT.P", "1000NEIROCTOUSDT.P", "1000PEPEPERP.P", "1000PEPEUSDT.P",
    "1000RATSUSDT.P", "1000TOSHIUSDT.P", "1000TURBOUSDT.P", "1000XECUSDT.P", "1000XUSDT.P", "1INCHUSDT.P", "A8USDT.P",
    "AAVEUSDT.P", "ACEUSDT.P", "ACHUSDT.P", "ACTUSDT.P", "ACXUSDT.P", "ADAUSDT.P", "AERGOUSDT.P", "AEROUSDT.P", "AEVOPERP.P",
    "AEVOUSDT.P", "AGIUSDT.P", "AGLDUSDT.P", "AI16ZUSDT.P", "AIOZUSDT.P", "AIUSDT.P", "AIXBTUSDT.P", "AKTUSDT.P", "ALCHUSDT.P",
    "ALEOUSDT.P", "ALGOUSDT.P", "ALICEUSDT.P", "ALPACAUSDT.P", "ALPHAUSDT.P", "ALTUSDT.P", "ALUUSDT.P", "AMBUSDT.P",
    "ANIMEUSDT.P", "ANKRUSDT.P", "APEUSDT.P", "API3USDT.P", "APTUSDT.P", "ARBPERP.P", "ARBUSDT.P", "ARCUSDT.P", "ARKMUSDT.P",
    "ARKUSDT.P", "ARPAUSDT.P", "ARUSDT.P", "ASTRUSDT.P", "ATAUSDT.P", "ATHUSDT.P", "ATOMUSDT.P", "AUCTIONUSDT.P", "AUDIOUSDT.P",
    "AVAAIUSDT.P", "AVAILUSDT.P", "AVAUSDT.P", "AVAXUSDT.P", "AXLUSDT.P", "AXSUSDT.P", "B3USDT.P", "BADGERUSDT.P", "BAKEUSDT.P",
    "BALUSDT.P", "BANANAUSDT.P", "BANDUSDT.P", "BANUSDT.P", "BATUSDT.P", "BBUSDT.P", "BCHUSDT.P", "BEAMUSDT.P", "BELUSDT.P",
    "BERAUSDT.P", "BICOUSDT.P", "BIGTIMEUSDT.P", "BILLYUSDT.P", "BIOUSDT.P", "BLASTUSDT.P", "BLUEUSDT.P", "BLURUSDT.P",
    "BNBPERP.P", "BNBUSDT.P", "BNTUSDT.P", "BNXUSDT.P", "BOBAUSDT.P", "BOMEUSDT.P", "BRETTUSDT.P", "BROCCOLIUSDT.P", "BSVUSDT.P",
    "BSWUSDT.P", "BTC-07MAR25.P", "BTC-21FEB25.P", "BTC-25APR25.P", "BTC-26DEC25.P", "BTC-26SEP25.P", "BTC-27JUN25.P",
    "BTC-28FEB25.P", "BTC-28MAR25.P", "BTCPERP.P", "BTCUSDT.P", "BUZZUSDT.P", "C98USDT.P", "CAKEUSDT.P", "CARVUSDT.P",
    "CATIUSDT.P", "CELOUSDT.P", "CELRUSDT.P", "CETUSUSDT.P", "CFXUSDT.P", "CGPTUSDT.P", "CHESSUSDT.P", "CHILLGUYUSDT.P",
    "CHRUSDT.P", "CHZUSDT.P", "CKBUSDT.P", "CLOUDUSDT.P", "COMBOUSDT.P", "COMPUSDT.P", "COOKIEUSDT.P", "COOKUSDT.P",
    "COREUSDT.P", "COSUSDT.P", "COTIUSDT.P", "COWUSDT.P", "CROUSDT.P", "CRVUSDT.P", "CTCUSDT.P", "CTKUSDT.P", "CTSIUSDT.P",
    "CVCUSDT.P", "CVXUSDT.P", "CYBERUSDT.P", "DASHUSDT.P", "DATAUSDT.P", "DBRUSDT.P", "DEEPUSDT.P", "DEGENUSDT.P", "DENTUSDT.P",
    "DEXEUSDT.P", "DGBUSDT.P", "DODOUSDT.P", "DOGEPERP.P", "DOGEUSDT.P", "DOGSUSDT.P", "DOGUSDT.P", "DOTPERP.P", "DOTUSDT.P",
    "DRIFTUSDT.P", "DUCKUSDT.P", "DUSKUSDT.P", "DYDXUSDT.P", "DYMUSDT.P", "EDUUSDT.P", "EGLDUSDT.P", "EIGENUSDT.P", "ENAPERP.P",
    "ENAUSDT.P", "ENJUSDT.P", "ENSUSDT.P", "EOSUSDT.P", "ETCPERP.P", "ETCUSDT.P", "ETH-07MAR25.P", "ETH-21FEB25.P",
    "ETH-25APR25.P", "ETH-26DEC25.P", "ETH-26SEP25.P", "ETH-27JUN25.P", "ETH-28FEB25.P", "ETH-28MAR25.P", "ETHBTCUSDT.P",
    "ETHFIPERP.P", "ETHFIUSDT.P", "ETHPERP.P", "ETHUSDT.P", "ETHWUSDT.P", "FARTCOINUSDT.P", "FBUSDT.P", "FDUSDUSDT.P",
    "FIDAUSDT.P", "FILUSDT.P", "FIOUSDT.P", "FIREUSDT.P", "FLMUSDT.P", "FLOCKUSDT.P", "FLOWUSDT.P", "FLRUSDT.P", "FLUXUSDT.P",
    "FORTHUSDT.P", "FOXYUSDT.P", "FTNUSDT.P", "FUELUSDT.P", "FUSDT.P", "FWOGUSDT.P", "FXSUSDT.P", "GALAUSDT.P", "GASUSDT.P",
    "GEMSUSDT.P", "GIGAUSDT.P", "GLMRUSDT.P", "GLMUSDT.P", "GMEUSDT.P", "GMTUSDT.P", "GMXUSDT.P", "GNOUSDT.P", "GOATUSDT.P",
    "GODSUSDT.P", "GOMININGUSDT.P", "GRASSUSDT.P", "GRIFFAINUSDT.P", "GRTUSDT.P", "GTCUSDT.P", "GUSDT.P", "HBARUSDT.P",
    "HEIUSDT.P", "HFTUSDT.P", "HIFIUSDT.P", "HIGHUSDT.P", "HIPPOUSDT.P", "HIVEUSDT.P", "HMSTRUSDT.P", "HNTUSDT.P", "HOOKUSDT.P",
    "HOTUSDT.P", "HPOS10IUSDT.P", "HYPEUSDT.P", "ICPUSDT.P", "ICXUSDT.P", "IDEXUSDT.P", "IDUSDT.P", "ILVUSDT.P", "IMXUSDT.P",
    "INJUSDT.P", "IOSTUSDT.P", "IOTAUSDT.P", "IOTXUSDT.P", "IOUSDT.P", "IPUSDT.P", "JAILSTOOLUSDT.P", "JASMYUSDT.P",
    "JELLYJELLYUSDT.P", "JOEUSDT.P", "JSTUSDT.P", "JTOUSDT.P", "JUPUSDT.P", "JUSDT.P", "KAIAUSDT.P", "KASUSDT.P", "KAVAUSDT.P",
    "KDAUSDT.P", "KMNOUSDT.P", "KNCUSDT.P", "KOMAUSDT.P", "KSMUSDT.P", "L3USDT.P", "LAIUSDT.P", "LDOUSDT.P", "LEVERUSDT.P",
    "LINAUSDT.P", "LINKPERP.P", "LINKUSDT.P", "LISTAUSDT.P", "LOOKSUSDT.P", "LPTUSDT.P", "LQTYUSDT.P", "LRCUSDT.P", "LSKUSDT.P",
    "LTCUSDT.P", "LUCEUSDT.P", "LUMIAUSDT.P", "LUNA2USDT.P", "MAGICUSDT.P", "MAJORUSDT.P", "MANAUSDT.P", "MANEKIUSDT.P",
    "MANTAUSDT.P", "MASAUSDT.P", "MASKUSDT.P", "MAVIAUSDT.P", "MAVUSDT.P", "MAXUSDT.P", "MBLUSDT.P", "MBOXUSDT.P", "MDTUSDT.P",
    "MELANIAUSDT.P", "MEMEFIUSDT.P", "MEMEUSDT.P", "MERLUSDT.P", "METISUSDT.P", "MEUSDT.P", "MEWUSDT.P", "MICHIUSDT.P",
    "MINAUSDT.P", "MKRUSDT.P", "MNTPERP.P", "MNTUSDT.P", "MOBILEUSDT.P", "MOCAUSDT.P", "MONUSDT.P", "MOODENGUSDT.P",
    "MORPHOUSDT.P", "MOTHERUSDT.P", "MOVEUSDT.P", "MOVRUSDT.P", "MTLUSDT.P", "MVLUSDT.P", "MYRIAUSDT.P", "MYROUSDT.P",
    "NCUSDT.P", "NEARUSDT.P", "NEIROETHUSDT.P", "NEOUSDT.P", "NFPUSDT.P", "NKNUSDT.P", "NMRUSDT.P", "NOTPERP.P", "NOTUSDT.P",
    "NSUSDT.P", "NTRNUSDT.P", "NULSUSDT.P", "NYANUSDT.P", "OGNUSDT.P", "OGUSDT.P", "OLUSDT.P", "OMGUSDT.P", "OMNIUSDT.P",
    "OMUSDT.P", "ONDOPERP.P", "ONDOUSDT.P", "ONEUSDT.P", "ONGUSDT.P", "ONTUSDT.P", "OPPERP.P", "OPUSDT.P", "ORBSUSDT.P",
    "ORCAUSDT.P", "ORDERUSDT.P", "ORDIPERP.P", "ORDIUSDT.P", "OSMOUSDT.P", "OXTUSDT.P", "PAXGUSDT.P", "PEAQUSDT.P",
    "PENDLEUSDT.P", "PENGUUSDT.P", "PEOPLEUSDT.P", "PERPUSDT.P", "PHAUSDT.P", "PHBUSDT.P", "PIPPINUSDT.P", "PIRATEUSDT.P",
    "PIXELUSDT.P", "PLUMEUSDT.P", "PNUTUSDT.P", "POLPERP.P", "POLUSDT.P", "POLYXUSDT.P", "PONKEUSDT.P", "POPCATPERP.P",
    "POPCATUSDT.P", "PORTALUSDT.P", "POWRUSDT.P", "PRCLUSDT.P", "PRIMEUSDT.P", "PROMUSDT.P", "PROSUSDT.P", "PUFFERUSDT.P",
    "PYRUSDT.P", "PYTHUSDT.P", "QIUSDT.P", "QNTUSDT.P", "QTUMUSDT.P", "QUICKUSDT.P", "RADUSDT.P", "RAREUSDT.P", "RAYDIUMUSDT.P",
    "RDNTUSDT.P", "RENDERUSDT.P", "RENUSDT.P", "REQUSDT.P", "REXUSDT.P", "REZUSDT.P", "RIFSOLUSDT.P", "RIFUSDT.P", "RLCUSDT.P",
    "RONINUSDT.P", "ROSEUSDT.P", "RPLUSDT.P", "RSRUSDT.P", "RSS3USDT.P", "RUNEUSDT.P", "RVNUSDT.P", "SAFEUSDT.P", "SAGAUSDT.P",
    "SANDUSDT.P", "SCAUSDT.P", "SCRTUSDT.P", "SCRUSDT.P", "SCUSDT.P", "SDUSDT.P", "SEIUSDT.P", "SENDUSDT.P", "SFPUSDT.P",
    "SHELLUSDT.P", "SHIB1000PERP.P", "SHIB1000USDT.P", "SKLUSDT.P", "SLERFUSDT.P", "SLFUSDT.P", "SLPUSDT.P", "SNTUSDT.P",
    "SNXUSDT.P", "SOL-07MAR25.P", "SOL-21FEB25.P", "SOL-28FEB25.P", "SOL-28MAR25.P", "SOLAYERUSDT.P", "SOLOUSDT.P", "SOLPERP.P",
    "SOLUSDT.P", "SOLUSDT-04APR25.P", "SOLUSDT-11APR25.P", "SOLVUSDT.P", "SONICUSDT.P", "SPECUSDT.P", "SPELLUSDT.P", "SPXUSDT.P",
    "SSVUSDT.P", "STEEMUSDT.P", "STGUSDT.P", "STMXUSDT.P", "STORJUSDT.P", "STPTUSDT.P", "STRKPERP.P", "STRKUSDT.P", "STXUSDT.P",
    "SUIPERP.P", "SUIUSDT.P", "SUNDOGUSDT.P", "SUNUSDT.P", "SUPERUSDT.P", "SUSDT.P", "SUSHIUSDT.P", "SWARMSUSDT.P", "SWEATUSDT.P",
    "SWELLUSDT.P", "SXPUSDT.P", "SYNUSDT.P", "SYSUSDT.P", "TAIKOUSDT.P", "TAIUSDT.P", "TAOUSDT.P", "THETAUSDT.P", "THEUSDT.P",
    "TIAPERP.P", "TIAUSDT.P", "TLMUSDT.P", "TNSRUSDT.P", "TOKENUSDT.P", "TONPERP.P", "TONUSDT.P", "TRBUSDT.P", "TROYUSDT.P",
    "TRUMPUSDT.P", "TRUUSDT.P", "TRXUSDT.P", "TSTBSCUSDT.P", "TUSDT.P", "TWTUSDT.P", "UMAUSDT.P", "UNIUSDT.P", "UROUSDT.P",
    "USDCUSDT.P", "USDEUSDT.P", "USTCUSDT.P", "USUALUSDT.P", "UXLINKUSDT.P", "VANAUSDT.P", "VANRYUSDT.P"
]
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
