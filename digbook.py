import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tradingview_ta import TA_Handler, Interval
from decimal import Decimal

def float_to_decimal(value):
    return Decimal(str(value))

def fetch_data(symbol, exchange, screener, interval):
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

def calculate_true_range(high, low, close):
    previous_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()
    true_range = pd.DataFrame({'TR1': tr1, 'TR2': tr2, 'TR3': tr3}).max(axis=1)
    return true_range

def calculate_weighted_atr(data):
    atr_data = []
    volume_data = []
    
    for interval, df in data.items():
        true_range = calculate_true_range(df['high'], df['low'], df['close'])
        weighted_tr = true_range * df['volume']
        atr_data.append(weighted_tr.sum())
        volume_data.append(df['volume'].sum())
    
    if sum(volume_data) == 0:
        return None
    
    weighted_atr = sum(atr_data) / sum(volume_data)
    return weighted_atr

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

def get_recommendation(current_price, r1, s1):
    distance_to_r1 = (r1 - current_price) / current_price * Decimal('100')
    distance_to_s1 = (current_price - s1) / current_price * Decimal('100')
    
    if distance_to_r1 > Decimal('5') and distance_to_s1 < Decimal('2.1'):
        return "LONG"
    elif distance_to_s1 > Decimal('5') and distance_to_r1 < Decimal('2.1'):
        return "SHORT"
    else:
        return "NEUTRAL"

def optimize_grid_settings(current_price, atr, position_type, s1, r1):
    grid_size = atr * Decimal('0.5')
    num_grids = 10
    
    if position_type == "LONG":
        entry_point = max(current_price - (num_grids / Decimal('2') * grid_size), s1)
        exit_point = min(current_price + (num_grids / Decimal('2') * grid_size), r1)
    elif position_type == "SHORT":
        entry_point = min(current_price + (num_grids / Decimal('2') * grid_size), r1)
        exit_point = max(current_price - (num_grids / Decimal('2') * grid_size), s1)
    else:
        entry_point = current_price
        exit_point = current_price
    
    stop_loss = entry_point * Decimal('0.95') if position_type == "LONG" else entry_point * Decimal('1.05')
    take_profit = exit_point
    
    return {
        "grid_size": grid_size,
        "num_grids": num_grids,
        "entry_point": entry_point,
        "exit_point": exit_point,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }

def calculate_grid_profit(suggested_grid_size, current_price):
    return (suggested_grid_size / current_price) * Decimal('100')

def grid_optimization_page():
    st.header("Grid Optimization")
    st.write("Optimize grid settings based on real-time market data, pivot points, and support/resistance levels.")
    
    symbol = st.text_input("Enter symbol (e.g., CKBUSDT):", value="CKBUSDT")
    
    if symbol:
        analysis = fetch_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_1_DAY)
        if analysis:
            current_price = float_to_decimal(analysis.indicators['close'])
            high = float_to_decimal(analysis.indicators['high'])
            low = float_to_decimal(analysis.indicators['low'])
            close = float_to_decimal(analysis.indicators['close'])
            volume = float_to_decimal(analysis.indicators['volume'])
            
            # Calculate pivot points
            pivots = calculate_pivot_points(high, low, close)
            
            # Display pivot points
            st.subheader("Pivot Points")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pivot", f"{pivots['pivot']:.8f}")
                st.metric("Support 1 (S1)", f"{pivots['s1']:.8f}")
                st.metric("Support 2 (S2)", f"{pivots['s2']:.8f}")
            with col2:
                st.metric("Resistance 1 (R1)", f"{pivots['r1']:.8f}")
                st.metric("Resistance 2 (R2)", f"{pivots['r2']:.8f}")
            with col3:
                st.metric("Support 3 (S3)", f"{pivots['s3']:.8f}")
                st.metric("Resistance 3 (R3)", f"{pivots['r3']:.8f}")
            
            # Calculate ATR
            atr = calculate_weighted_atr({
                Interval.INTERVAL_1_DAY: pd.DataFrame({
                    'close': [close],
                    'high': [high],
                    'low': [low],
                    'volume': [volume]
                })
            })
            
            if atr is None:
                st.error("Unable to calculate ATR. Please check the data.")
                return
            
            # Calculate GRID Profit
            grid_size = atr * Decimal('0.5')
            grid_profit = calculate_grid_profit(grid_size, current_price)
            
            st.subheader("Market Data")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Price", f"{current_price:.8f}")
            with col2:
                st.metric("ATR", f"{atr:.8f}")
            with col3:
                st.metric("GRID Profit (%)", f"{grid_profit:.2f}%")
            
            # Get recommendation
            recommendation = get_recommendation(current_price, pivots['r1'], pivots['s1'])
            
            # Optimize grid settings
            best_settings = optimize_grid_settings(current_price, atr, recommendation, pivots['s1'], pivots['r1'])
            
            st.subheader("Strategy Recommendation")
            st.markdown(f"<h1 style='text-align: center; color: {'green' if recommendation == 'LONG' else 'red' if recommendation == 'SHORT' else 'yellow'};'>{recommendation}</h1>", unsafe_allow_html=True)
            
            st.subheader("Optimized Grid Settings")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Optimal Entry Point", f"{best_settings['entry_point']:.8f}")
                st.metric("Stop Loss", f"{best_settings['stop_loss']:.8f}")
            with col2:
                st.metric("Optimal Exit Point", f"{best_settings['exit_point']:.8f}")
                st.metric("Take Profit", f"{best_settings['take_profit']:.8f}")
            with col3:
                st.metric("Grid Size", f"{best_settings['grid_size']:.8f}")
                st.metric("Number of Grids", best_settings['num_grids'])
            
            # Visualization
            st.subheader("Grid Visualization")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0], y=[float(current_price)], mode='markers', marker=dict(color='blue', size=10), name='Current Price'))

            # Add pivot lines
            for level, name in [(pivots['pivot'], 'Pivot'), (pivots['s1'], 'S1'), (pivots['r1'], 'R1'), 
                                (pivots['s2'], 'S2'), (pivots['r2'], 'R2'), (pivots['s3'], 'S3'), (pivots['r3'], 'R3')]:
                fig.add_trace(go.Scatter(x=[-1, 1], y=[float(level), float(level)], mode='lines', line=dict(color='gray', dash='dash'), name=name))

            if recommendation != "NEUTRAL":
                for i in range(best_settings['num_grids']):
                    level = float(best_settings['entry_point'] + i * best_settings['grid_size']) if recommendation == "LONG" else float(best_settings['entry_point'] - i * best_settings['grid_size'])
                    if (recommendation == "LONG" and level <= float(best_settings['exit_point'])) or (recommendation == "SHORT" and level >= float(best_settings['exit_point'])):
                        color = 'green' if (recommendation == "LONG" and level > float(current_price)) or (recommendation == "SHORT" and level < float(current_price)) else 'red'
                        fig.add_trace(go.Scatter(x=[-0.5, 0.5], y=[level, level], mode='lines', line=dict(color=color), name=f'Grid Level {i+1}'))

            fig.update_layout(title="Grid Optimization", xaxis_title="", yaxis_title="Price", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Strategy explanation
            st.subheader("Strategy Explanation")
            st.write(f"""
            Based on the current market conditions for {symbol}:
            
            1. The distance from the current price to R1 is {((pivots['r1'] - current_price) / current_price * 100):.2f}%
            2. The distance from the current price to S1 is {((current_price - pivots['s1']) / current_price * 100):.2f}%
            
            Our strategy recommends a {recommendation} position because:
            
            - For a LONG position: The distance to R1 should be > 5% and the distance to S1 should be < 2.1%
            - For a SHORT position: The distance to S1 should be > 5% and the distance to R1 should be < 2.1%
            
            The grid settings are optimized to place orders between the current price and the target (R1 for LONG, S1 for SHORT),
            with a stop loss set at 5% beyond the entry point.
            """)

def main():
    st.set_page_config(page_title="Advanced Futures Grid Bot Strategy", layout="wide")
    
    st.title("Advanced Futures Grid Bot Strategy")
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Chapters", ["Introduction", "Grid Optimization", "Profit Projections", "Strategy Card"])

    if page == "Grid Optimization":
        grid_optimization_page()
    elif page == "Introduction":
        st.header("Welcome to the Advanced Futures Grid Bot Strategy")
        st.write("This interactive tool helps you optimize your grid bot strategy using market data, pivot points, and support/resistance levels.")
        st.write("Navigate to the Grid Optimization page to analyze a specific symbol and get recommendations.")
    elif page == "Profit Projections":
        st.header("Profit Projections")
        st.write("Project potential profits based on your grid bot settings.")
        st.info("This section is under development. Check back soon for updates!")
    elif page == "Strategy Card":
        st.header("Strategy Card")
        st.write("Summarize your settings and download a personalized strategy card.")
        st.info("This section is under development. Check back soon for updates!")

if __name__ == "__main__":
    main()
