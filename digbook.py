import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from tradingview_ta import TA_Handler, Interval

# Utility functions to fetch and process market data
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
    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    return {
        'pivot': pivot,
        'r1': r1, 's1': s1,
        'r2': r2, 's2': s2,
        'r3': r3, 's3': s3
    }

def optimize_grid_settings(current_price, atr, position_type, support, resistance):
    grid_size = atr * 0.5
    num_grids = 10
    
    if position_type == "LONG":
        entry_point = max(current_price - (num_grids / 2 * grid_size), support)
        exit_point = min(current_price + (num_grids / 2 * grid_size), resistance)
    else:
        entry_point = min(current_price + (num_grids / 2 * grid_size), resistance)
        exit_point = max(current_price - (num_grids / 2 * grid_size), support)
    
    stop_loss = entry_point * 0.95 if position_type == "LONG" else entry_point * 1.05
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
    return (suggested_grid_size / current_price) * 100

# Main Streamlit App Logic
def main():
    st.set_page_config(layout="wide")
    st.title('Digital Book for Grid Bot Optimization')

    # Sidebar navigation
    page = st.sidebar.selectbox("Chapters", ["Introduction", "Grid Optimization", "Profit Projections", "Strategy Card"])

    # Introduction Page
    if page == "Introduction":
        st.header("Welcome to the Digital Book on Grid Bot Strategies")
        st.write("In this interactive digital book, you'll learn how to optimize your grid bot strategy using mathematical and statistical principles.")
        st.write("Navigate through the chapters to explore concepts like Pivot Points, Grid Bot Optimization, Risk Management, and Profit Projections.")
    
    # Grid Optimization Page (Merged Pivot Points and Grid Optimization)
    elif page == "Grid Optimization":
        st.header("Grid Optimization")
        st.write("Optimize grid settings based on volatility, price structure, and pivot points.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            symbol = st.text_input("Enter symbol (e.g., CKBUSDT):", value="CKBUSDT")
        with col2:
            position_type = st.selectbox("Select position type", ["LONG", "SHORT"])
        
        if symbol:
            analysis = fetch_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_1_DAY)
            if analysis:
                current_price = analysis.indicators['close']
                atr = calculate_weighted_atr({
                    Interval.INTERVAL_1_DAY: pd.DataFrame({
                        'close': [current_price],
                        'high': [analysis.indicators['high']],
                        'low': [analysis.indicators['low']],
                        'volume': [analysis.indicators['volume']]
                    })
                })

                # Calculate pivot points
                pivots = calculate_pivot_points(analysis.indicators['high'], analysis.indicators['low'], analysis.indicators['close'])
                
                # Display pivot points
                st.subheader("Pivot Points")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Pivot", f"{pivots['pivot']:.4f}")
                    st.metric("Support 1 (S1)", f"{pivots['s1']:.4f}")
                    st.metric("Support 2 (S2)", f"{pivots['s2']:.4f}")
                with col2:
                    st.metric("Resistance 1 (R1)", f"{pivots['r1']:.4f}")
                    st.metric("Resistance 2 (R2)", f"{pivots['r2']:.4f}")
                with col3:
                    st.metric("Support 3 (S3)", f"{pivots['s3']:.4f}")
                    st.metric("Resistance 3 (R3)", f"{pivots['r3']:.4f}")
                
                # Calculate GRID Profit
                grid_size = atr * 0.5
                grid_profit = calculate_grid_profit(grid_size, current_price)
                
                st.subheader("Market Data")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"{current_price:.4f}")
                with col2:
                    st.metric("Weighted ATR", f"{atr:.4f}")
                with col3:
                    st.metric("GRID Profit (%)", f"{grid_profit:.2f}%")
                
                # Optimize grid settings
                support = pivots['s1'] if position_type == "LONG" else pivots['s2']
                resistance = pivots['r1'] if position_type == "LONG" else pivots['r2']
                best_settings = optimize_grid_settings(current_price, atr, position_type, support, resistance)
                
                st.subheader("Optimized Grid Settings")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Optimal Entry Point", f"{best_settings['entry_point']:.4f}")
                    st.metric("Stop Loss", f"{best_settings['stop_loss']:.4f}")
                with col2:
                    st.metric("Optimal Exit Point", f"{best_settings['exit_point']:.4f}")
                    st.metric("Take Profit", f"{best_settings['take_profit']:.4f}")
                with col3:
                    st.metric("Grid Size", f"{best_settings['grid_size']:.4f}")
                    st.metric("Number of Grids", best_settings['num_grids'])
                
                # Plot grid optimization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0], y=[current_price], mode='markers', marker=dict(color='blue', size=10), name='Current Price'))

                # Add pivot lines
                for level, name in [(pivots['pivot'], 'Pivot'), (pivots['s1'], 'S1'), (pivots['r1'], 'R1'), 
                                    (pivots['s2'], 'S2'), (pivots['r2'], 'R2'), (pivots['s3'], 'S3'), (pivots['r3'], 'R3')]:
                    fig.add_trace(go.Scatter(x=[-1, 1], y=[level, level], mode='lines', line=dict(color='gray', dash='dash'), name=name))

                if position_type == "LONG":
                    for i in range(best_settings['num_grids']):
                        level = best_settings['entry_point'] + i * best_settings['grid_size']
                        if level <= best_settings['exit_point']:
                            color = 'green' if level > current_price else 'red'
                            fig.add_trace(go.Scatter(x=[-0.5, 0.5], y=[level, level], mode='lines', line=dict(color=color), name=f'Grid Level {i+1}'))
                else:
                    for i in range(best_settings['num_grids']):
                        level = best_settings['entry_point'] - i * best_settings['grid_size']
                        if level >= best_settings['exit_point']:
                            color = 'red' if level > current_price else 'green'
                            fig.add_trace(go.Scatter(x=[-0.5, 0.5], y=[level, level], mode='lines', line=dict(color=color), name=f'Grid Level {i+1}'))

                fig.update_layout(title="Grid Optimization", xaxis_title="", yaxis_title="Price", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate grid metrics
                num_green_lines = len([line for line in fig.data if line.line.color == 'green'])
                num_red_lines = len([line for line in fig.data if line.line.color == 'red'])
                
                ratio = num_green_lines / num_red_lines if num_red_lines > 0 else float('inf')
                ratio_label = "Favorable" if ratio > 1 else "Not Favorable"
                
                st.subheader("Grid Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Green Lines", num_green_lines)
                with col2:
                    st.metric("Number of Red Lines", num_red_lines)
                with col3:
                    st.metric("Green to Red Line Ratio", f"{ratio:.2f}")
                st.info(f"Current grid setup is: {ratio_label}")

    # Chapter 3: Profit Projections
    elif page == "Profit Projections":
        st.header("Profit Projections")
        st.write("Project potential profits based on your grid bot settings.")
        st.info("This section is under development. Check back soon for updates!")

    # Chapter 4: Strategy Card
    elif page == "Strategy Card":
        st.header("Strategy Card")
        st.write("Summarize your settings and download a personalized strategy card.")
        st.info("This section is under development. Check back soon for updates!")

if __name__ == "__main__":
    main()
