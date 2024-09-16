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

def optimize_grid_settings(current_price, atr, position_type):
    grid_size = atr * 0.5
    num_grids = 10
    
    if position_type == "LONG":
        entry_point = current_price - (num_grids / 2 * grid_size)
        exit_point = current_price + (num_grids / 2 * grid_size)
    else:
        entry_point = current_price + (num_grids / 2 * grid_size)
        exit_point = current_price - (num_grids / 2 * grid_size)
    
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
    st.title('Digital Book for Grid Bot Optimization')

    # Sidebar navigation
    page = st.sidebar.selectbox("Chapters", ["Introduction", "Pivot Points", "Grid Optimization", "Profit Projections", "Strategy Card"])

    # Introduction Page
    if page == "Introduction":
        st.header("Welcome to the Digital Book on Grid Bot Strategies")
        st.write("In this interactive digital book, you’ll learn how to optimize your grid bot strategy using mathematical and statistical principles.")
        st.write("Navigate through the chapters to explore concepts like Pivot Points, Grid Bot Optimization, Risk Management, and Profit Projections.")
    
    # Chapter 1: Pivot Points
    elif page == "Pivot Points":
        st.header("Chapter 1: Pivot Points")
        st.write("Pivot Points are critical in identifying potential support and resistance levels for grid bot trading.")
        
        symbol = st.text_input("Enter symbol (e.g., CKBUSDT.P):", value="CKBUSDT.P")
        if symbol:
            analysis = fetch_data(symbol, "BYBIT", "crypto", Interval.INTERVAL_1_DAY)
            if analysis:
                high = analysis.indicators['high']
                low = analysis.indicators['low']
                close = analysis.indicators['close']
                
                pivots = calculate_pivot_points(high, low, close)
                st.write(f"Pivot: {pivots['pivot']:.4f}")
                st.write(f"Resistance 1 (R1): {pivots['r1']:.4f}")
                st.write(f"Support 1 (S1): {pivots['s1']:.4f}")
                st.write(f"Resistance 2 (R2): {pivots['r2']:.4f}")
                st.write(f"Support 2 (S2): {pivots['s2']:.4f}")
                
                # Visualize Pivot Points
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0, 1], y=[pivots['pivot'], pivots['pivot']], mode='lines', name='Pivot'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[pivots['r1'], pivots['r1']], mode='lines', name='R1'))
                fig.add_trace(go.Scatter(x=[0, 1], y=[pivots['s1'], pivots['s1']], mode='lines', name='S1'))
                fig.update_layout(title="Pivot Points", xaxis_title="Time", yaxis_title="Price")
                st.plotly_chart(fig)

    # Chapter 2: Grid Optimization
    elif page == "Grid Optimization":
        st.header("Chapter 2: Grid Bot Optimization")
        st.write("Optimize grid settings based on volatility and price structure.")
        
        symbol = st.text_input("Enter symbol (e.g., CKBUSDT):", value="CKBUSDT")
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

                # Get pivot points for the plot
                pivots = calculate_pivot_points(analysis.indicators['high'], analysis.indicators['low'], analysis.indicators['close'])
                r1 = pivots['r1']
                s1 = pivots['s1']
                
                # Calculate GRID Profit
                grid_size = atr * 0.5
                grid_profit = calculate_grid_profit(grid_size, current_price)
                
                st.write(f"Current Price: {current_price:.4f}")
                st.write(f"Weighted ATR: {atr:.4f}")
                st.metric("GRID Profit (%)", f"{grid_profit:.2f}%")
                
                # Optimize grid settings
                best_settings = optimize_grid_settings(current_price, atr, position_type)
                st.write(f"Optimal Entry Point: {best_settings['entry_point']:.4f}")
                st.write(f"Optimal Exit Point: {best_settings['exit_point']:.4f}")
                
                # Plot grid optimization
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=[0], y=[current_price], mode='markers', marker=dict(color='blue', size=10), name='Current Price'))

                if position_type == "LONG":
                    for i in range(best_settings['num_grids']):
                        level = current_price + (i - best_settings['num_grids'] / 2) * best_settings['grid_size']
                        if level > current_price:
                            fig.add_trace(go.Scatter(x=[0], y=[level], mode='lines', line=dict(color='green'), name=f'Grid Level {i+1} (Long)'))
                        else:
                            fig.add_trace(go.Scatter(x=[0], y=[level], mode='lines', line=dict(color='red'), name=f'Grid Level {i+1} (Short)'))
                else:
                    for i in range(best_settings['num_grids']):
                        level = current_price - (i - best_settings['num_grids'] / 2) * best_settings['grid_size']
                        if level > current_price:
                            fig.add_trace(go.Scatter(x=[0], y=[level], mode='lines', line=dict(color='red'), name=f'Grid Level {i+1} (Short)'))
                        else:
                            fig.add_trace(go.Scatter(x=[0], y=[level], mode='lines', line=dict(color='green'), name=f'Grid Level {i+1} (Long)'))

                fig.update_layout(title="Grid Optimization", xaxis_title="Time", yaxis_title="Price")
                st.plotly_chart(fig)
                
                 # Calculate grid metrics
                num_green_lines = len([line for line in fig.data if line.name.endswith('(Long)')])
                num_red_lines = len([line for line in fig.data if line.name.endswith('(Short)')])
                
                ratio = num_green_lines / num_red_lines if num_red_lines > 0 else float('inf')
                ratio_label = "Favorable" if ratio > 1 else "Not Favorable"
                
                st.metric("Number of Green Lines", num_green_lines)
                st.metric("Number of Red Lines", num_red_lines)
                st.metric("Green to Red Line Ratio", f"{ratio:.2f}")
                st.write(f"Ratio is: {ratio_label}")

    # Chapter 3: Profit Projections
    elif page == "Profit Projections":
        st.header("Chapter 3: Profit Projections")
        st.write("Project potential profits based on your grid bot settings.")

    # Chapter 4: Strategy Card
    elif page == "Strategy Card":
        st.header("Chapter 4: Strategy Card")
        st.write("Summarize your settings and download a personalized strategy card.")

if __name__ == "__main__":
    main()
