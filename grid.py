import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def calculate_fibonacci_levels(start_price, end_price):
    """
    Calcula los niveles Fibonacci entre dos precios
    """
    fib_levels = {
        0: 1,
        0.236: 0.764,
        0.382: 0.618,
        0.5: 0.5,
        0.618: 0.382,
        0.786: 0.214,
        1: 0
    }
    
    diff = end_price - start_price
    levels = {}
    
    for fib, multiplier in fib_levels.items():
        levels[fib] = start_price + (diff * multiplier)
    
    return levels

def calculate_grid_profit_percentage(take_profit, stop_loss, n_grids, leverage):
    """
    Calcula el porcentaje de ganancia por grid basado en el modelo de Bybit
    """
    total_range = abs(take_profit - stop_loss)
    avg_price = (take_profit + stop_loss) / 2
    
    # Calcular porcentaje base
    range_percentage = (total_range / avg_price) * 100
    base_grid_percentage = range_percentage / n_grids
    
    # Aplicar factor de apalancamiento con ajuste de 0.9
    grid_profit_percentage = base_grid_percentage * leverage * 0.9
    
    return grid_profit_percentage

def calculate_grid_levels(entry_price, box_high, box_low, leverage=5, direction='LONG'):
    """
    Calcula los niveles de grid usando la combinación correcta de cálculos
    """
    box_range = box_high - box_low
    
    if direction == 'LONG':
        # Take Profit: Calcular desde entry hasta extensión superior
        extension_high = box_high + box_range
        take_profit_levels = calculate_fibonacci_levels(entry_price, extension_high)
        take_profit = take_profit_levels[0.236]
        
        # Stop Loss: Proyección hacia abajo desde box_low
        stop_loss = box_low - (box_range * 0.236)
        
    else:  # SHORT
        # Take Profit: Calcular desde entry hasta extensión inferior
        extension_low = box_low - box_range
        take_profit_levels = calculate_fibonacci_levels(entry_price, extension_low)
        take_profit = take_profit_levels[0.236]
        
        # Stop Loss: Proyección hacia arriba desde box_high
        stop_loss = box_high + (box_range * 0.236)
    
    # Calcular número de grids y profit
    n_grids = 21  # Fijo según el ejemplo de Bybit
    profit_percentage = calculate_grid_profit_percentage(take_profit, stop_loss, n_grids, leverage)
    
    # Calcular niveles de grid
    grid_range = abs(take_profit - stop_loss)
    grid_step = grid_range / n_grids
    
    if direction == 'LONG':
        grid_levels = [stop_loss + (grid_step * i) for i in range(n_grids + 1)]
    else:
        grid_levels = [stop_loss - (grid_step * i) for i in range(n_grids + 1)]
    
    return {
        'entry': entry_price,
        'take_profit': take_profit,
        'stop_loss': stop_loss,
        'risk': abs(entry_price - stop_loss),
        'reward': abs(take_profit - entry_price),
        'rr_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss),
        'n_grids': n_grids,
        'grid_levels': grid_levels,
        'profit_percentage': profit_percentage,
        'box_high': box_high,
        'box_low': box_low,
        'direction': direction
    }

def plot_grid_levels(results):
    """
    Visualiza los niveles de grid y las proyecciones
    """
    fig = go.Figure()
    
    # Box range
    fig.add_hline(y=results['box_high'], line_color="rgba(255,255,255,0.2)", 
                 line_dash="dash", annotation_text=f"Box High {results['box_high']:.4f}")
    fig.add_hline(y=results['box_low'], line_color="rgba(255,255,255,0.2)", 
                 line_dash="dash", annotation_text=f"Box Low {results['box_low']:.4f}")
    
    # Niveles principales
    direction_color = "green" if results['direction'] == 'LONG' else "red"
    
    fig.add_hline(y=results['entry'], line_color="white",
                 annotation_text=f"Entry {results['entry']:.4f}")
    fig.add_hline(y=results['take_profit'], line_color=direction_color,
                 annotation_text=f"TP {results['take_profit']:.4f}")
    fig.add_hline(y=results['stop_loss'], line_color="red",
                 annotation_text=f"SL {results['stop_loss']:.4f}")
    
    # Niveles de grid
    for i, level in enumerate(results['grid_levels']):
        fig.add_hline(y=level, line_color="rgba(0, 255, 255, 0.1)",
                     annotation_text=f"Grid {i+1}: {level:.4f}")
    
    fig.update_layout(
        template="plotly_dark",
        title=f"Grid Levels - {results['direction']} - Profit per Grid: {results['profit_percentage']:.4f}%",
        yaxis_title="Price",
        showlegend=False,
        height=800,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(gridcolor='rgba(255,255,255,0.1)')
    )
    
    return fig

def main():
    st.set_page_config(page_title="Grid Range Calculator", page_icon="📊", layout="wide")
    
    # Dark Mode Style
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stNumberInput>div>div>input {
            background-color: #262730;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.title("📊 Grid Range Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Parameters")
        
        direction = st.radio(
            "Trading Direction",
            ('LONG', 'SHORT'),
            horizontal=True
        )
        
        entry_price = st.number_input("Entry Price", value=1.5485, format="%.4f", step=0.0001)
        box_high = st.number_input("Box High", value=1.5695, format="%.4f", step=0.0001)
        box_low = st.number_input("Box Low", value=1.4436, format="%.4f", step=0.0001)
        leverage = st.slider("Leverage", min_value=1, max_value=20, value=5)
    
    if st.button("Calculate Levels", key="calculate"):
        # Validación
        if box_low >= box_high:
            st.error("Box Low must be lower than Box High")
            return
        
        # Cálculos
        results = calculate_grid_levels(
            entry_price=entry_price,
            box_high=box_high,
            box_low=box_low,
            leverage=leverage,
            direction=direction
        )
        
        # Resultados
        with col2:
            st.subheader("Results")
            col_tp, col_sl = st.columns(2)
            with col_tp:
                st.metric(
                    "Take Profit", 
                    f"{results['take_profit']:.4f}",
                    delta=f"{((results['take_profit']/entry_price - 1) * 100):.2f}%"
                )
            with col_sl:
                st.metric(
                    "Stop Loss", 
                    f"{results['stop_loss']:.4f}",
                    delta=f"{((results['stop_loss']/entry_price - 1) * 100):.2f}%"
                )
            
            col_profit, col_rr = st.columns(2)
            with col_profit:
                st.metric(
                    "Grid Profit", 
                    f"{results['profit_percentage']:.4f}%"
                )
            with col_rr:
                st.metric(
                    "Number of Grids", 
                    results['n_grids']
                )
                
            st.metric("Risk/Reward Ratio", f"{results['rr_ratio']:.2f}")
        
        # Visualización
        st.subheader("Grid Visualization")
        fig = plot_grid_levels(results)
        st.plotly_chart(fig, use_container_width=True)
        
        # Export Data
        df = pd.DataFrame({
            'Level': ['Box High', 'Box Low', 'Entry', 'Take Profit', 'Stop Loss'] + 
                    [f'Grid {i+1}' for i in range(len(results['grid_levels']))],
            'Price': [results['box_high'], results['box_low'], results['entry'],
                     results['take_profit'], results['stop_loss']] + results['grid_levels'],
            'Direction': direction,
            'Leverage': leverage,
            'Profit per Grid': f"{results['profit_percentage']:.4f}%"
        })
        
        # Download Button
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "Download CSV",
            csv,
            f"grid_levels_{direction}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv"
        )

if __name__ == "__main__":
    main()
