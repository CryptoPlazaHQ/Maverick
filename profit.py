import streamlit as st
import pandas as pd

def calculate_parameters(grid_size):
    tp_base = 0.04
    sl_base = 0.05
    
    tp = round(grid_size * (tp_base / 0.02), 4)
    sl = round(grid_size * (sl_base / 0.02), 4)
    
    num_grids_min = max(1, int(2 / grid_size))
    num_grids_max = min(50, int(5 / grid_size))
    
    return tp, sl, num_grids_min, num_grids_max

def main():
    st.title("Grid Bot Strategy Calculator")
    
    st.sidebar.header("Input Parameters")
    grid_size = st.sidebar.number_input("Grid Size (%)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
    num_bots = 3
    
    tp, sl, num_grids_min, num_grids_max = calculate_parameters(grid_size/100)
    
    st.header("Calculated Strategy Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Take Profit (TP)", f"{tp:.2%}")
        st.metric("Stop Loss (SL)", f"{sl:.2%}")
    with col2:
        st.metric("Min Number of Grids", num_grids_min)
        st.metric("Max Number of Grids", num_grids_max)
    
    st.header("Adjustable Parameters")
    col3, col4 = st.columns(2)
    with col3:
        adjusted_tp = st.slider("Adjusted TP", min_value=tp-0.01, max_value=tp+0.01, value=tp, step=0.0001, format="%.4f")
    with col4:
        adjusted_sl = st.slider("Adjusted SL", min_value=sl-0.01, max_value=sl+0.01, value=sl, step=0.0001, format="%.4f")
    
    st.header("Multi-Bot Configuration")
    st.write(f"Number of Bots: {num_bots}")
    
    total_risk = adjusted_sl * num_bots
    global_sl = min(0.20, total_risk)
    
    st.metric("Total Risk Exposure", f"{total_risk:.2%}")
    st.metric("Global Stop Loss (max 20%)", f"{global_sl:.2%}")
    
    if total_risk > 0.20:
        st.warning("Warning: Total risk exposure exceeds 20%. Consider adjusting individual bot stop losses.")
    
    st.header("Strategy Summary")
    summary_data = {
        'Parameter': ['Grid Size', 'Take Profit', 'Stop Loss', 'Number of Bots', 'Global Stop Loss'],
        'Value': [f"{grid_size:.1f}%", f"{adjusted_tp:.2%}", f"{adjusted_sl:.2%}", num_bots, f"{global_sl:.2%}"]
    }
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

if __name__ == "__main__":
    main()
