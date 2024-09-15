import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import time

class GridTradingBot:
    def __init__(self, symbol, position, amount_per_trade, lower_price, upper_price, grid_profit_percentage, take_profit, stop_loss, entry_price):
        self.symbol = symbol
        self.position = position
        self.amount_per_trade = amount_per_trade
        self.lower_price = lower_price
        self.upper_price = upper_price
        self.grid_profit_percentage = grid_profit_percentage
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self.entry_price = entry_price
        
        self.current_price = entry_price
        self.grids = self.create_grids()
        self.open_positions = []
        self.realized_profit = 0
        self.unrealized_profit = 0
        self.history = []
        self.trades = []

    def create_grids(self):
        num_grids = int((self.upper_price - self.lower_price) / (self.lower_price * self.grid_profit_percentage / 100))
        return [self.lower_price + i * (self.upper_price - self.lower_price) / num_grids for i in range(num_grids + 1)]

    def update_price(self, i, total_iterations):
        phase = (i / total_iterations) * 2 * np.pi
        trend = np.sin(phase) * (self.upper_price - self.lower_price) / 2
        noise = random.uniform(-0.00001, 0.00001) * self.current_price
        self.current_price = ((self.upper_price + self.lower_price) / 2) + trend + noise
        self.current_price = max(min(self.current_price, self.upper_price), self.lower_price)

    def check_and_execute_trades(self):
        for grid_price in self.grids:
            if self.position == "Long":
                if self.current_price <= grid_price and grid_price not in [pos['price'] for pos in self.open_positions]:
                    self.open_long_position(grid_price)
                elif self.current_price > grid_price and grid_price in [pos['price'] for pos in self.open_positions]:
                    self.close_long_position(grid_price)
            elif self.position == "Short":
                if self.current_price >= grid_price and grid_price not in [pos['price'] for pos in self.open_positions]:
                    self.open_short_position(grid_price)
                elif self.current_price < grid_price and grid_price in [pos['price'] for pos in self.open_positions]:
                    self.close_short_position(grid_price)

    def open_long_position(self, price):
        self.open_positions.append({'price': price, 'amount': self.amount_per_trade / price})
        self.trades.append(('buy', price, self.current_price))

    def close_long_position(self, price):
        for pos in self.open_positions:
            if pos['price'] == price:
                profit = (self.current_price - pos['price']) * pos['amount']
                self.realized_profit += profit
                self.open_positions.remove(pos)
                self.trades.append(('sell', price, self.current_price))
                break

    def open_short_position(self, price):
        self.open_positions.append({'price': price, 'amount': self.amount_per_trade / price})
        self.trades.append(('sell', price, self.current_price))

    def close_short_position(self, price):
        for pos in self.open_positions:
            if pos['price'] == price:
                profit = (pos['price'] - self.current_price) * pos['amount']
                self.realized_profit += profit
                self.open_positions.remove(pos)
                self.trades.append(('buy', price, self.current_price))
                break

    def calculate_unrealized_profit(self):
        self.unrealized_profit = sum(
            [(self.current_price - pos['price']) * pos['amount'] if self.position == "Long" 
             else (pos['price'] - self.current_price) * pos['amount'] 
             for pos in self.open_positions]
        )

    def run_iteration(self, i, total_iterations):
        self.update_price(i, total_iterations)
        self.check_and_execute_trades()
        self.calculate_unrealized_profit()
        self.history.append({
            'price': self.current_price,
            'realized_profit': self.realized_profit,
            'unrealized_profit': self.unrealized_profit,
            'open_positions': len(self.open_positions)
        })

def create_animated_plot(bot, hypothetical_time):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        subplot_titles=('Price and Grids', 'Profit'))

    # Price and Grids
    fig.add_trace(go.Scatter(x=[0], y=[bot.entry_price], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
    for grid in bot.grids:
        fig.add_trace(go.Scatter(x=[0, hypothetical_time], y=[grid, grid], mode='lines', name=f'Grid {grid:.6f}', line=dict(color='gray', dash='dash')), row=1, col=1)

    # Profit
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Realized Profit', line=dict(color='green')), row=2, col=1)
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='lines', name='Unrealized Profit', line=dict(color='orange')), row=2, col=1)

    # Layout
    fig.update_layout(height=800, title_text="Grid Trading Bot Simulation")
    fig.update_xaxes(title_text="Time (hours)", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Profit", row=2, col=1)

    return fig

def update_plot(fig, bot, i, hypothetical_time, total_iterations):
    time_point = hypothetical_time * i / total_iterations

    # Update Price
    fig.data[0].x = fig.data[0].x + (time_point,)
    fig.data[0].y = fig.data[0].y + (bot.current_price,)

    # Update Profits
    fig.data[-2].x = fig.data[-2].x + (time_point,)
    fig.data[-2].y = fig.data[-2].y + (bot.realized_profit,)
    fig.data[-1].x = fig.data[-1].x + (time_point,)
    fig.data[-1].y = fig.data[-1].y + (bot.unrealized_profit,)

    # Add markers for trades
    for trade in bot.trades:
        action, grid_price, executed_price = trade
        color = 'green' if action == 'buy' else 'red'
        symbol = 'triangle-up' if action == 'buy' else 'triangle-down'
        fig.add_trace(go.Scatter(x=[time_point], y=[executed_price], 
                                 mode='markers', 
                                 marker=dict(color=color, size=10, symbol=symbol),
                                 name=f'{action.capitalize()} at {executed_price:.6f}'), row=1, col=1)

    # Clear trades after adding to plot
    bot.trades.clear()

    return fig

def run_simulation(bot, progress_bar, hypothetical_time, plot_placeholder, total_iterations=1000):
    fig = create_animated_plot(bot, hypothetical_time)
    
    for i in range(total_iterations):
        bot.run_iteration(i, total_iterations)
        progress_bar.progress((i + 1) / total_iterations)
        
        if i % 10 == 0:  # Update plot every 10 iterations for performance
            fig = update_plot(fig, bot, i, hypothetical_time, total_iterations)
            plot_placeholder.plotly_chart(fig, use_container_width=True)
        
        if bot.realized_profit >= bot.take_profit or bot.realized_profit <= -bot.stop_loss:
            break
        
        time.sleep(0.01)

def main():
    st.title("Grid Trading Bot - Profit Projections")

    st.sidebar.header("Bot Parameters")
    symbol = st.sidebar.text_input("Symbol", "RLC/USDT")
    position = st.sidebar.selectbox("Position", ["Long", "Short"])
    amount_per_trade = st.sidebar.number_input("Amount per Trade")
    lower_price = st.sidebar.number_input("Lower Price", format="%.5f")
    upper_price = st.sidebar.number_input("Upper Price", format="%.5f")
    grid_profit_percentage = st.sidebar.number_input("Grid Profit Percentage")
    take_profit = st.sidebar.number_input("Take Profit")
    stop_loss = st.sidebar.number_input("Stop Loss")
    entry_price = st.sidebar.number_input("Entry Price", min_value=lower_price, max_value=upper_price, value=(lower_price+upper_price)/2, format="%.5f")
    hypothetical_time = st.sidebar.number_input("Hypothetical Time (hours)", min_value=1, value=24)

    if st.sidebar.button("Run Simulation"):
        bot = GridTradingBot(symbol, position, amount_per_trade, lower_price, upper_price, 
                             grid_profit_percentage, take_profit, stop_loss, entry_price)
        
        progress_bar = st.progress(0)
        plot_placeholder = st.empty()
        
        run_simulation(bot, progress_bar, hypothetical_time, plot_placeholder)

        df = pd.DataFrame(bot.history)

        st.subheader("Simulation Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Realized Profit", f"{bot.realized_profit:.2f}")
        col2.metric("Final Unrealized Profit", f"{bot.unrealized_profit:.2f}")
        col3.metric("Final Open Positions", len(bot.open_positions))

        st.subheader("Open Positions Over Time")
        st.line_chart(df['open_positions'])

        st.subheader("Profit Over Time")
        st.line_chart(df[['realized_profit', 'unrealized_profit']])

        st.subheader("Raw Data")
        st.dataframe(df)

if __name__ == "__main__":
    main()