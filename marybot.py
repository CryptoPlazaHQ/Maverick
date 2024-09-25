import os
import discord
from discord.ext import tasks, commands
import pandas as pd
import asyncio
import platform
from tradingview_ta import TA_Handler, Interval
from decimal import Decimal
import plotly.graph_objects as go
import io
from datetime import datetime, timezone
import logging
from dotenv import load_dotenv
import json
import random

# Load environment variables
load_dotenv()

# Configuration
DISCORD_BOT_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID'))

# Load quotes
with open('quotes.json', 'r', encoding='utf-8') as f:
    quotes = json.load(f)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# List of symbols to analyze (you can modify this list)
symbols = [
    "10000LADYSUSDT.P", "10000NFTUSDT.P", "1000BONKUSDT.P", "1000BTTUSDT.P", "1000BEERUSDT.P",
    "1000FLOKIUSDT.P", "1000LUNCUSDT.P", "1000PEPEUSDT.P", "1000XECUSDT.P", "1000000MOGUSDT.P",
    "1INCHUSDT.P", "AAVEUSDT.P", "ACHUSDT.P", "ADAUSDT.P", "AGLDUSDT.P", "AVAILUSDT.P",
    "AKROUSDT.P", "ALGOUSDT.P", "ALICEUSDT.P", "ALPACAUSDT.P", "1000APUUSDT.P", "1CATUSDT.P",
    "ALPHAUSDT.P", "AMBUSDT.P", "ANKRUSDT.P", "APEUSDT.P", "API3USDT.P", "A8USDT.P",
    "APTUSDT.P", "ARUSDT.P", "ARBUSDT.P", "ARKUSDT.P", "DOP1USDT.P", "1000RATSUSDT.P",
    "ARKMUSDT.P", "ARPAUSDT.P", "ASTRUSDT.P", "ATAUSDT.P", "ATOMUSDT.P", 
    "AUCTIONUSDT.P", "AUDIOUSDT.P", "AVAXUSDT.P", "AXSUSDT.P", "BADGERUSDT.P", 
    "BAKEUSDT.P", "BALUSDT.P", "BANDUSDT.P", "BATUSDT.P", "BCHUSDT.P", 
    "BELUSDT.P", "BICOUSDT.P", "BIGTIMEUSDT.P", "BLURUSDT.P", "BLZUSDT.P", "CETUSUSDT.P",
    "BTCUSDT.P", "C98USDT.P", "CEEKUSDT.P", "CELOUSDT.P", "CELRUSDT.P", "CFXUSDT.P",
    "CHRUSDT.P", "CHZUSDT.P", "CKBUSDT.P", "COMBOUSDT.P", "COMPUSDT.P", "DRIFTUSDT.P",
    "COREUSDT.P", "COTIUSDT.P", "CROUSDT.P", "CRVUSDT.P", "CTCUSDT.P", "DEGENUSDT.P",
    "CTKUSDT.P", "CTSIUSDT.P", "CVCUSDT.P", "CVXUSDT.P", "CYBERUSDT.P", "DARUSDT.P",
    "DASHUSDT.P", "DENTUSDT.P", "DGBUSDT.P", "DODOUSDT.P", "DOGEUSDT.P", "DOTUSDT.P",
    "DUSKUSDT.P", "DYDXUSDT.P", "EDUUSDT.P", "EGLDUSDT.P", "ENJUSDT.P", "ENSUSDT.P",
    "EOSUSDT.P", "ETCUSDT.P", "ETHUSDT.P", "ETHWUSDT.P", "FILUSDT.P", "DOGUSDT.P", "FIREUSDT.P",
    "FITFIUSDT.P", "FLOWUSDT.P", "FLRUSDT.P", "FORTHUSDT.P", "FRONTUSDT.P", "FTMUSDT.P",
    "FXSUSDT.P", "GALAUSDT.P", "GFTUSDT.P", "GLMUSDT.P", "BENDOGUSDT.P", "L3USDT.P",
    "GLMRUSDT.P", "GMTUSDT.P", "GMXUSDT.P", "GRTUSDT.P", "GTCUSDT.P", "HBARUSDT.P", 
    "HFTUSDT.P", "HIFIUSDT.P", "HIGHUSDT.P", "HNTUSDT.P", "PENGUSDT.P", "1000000PEIPEIUSDT.P",
    "HOOKUSDT.P", "HOTUSDT.P", "ICPUSDT.P", "ICXUSDT.P", "IDUSDT.P", "IDEXUSDT.P",
    "ILVUSDT.P", "IMXUSDT.P", "INJUSDT.P", "IOSTUSDT.P", "IOTAUSDT.P", "IOTXUSDT.P",
    "JASMYUSDT.P", "JOEUSDT.P", "JSTUSDT.P", "KASUSDT.P", "KAVAUSDT.P", "KDAUSDT.P",
    "KEYUSDT.P", "KLAYUSDT.P", "KNCUSDT.P", "KSMUSDT.P", "LDOUSDT.P", "LEVERUSDT.P",
    "LINAUSDT.P", "LINKUSDT.P", "LITUSDT.P", "LOOKSUSDT.P", "LOOMUSDT.P", "LPTUSDT.P", "LAIUSDT.P",
    "LQTYUSDT.P", "LRCUSDT.P", "LTCUSDT.P", "LUNA2USDT.P", "MAGICUSDT.P", "MOTHERUSDT.P", "MYRIAUSDT.P",
    "MANAUSDT.P", "MASKUSDT.P", "MATICUSDT.P", "MAVUSDT.P", "MDTUSDT.P", "POPCATUSDT.P", "MANEKIUSDT.P",
    "MINAUSDT.P", "MKRUSDT.P", "MNTUSDT.P", "MTLUSDT.P", "NEARUSDT.P", "PIXFIUSDT.P",
    "NEOUSDT.P", "NKNUSDT.P", "NMRUSDT.P", "NTRNUSDT.P", "NEIROETHUSDT.P", "OGUSDT.P",
    "OGNUSDT.P", "OMGUSDT.P", "ONEUSDT.P", "ONTUSDT.P", "OPUSDT.P", "ORBSUSDT  .P", "ORDERUSDT.P",
    "ORDIUSDT.P", "OXTUSDT.P", "PAXGUSDT.P", "PENDLEUSDT.P", "PEOPLEUSDT.P", "PERPUSDT.P",
    "PHBUSDT.P", "PROMUSDT.P", "PONKEUSDT.P", "QNTUSDT.P", "QTUMUSDT.P", "RADUSDT.P", "RDNTUSDT.P", 
    "REEFUSDT.P", "RENUSDT.P", "REQUSDT.P", "RLCUSDT.P", "ROSEUSDT.P", "SCAUSDT.P", "SAGAUSDT.P",
    "RPLUSDT.P", "RSRUSDT.P", "RSS3USDT.P", "RUNEUSDT.P", "RVNUSDT.P", "SUNDOGUSDT.P", "SAFEUSDT.P",
    "SANDUSDT.P", "SCUSDT.P", "SCRTUSDT.P", "SEIUSDT.P", "SFPUSDT.P", "SHIB1000USDT.P", "SILLYUSDT.P",
    "SKLUSDT.P", "SLPUSDT.P", "SNXUSDT.P", "SOLUSDT.P", "SPELLUSDT.P", "SSVUSDT.P", "PRCLUSDT.P",
    "STGUSDT.P", "STMXUSDT.P", "STORJUSDT.P", "STPTUSDT.P", "STXUSDT.P", "SUIUSDT.P", "SLFUSDT.P",
    "SUNUSDT.P", "SUSHIUSDT.P", "SWEATUSDT.P", "SXPUSDT.P", "UXLINKUSDT.P", "PIRATEUSDT.P",
    "TUSDT.P", "THETAUSDT.P", "TLMUSDT.P", "TOMIUSDT.P", "TONUSDT.P", "STRKUSDT.P",
    "TRBUSDT.P", "TRUUSDT.P", "TRXUSDT.P", "TWTUSDT.P", "UMAUSDT.P", "UNFIUSDT.P",
    "UNIUSDT.P", "USDCUSDT.P", "VETUSDT.P", "VGXUSDT.P", "VRAUSDT.P",
    "WAVESUSDT.P", "WAXPUSDT.P", "WLDUSDT.P", "WOOUSDT.P", "XCNUSDT.P", "ZCXUSDT.P",
    "XEMUSDT.P", "XLMUSDT.P", "XMRUSDT.P", "XNOUSDT.P", "XRPUSDT.P", "XTZUSDT.P", "ZBCNUSDT.P",
    "XVGUSDT.P", "XVSUSDT.P", "YFIUSDT.P", "YGGUSDT.P", "ZECUSDT.P", "ZENUSDT.P", "ZILUSDT.P", "ZRXUSDT.P"
]

def float_to_decimal(value):
    """Convert float to Decimal."""
    return Decimal(str(value))

async def fetch_data(symbol):
    """Fetch trading data for a given symbol."""
    try:
        handler = TA_Handler(
            symbol=symbol,
            exchange="BYBIT",
            screener="crypto",
            interval=Interval.INTERVAL_1_DAY,
            timeout=None
        )
        return await asyncio.to_thread(handler.get_analysis)
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {e}")
        return None

def calculate_pivot_points(high, low, close):
    """Calculate pivot points and levels."""
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
    """Generate trading recommendation based on pivot points."""
    distance_to_r1 = (r1 - current_price) / current_price * Decimal('100')
    distance_to_s1 = (current_price - s1) / current_price * Decimal('100')
    
    if distance_to_r1 > Decimal('4.5') and distance_to_s1 < Decimal('3.0'):
        return "LONG"
    elif distance_to_s1 > Decimal('4.5') and distance_to_r1 < Decimal('3.0'):
        return "SHORT"
    return "NEUTRAL"

def calculate_weighted_atr(high, low, close, volume):
    """Calculate Weighted ATR."""
    true_range = max(high - low, abs(high - close), abs(low - close))
    weighted_tr = true_range * volume
    return weighted_tr / volume if volume != 0 else 0

def optimize_grid_settings(current_price, atr, position_type, s1, r1):
    """Optimize grid settings based on ATR and pivot points."""
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

def calculate_grid_profit(grid_size, current_price):
    """Calculate grid profit percentage."""
    return (grid_size / current_price) * Decimal('100')

async def analyze_symbol(symbol):
    """Analyze a single symbol and return the results."""
    analysis = await fetch_data(symbol)
    if analysis is None:
        return None

    indicators = analysis.indicators
    current_price = float_to_decimal(indicators['close'])
    high = float_to_decimal(indicators['high'])
    low = float_to_decimal(indicators['low'])
    volume = float_to_decimal(indicators['volume'])
    
    pivot_points = calculate_pivot_points(high, low, current_price)
    recommendation = get_recommendation(current_price, pivot_points['r1'], pivot_points['s1'])
    
    atr = calculate_weighted_atr(high, low, current_price, volume)
    grid_settings = optimize_grid_settings(current_price, atr, recommendation, pivot_points['s1'], pivot_points['r1'])
    grid_profit = calculate_grid_profit(grid_settings['grid_size'], current_price)
    
    return {
        'Symbol': symbol,
        'Price': current_price,
        'RSI': indicators['RSI'],
        'Recommendation': recommendation,
        'Entry Point': grid_settings['entry_point'],
        'Exit Point': grid_settings['exit_point'],
        'Stop Loss': grid_settings['stop_loss'],
        'Take Profit': grid_settings['take_profit'],
        'Grid Size': grid_settings['grid_size'],
        'Grid Profit': grid_profit,
        'ATR': atr
    }

def create_scatter_plot(df):
    """Create a scatter plot of RSI values."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Symbol'],
        y=df['RSI'],
        mode='markers',
        marker=dict(size=10, color=df['RSI'], colorscale='RdYlGn', showscale=True),
        text=df['Symbol'],
        hoverinfo='text+y'
    ))
    fig.update_layout(
        title='Market Momentum Density',
        xaxis_title='Symbols',
        yaxis_title='RSI',
        showlegend=False
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    
    img_bytes = fig.to_image(format="png")
    return io.BytesIO(img_bytes)

def format_quote(quote):
    """Format the quote for better readability in Discord."""
    return f"**{quote['quote']}**\n- *{quote['author']}*"

async def send_alerts():
    """Fetch data, generate recommendations, and send alerts."""
    try:
        tasks = [analyze_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        results = [result for result in results if result is not None]
        
        df = pd.DataFrame(results)
        long_df = df[df['Recommendation'] == 'LONG'].sort_values(by='RSI', ascending=False)
        short_df = df[df['Recommendation'] == 'SHORT'].sort_values(by='RSI', ascending=True)
        
        embed = discord.Embed(title="🚨 Grid Bot Alert | Maverick Book", color=0x00ff00)
        embed.set_thumbnail(url="https://raw.githubusercontent.com/CryptoPlazaHQ/Stock/main/cryptocoin_logo_color.png")

        if not long_df.empty:
            long_list = []
            for _, row in long_df.iterrows():
                long_list.append(
                    f"**{row['Symbol']}**\n"
                    f"💰 Price: ${row['Price']:.4f}\n"
                    f"📊 RSI: {row['RSI']:.2f}\n"
                    f"🎯 Entry: ${row['Entry Point']:.4f}\n"
                    f"🏁 Exit: ${row['Exit Point']:.4f}\n"
                    f"🛑 Stop Loss: ${row['Stop Loss']:.4f}\n"
                    f"💹 Grid Profit: {row['Grid Profit']:.2f}%\n"
                )
            embed.add_field(name="🚀 LONG Recommendations", value="\n".join(long_list[:5]), inline=False)
        
        if not short_df.empty:
            short_list = []
            for _, row in short_df.iterrows():
                short_list.append(
                    f"**{row['Symbol']}**\n"
                    f"💰 Price: ${row['Price']:.4f}\n"
                    f"📊 RSI: {row['RSI']:.2f}\n"
                    f"🎯 Entry: ${row['Entry Point']:.4f}\n"
                    f"🏁 Exit: ${row['Exit Point']:.4f}\n"
                    f"🛑 Stop Loss: ${row['Stop Loss']:.4f}\n"
                    f"💹 Grid Profit: {row['Grid Profit']:.2f}%\n"
                )
            embed.add_field(name="🔻 SHORT Recommendations", value="\n".join(short_list[:5]), inline=False)
        
        current_time = datetime.now(timezone.utc)
        embed.set_footer(
            text=f"Last Update: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            icon_url="https://raw.githubusercontent.com/CryptoPlazaHQ/Stock/main/cryptoplaza_logo_white.png"
        )

        # Add a random quote if no recommendations are available
        if long_df.empty and short_df.empty:
            quote = random.choice(quotes)
            formatted_quote = format_quote(quote)
            embed.add_field(name="📜 Maverick's Wisdom", value=formatted_quote, inline=False)

        channel = bot.get_channel(CHANNEL_ID)
        if channel is None:
            logger.error(f"Error: Could not find channel with ID {CHANNEL_ID}.")
            return
        
        # Create and send scatter plot
        plot_img = create_scatter_plot(df)
        plot_file = discord.File(plot_img, filename="rsi_scatter.png")
        
        await channel.send(embed=embed, file=plot_file)
        logger.info(f"Alert sent successfully at {current_time}")
        
    except Exception as e:
        logger.error(f"An error occurred in send_alerts: {e}")

@bot.event
async def on_ready():
    """Event handler for when the bot is ready."""
    logger.info(f'{bot.user} has connected to Discord!')
    for guild in bot.guilds:
        logger.info(f"- {guild.name} (id: {guild.id})")
        for channel in guild.text_channels:
            logger.info(f"  - #{channel.name} (id: {channel.id})")
    
    # Start the scheduled task
    if not scheduled_alerts.is_running():
        scheduled_alerts.start()

@tasks.loop(minutes=12)
async def scheduled_alerts():
    """Send alerts every 12 minutes."""
    logger.info("Running scheduled alerts...")
    await send_alerts()

@bot.command(name='force_alert')
async def force_alert(ctx):
    """Force an immediate alert."""
    logger.info("Forcing an immediate alert...")
    await send_alerts()
    await ctx.send("Alert sent!")

def run_bot():
    """Start the Discord bot.""" 
    try:
        bot.run(DISCORD_BOT_TOKEN)
    except KeyboardInterrupt:
        logger.info("Bot is shutting down...")

if __name__ == "__main__":
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    run_bot()
