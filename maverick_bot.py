# Standard library imports
import os
import sys
import logging
import asyncio
import platform
import math
from typing import Dict, List, Optional, Tuple, Set, Any, Generator
from pathlib import Path
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from dataclasses import dataclass
from enum import Enum
import base64
import io
import traceback
import random

# Add the Windows-specific event loop policy right here
if platform.system() == 'Windows':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Third-party imports
import discord
from discord.ext import commands
import pandas as pd
import numpy as np
from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Configure logging
def setup_logging():
    """Configure application logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    log_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(
        f'logs/maverick_{datetime.now().strftime("%Y%m%d")}.log'
    )
    file_handler.setFormatter(log_format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

# Custom Exceptions
class MaverickError(Exception):
    """Base exception for Maverick bot"""
    pass

class ConfigurationError(MaverickError):
    """Configuration related errors"""
    pass

class AnalysisError(MaverickError):
    """Analysis related errors"""
    pass

class ValidationError(MaverickError):
    """Validation related errors"""
    pass

@dataclass
class GridLevel:
    """
    Unified grid level class for Maverick trading system.
    
    Attributes:
        price: The price level
        type: Type of level ('entry', 'tp', 'sl', 'midpoint', 'support', 'resistance')
        status: Current status ('active', 'triggered', 'cancelled')
        strength: Level strength from 1-5
        timeframe: Timeframe this level belongs to
        description: Human-readable description of the level
        created_at: When this level was created
    """
    price: Decimal
    type: str
    status: str = 'active'
    strength: int = 3
    timeframe: str = '4H'
    description: str = ''
    created_at: datetime = datetime.now()

    def __eq__(self, other):
        return isinstance(other, GridLevel) and self.price == other.price
        
    def __hash__(self):
        return hash(self.price)
    
@dataclass
class PriceAlert:
    """
    Represents a price alert configuration
    
    Attributes:
        price: Target price level
        direction: Direction of price movement ('above' or 'below')
        message: Alert message to display
        channel_id: Discord channel ID for notification
        user_id: Discord user ID who set the alert
        created_at: When the alert was created
        expires_at: When the alert should expire
        triggered: Whether the alert has been triggered
    """
    price: Decimal
    direction: str  # 'above' or 'below'
    message: str
    channel_id: int
    user_id: int
    created_at: datetime
    expires_at: datetime
    triggered: bool = False

    def __eq__(self, other):
        if not isinstance(other, PriceAlert):
            return False
        return (self.price == other.price and 
                self.user_id == other.user_id and 
                self.direction == other.direction)
    
    def __hash__(self):
        return hash((self.price, self.user_id, self.direction))

class Config:
    """Enhanced configuration management for Maverick Trading Bot"""
    
    # Analysis Settings
    TIMEFRAMES: List[str] = ['4H', '1H', '15M']
    MAX_IMAGE_SIZE: int = 2 * 1024 * 1024  # 2MB
    SUPPORTED_FORMATS: List[str] = ['JPEG', 'PNG', 'GIF', 'WEBP']
    
    # Trading Parameters
    SETUP_TYPES: List[str] = ['Bounce', 'Pullback', 'Reversal', 'Breakout']
    MIN_RR_RATIO: Decimal = Decimal('1.5')
    
    # API Settings
    DISCORD_TOKEN: Optional[str] = os.getenv('DISCORD_BOT_TOKEN')
    ANTHROPIC_API_KEY: Optional[str] = os.getenv('ANTHROPIC_API_KEY')
    
    # Analysis Parameters
    RSI_PERIOD: int = 14
    ATR_PERIOD: int = 14
    VOLUME_MA_PERIOD: int = 20
    
    # Grid Parameters
    GRID_SETTINGS = {
        'low_volatility': {
            'entry_spacing': Decimal('0.5'),
            'tp_spacing': Decimal('1.0'),
            'sl_spacing': Decimal('1.0')
        },
        'normal_volatility': {
            'entry_spacing': Decimal('0.75'),
            'tp_spacing': Decimal('1.5'),
            'sl_spacing': Decimal('1.25')
        },
        'high_volatility': {
            'entry_spacing': Decimal('1.0'),
            'tp_spacing': Decimal('2.0'),
            'sl_spacing': Decimal('1.5')
        }
    }
    
    # Alert Settings
    MAX_ALERTS_PER_USER: int = 10
    ALERT_CHECK_INTERVAL: int = 60  # seconds
    ALERT_CLEANUP_INTERVAL: int = 3600  # 1 hour
    MAX_ALERT_DURATION: int = 7 * 24 * 60 * 60  # 7 days in seconds
    
    # Logging Settings
    LOG_FORMAT: str = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    LOG_FILE_PATH: str = 'logs/maverick_bot_{date}.log'
    
    @classmethod
    def validate(cls) -> None:
        """Validate all configuration settings"""
        errors = []
        
        if not cls.DISCORD_TOKEN:
            errors.append("DISCORD_BOT_TOKEN not configured")
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY not configured")
            
        if not all(tf in ['1M', '5M', '15M', '30M', '1H', '4H', '1D'] for tf in cls.TIMEFRAMES):
            errors.append("Invalid timeframes specified")
            
        if cls.MAX_IMAGE_SIZE > 5 * 1024 * 1024:  # Max 5MB
            errors.append("MAX_IMAGE_SIZE exceeds 5MB limit")
            
        if cls.MIN_RR_RATIO < Decimal('1.0'):
            errors.append("MIN_RR_RATIO must be at least 1.0")
            
        if cls.RSI_PERIOD < 2:
            errors.append("RSI_PERIOD must be at least 2")
        if cls.ATR_PERIOD < 2:
            errors.append("ATR_PERIOD must be at least 2")
        if cls.VOLUME_MA_PERIOD < 2:
            errors.append("VOLUME_MA_PERIOD must be at least 2")
            
        for volatility, settings in cls.GRID_SETTINGS.items():
            if not all(isinstance(v, Decimal) for v in settings.values()):
                errors.append(f"Invalid grid settings for {volatility}")
                
        if cls.MAX_ALERTS_PER_USER < 1:
            errors.append("MAX_ALERTS_PER_USER must be positive")
        if cls.ALERT_CHECK_INTERVAL < 10:
            errors.append("ALERT_CHECK_INTERVAL must be at least 10 seconds")
            
        if errors:
            raise ValueError("\n".join(errors))
    
    @classmethod
    def get_log_file_path(cls) -> str:
        """Get formatted log file path with current date"""
        return cls.LOG_FILE_PATH.format(
            date=datetime.now().strftime('%Y%m%d')
        )
    
    @classmethod
    def get_grid_settings(cls, volatility: str) -> dict:
        """Get grid settings for given volatility level"""
        if volatility not in cls.GRID_SETTINGS:
            raise ValueError(f"Invalid volatility level: {volatility}")
        return cls.GRID_SETTINGS[volatility]
    
    @classmethod
    def is_valid_timeframe(cls, timeframe: str) -> bool:
        """Check if timeframe is valid"""
        return timeframe in cls.TIMEFRAMES
    
    @classmethod
    def is_valid_setup_type(cls, setup_type: str) -> bool:
        """Check if setup type is valid"""
        return setup_type.title() in cls.SETUP_TYPES

# Enums for market analysis
class MarketPhase(Enum):
    CONTRACTION = "Contraction"
    EXPANSION = "Expansion"
    TREND = "Trend"

class SetupType(Enum):
    BOUNCE = "Bounce Back"
    PULLBACK = "Pullback"
    REVERSAL = "Reversal"
    BREAKOUT = "Breakout"

@dataclass
class SetupConfirmation:
    name: str
    status: bool
    description: str
    timeframe: str

class AnalysisState:
    """Tracks state of ongoing analyses"""
    def __init__(self, pair: str):
        self.pair = pair
        self.timeframes: Dict[str, bool] = {tf: False for tf in Config.TIMEFRAMES}
        self.images: List[Tuple[str, str]] = []  # [(base64_data, mime_type)]
        self.phase: Optional[str] = None
        self.setup_type: Optional[str] = None
        self.last_update = datetime.now()
        self.current_price: Optional[Decimal] = None
        self.atr: Optional[Decimal] = None
        self.supports: List[Decimal] = []
        self.resistances: List[Decimal] = []
        self.historical_data: Optional[pd.DataFrame] = None
        self.analysis: Dict[str, Any] = {}

class MaverickBot:
    """Core bot class handling Discord and Analysis coordination"""
    
    def __init__(self):
        """Initialize MaverickBot instance"""
        self.anthropic: Optional[AsyncAnthropic] = None
        self.analyzer: Optional['MaverickAnalyzer'] = None
        self.grid_calculator: Optional['MaverickGridCalculator'] = None
        self.alert_manager: Optional['AlertManager'] = None
        self.backtester: Optional['MaverickBacktester'] = None
        self.active_analyses: Dict[int, AnalysisState] = {}
        self._setup_validator: Optional['MaverickSetupValidator'] = None
        
    async def initialize(self) -> None:
        """Asynchronous initialization of bot components"""
        try:
            # Initialize API client
            self.anthropic = AsyncAnthropic(
                api_key=Config.ANTHROPIC_API_KEY
            )
            
            # Initialize components
            self.analyzer = MaverickAnalyzer()
            self.grid_calculator = MaverickGridCalculator()
            self.alert_manager = AlertManager(self)
            self.backtester = MaverickBacktester()
            
            # Test API connections
            await self._test_connections()
            
            logger.info("MaverickBot initialized successfully")
            
        except Exception as e:
            logger.error(f"Initialization failed: {traceback.format_exc()}")
            raise ConfigurationError(f"Bot initialization failed: {str(e)}")

    async def process_image(self, image_data: bytes, filename: str) -> Tuple[str, str]:
        """
        Process and validate uploaded chart image
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            
        Returns:
            Tuple of (base64_encoded_data, mime_type)
        """
        try:
            # Open and validate image
            image = Image.open(io.BytesIO(image_data))
            
            # Validate format
            if image.format not in Config.SUPPORTED_FORMATS:
                raise ValidationError(
                    f"Unsupported format: {image.format}. "
                    f"Supported formats: {', '.join(Config.SUPPORTED_FORMATS)}"
                )
                
            # Validate size
            if len(image_data) > Config.MAX_IMAGE_SIZE:
                raise ValidationError(
                    f"Image too large. Maximum size: "
                    f"{Config.MAX_IMAGE_SIZE / (1024*1024)}MB"
                )
                
            # Convert to base64
            b64_data = base64.b64encode(image_data).decode('utf-8')
            mime_type = f"image/{image.format.lower()}"
            
            return b64_data, mime_type
            
        except ValidationError:
            raise
        except Exception as e:
            logger.error(f"Image processing error: {traceback.format_exc()}")
            raise AnalysisError(f"Failed to process image: {str(e)}")

    async def analyze_chart(self, state: AnalysisState) -> Dict:
        """
        Analyze chart using Claude Vision API
        
        Args:
            state: Current analysis state containing images and context
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Create analysis prompt
            prompt = self._create_analysis_prompt(state)
            
            # Prepare messages with images
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime_type,
                            "data": b64_data
                        }
                    } for b64_data, mime_type in state.images
                ] + [{
                    "type": "text",
                    "text": prompt
                }]
            }]
            
            # Get Claude's analysis
            response = await self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=messages
            )
            
            # Parse and enhance analysis
            analysis = self._parse_analysis(response.content[0].text)
            
            # Store analysis in state
            state.analysis = analysis
            state.phase = analysis.get('phase')
            state.current_price = self._extract_current_price(analysis)
            
            # Extract and store levels
            levels = analysis.get('levels', {})
            state.supports = [Decimal(str(p)) for p in levels.get('support', [])]
            state.resistances = [Decimal(str(p)) for p in levels.get('resistance', [])]
            
            return analysis
            
        except Exception as e:
            logger.error(f"Analysis error: {traceback.format_exc()}")
            raise AnalysisError(f"Chart analysis failed: {str(e)}")

    def _create_analysis_prompt(self, state: AnalysisState) -> str:
        """Create detailed analysis prompt for Claude"""
        return f"""Analyze these {state.pair} charts using the Maverick framework.

Key Analysis Requirements:

1. Market Phase Identification
- Current phase (Contraction/Expansion/Trend)
- Phase characteristics and evidence
- Phase transition signals if any

2. Technical Structure
- Key support/resistance levels (with price values)
- Volume profile analysis
- RSI conditions and divergences
- Market structure quality

3. Setup Classification
- Identify any valid setups (Bounce/Pullback/Reversal/Breakout)
- Setup quality assessment
- Required confirmations
- Risk/Reward potential

4. Entry/Exit Points
- Key entry levels
- Stop loss zones
- Take profit targets
- Position sizing considerations

Please provide specific price levels and clear confirmation signals for all identified setups.
Prioritize high-probability setups that align with the current market phase."""

    def _parse_analysis(self, response: str) -> Dict:
        """
        Parse Claude's analysis response into structured data
        
        Args:
            response: Raw text response from Claude
            
        Returns:
            Dictionary containing parsed analysis components
        """
        analysis = {
            'phase': self._extract_phase(response),
            'levels': self._extract_levels(response),
            'setup': self._extract_setup(response),
            'risk_reward': self._extract_rr(response),
            'volume_analysis': self._extract_volume_analysis(response),
            'momentum': self._extract_momentum_analysis(response),
            'structure': self._extract_structure_analysis(response)
        }
        
        return analysis

    def _extract_phase(self, text: str) -> str:
        """Extract market phase from analysis text"""
        phase_map = {
            'contraction': MarketPhase.CONTRACTION.value,
            'expansion': MarketPhase.EXPANSION.value,
            'trend': MarketPhase.TREND.value
        }
        
        text_lower = text.lower()
        for key, value in phase_map.items():
            if key in text_lower:
                if any(indicator in text_lower for indicator in [
                    'compression', 'consolidation', 'range bound'
                ]):
                    return MarketPhase.CONTRACTION.value
                elif any(indicator in text_lower for indicator in [
                    'breakout', 'momentum', 'volume surge'
                ]):
                    return MarketPhase.EXPANSION.value
                elif any(indicator in text_lower for indicator in [
                    'trending', 'higher highs', 'lower lows'
                ]):
                    return MarketPhase.TREND.value
        
        return "Unknown"

    def _extract_levels(self, text: str) -> Dict[str, List[float]]:
        """Extract price levels from analysis text"""
        import re
        
        levels = {
            'support': [],
            'resistance': []
        }
        
        # Look for price levels in the text
        price_pattern = r'(\d+\.?\d*)'
        
        # Extract support levels
        support_matches = re.finditer(
            rf'support.*?{price_pattern}|{price_pattern}.*?support',
            text.lower()
        )
        levels['support'] = [
            float(match.group(1)) for match in support_matches
        ]
        
        # Extract resistance levels
        resistance_matches = re.finditer(
            rf'resistance.*?{price_pattern}|{price_pattern}.*?resistance',
            text.lower()
        )
        levels['resistance'] = [
            float(match.group(1)) for match in resistance_matches
        ]
        
        return levels

    def _extract_setup(self, text: str) -> Dict:
        """Extract setup information from analysis text"""
        setup_info = {
            'type': None,
            'quality': 'standard',
            'confirmations': []
        }
        
        text_lower = text.lower()
        
        # Check for setup types
        for setup in SetupType:
            if setup.name.lower() in text_lower:
                setup_info['type'] = setup
                break
        
        # Assess quality
        if any(word in text_lower for word in ['high quality', 'strong', 'excellent']):
            setup_info['quality'] = 'high'
        elif any(word in text_lower for word in ['weak', 'poor', 'low quality']):
            setup_info['quality'] = 'low'
        
        return setup_info

    def _extract_rr(self, text: str) -> Optional[float]:
        """Extract risk-reward ratio from analysis text"""
        import re
        
        # Look for R:R mentions
        rr_patterns = [
            r'R:?R.*?(\d+\.?\d*):1',
            r'risk[- ]reward.*?(\d+\.?\d*):1',
            r'reward[- ]risk.*?(\d+\.?\d*):1'
        ]
        
        for pattern in rr_patterns:
            match = re.search(pattern, text.lower())
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        
        return None

    def _extract_volume_analysis(self, text: str) -> Dict:
        """Extract volume analysis from text"""
        return {
            'trend': self._detect_volume_trend(text),
            'state': self._detect_volume_state(text),
            'quality': self._assess_volume_quality(text)
        }

    def _extract_momentum_analysis(self, text: str) -> Dict:
        """Extract momentum analysis from text"""
        return {
            'rsi_state': self._detect_rsi_state(text),
            'strength': self._assess_momentum_strength(text),
            'divergence': self._detect_divergence(text)
        }

    def _extract_structure_analysis(self, text: str) -> Dict:
        """Extract market structure analysis from text"""
        return {
            'state': self._detect_structure_state(text),
            'quality': self._assess_structure_quality(text),
            'trend': self._detect_structure_trend(text)
        }

    def _detect_volume_trend(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['increasing volume', 'volume surge']):
            return 'increasing'
        elif any(word in text_lower for word in ['decreasing volume', 'volume decline']):
            return 'decreasing'
        return 'stable'

    def _detect_volume_state(self, text: str) -> str:
        text_lower = text.lower()
        if 'above average' in text_lower:
            return 'above_average'
        elif 'below average' in text_lower:
            return 'below_average'
        return 'average'

    def _assess_volume_quality(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['clean volume', 'strong volume']):
            return 'high'
        elif any(word in text_lower for word in ['weak volume', 'poor volume']):
            return 'low'
        return 'medium'

    def _detect_rsi_state(self, text: str) -> str:
        text_lower = text.lower()
        if 'oversold' in text_lower:
            return 'oversold'
        elif 'overbought' in text_lower:
            return 'overbought'
        return 'neutral'

    def _assess_momentum_strength(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['strong momentum', 'powerful']):
            return 'strong'
        elif any(word in text_lower for word in ['weak momentum', 'lacking']):
            return 'weak'
        return 'moderate'

    def _detect_divergence(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        if 'bullish divergence' in text_lower:
            return 'bullish'
        elif 'bearish divergence' in text_lower:
            return 'bearish'
        return None

    def _detect_structure_state(self, text: str) -> str:
        text_lower = text.lower()
        if 'breakout' in text_lower:
            return 'breakout'
        elif 'range' in text_lower:
            return 'range'
        return 'trending'

    def _assess_structure_quality(self, text: str) -> str:
        text_lower = text.lower()
        if any(word in text_lower for word in ['clean structure', 'clear structure']):
            return 'high'
        elif any(word in text_lower for word in ['messy', 'unclear']):
            return 'low'
        return 'medium'

    def _detect_structure_trend(self, text: str) -> Optional[str]:
        text_lower = text.lower()
        if any(phrase in text_lower for phrase in ['uptrend', 'higher highs']):
            return 'uptrend'
        elif any(phrase in text_lower for phrase in ['downtrend', 'lower lows']):
            return 'downtrend'
        return None

    async def _test_connections(self) -> None:
        """Test all external API connections"""
        try:
            # Test Anthropic API
            test_message = await self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": "Test connection"
                }]
            )
            if not test_message:
                raise ConfigurationError("Failed to connect to Anthropic API")
                
            logger.info("API connections tested successfully")
            
        except Exception as e:
            logger.error(f"Connection test failed: {str(e)}")
            raise ConfigurationError("Failed to establish required connections")

    def get_setup_validator(self) -> 'MaverickSetupValidator':
        """Get or create setup validator instance"""
        if not self._setup_validator:
            self._setup_validator = MaverickSetupValidator()
        return self._setup_validator

    def _extract_current_price(self, analysis: Dict) -> Optional[Decimal]:
        """Extract current price from analysis"""
        if 'current_price' in analysis:
            return Decimal(str(analysis['current_price']))
        
        # Fallback: use middle of recent range
        levels = analysis.get('levels', {})
        supports = levels.get('support', [])
        resistances = levels.get('resistance', [])
        
        if supports and resistances:
            avg_support = sum(supports) / len(supports)
            avg_resistance = sum(resistances) / len(resistances)
            return Decimal(str((avg_support + avg_resistance) / 2))
            
        return None

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Global bot instance
maverick_bot: Optional[MaverickBot] = None

@bot.event
async def on_ready():
    """Bot startup handler"""
    global maverick_bot
    try:
        logger.info(f'Bot connected as {bot.user}')
        maverick_bot = MaverickBot()
        await maverick_bot.initialize()
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        await bot.close()

@bot.command(name='analyze')
@commands.cooldown(1, 30, commands.BucketType.user)  # 30 second cooldown
async def analyze(ctx, pair: str = None):
    """Start chart analysis for a trading pair"""
    if not pair:
        await ctx.send("⚠️ Please provide a trading pair. Example: `!analyze BTCUSDT`")
        return

    try:
        # Create analysis session
        user_id = ctx.author.id
        maverick_bot.active_analyses[user_id] = AnalysisState(pair.upper())
        
        # Create instruction embed
        embed = discord.Embed(
            title=f"📊 Chart Analysis - {pair.upper()}",
            description="Upload your chart images for analysis",
            color=0x4DA3FF
        )
        
        # Add required timeframes field
        embed.add_field(
            name="Required Timeframes",
            value=(
                "🔹 4H - Major trend and structure\n"
                "🔹 1H - Pattern confirmation\n"
                "🔹 15M - Entry precision"
            ),
            inline=False
        )
        
        # Add setup requirements
        embed.add_field(
            name="Setup Requirements",
            value=(
                "✅ Show RSI indicator\n"
                "✅ Include volume data\n"
                "✅ Clear support/resistance levels"
            ),
            inline=False
        )
        
        # Add image guidelines
        embed.add_field(
            name="Image Guidelines",
            value=(
                "📌 Max size: 2MB per image\n"
                "📌 Formats: JPG, PNG, WebP\n"
                "📌 Name format: <PAIR>_<TIMEFRAME>.png"
            ),
            inline=False
        )
        
        await ctx.send(embed=embed)
        logger.info(f"Analysis session started for {ctx.author} - {pair}")
        
    except Exception as e:
        logger.error(f"Analysis start error: {str(e)}")
        await ctx.send("❌ Failed to start analysis session. Please try again.")

@bot.command(name='status')
async def check_status(ctx):
    """Check status of current analysis"""
    user_id = ctx.author.id
    if user_id not in maverick_bot.active_analyses:
        await ctx.send("❌ No active analysis session. Start with `!analyze <pair>`")
        return
        
    state = maverick_bot.active_analyses[user_id]
    
    embed = discord.Embed(
        title="📊 Analysis Status",
        color=0x4DA3FF
    )
    
    # Add pair info
    embed.add_field(
        name="Pair",
        value=state.pair,
        inline=True
    )
    
    # Add timeframe status
    tf_status = ""
    for tf, uploaded in state.timeframes.items():
        emoji = "✅" if uploaded else "❌"
        tf_status += f"{emoji} {tf}\n"
    
    embed.add_field(
        name="Timeframes",
        value=tf_status,
        inline=True
    )
    
    # Add completion status
    completion = sum(1 for tf in state.timeframes.values() if tf)
    embed.add_field(
        name="Completion",
        value=f"{completion}/3 timeframes",
        inline=True
    )
    
    await ctx.send(embed=embed)

@bot.command(name='reset')
async def reset_analysis(ctx):
    """Reset current analysis session"""
    user_id = ctx.author.id
    if user_id in maverick_bot.active_analyses:
        del maverick_bot.active_analyses[user_id]
        await ctx.send("✅ Analysis session reset. Start new analysis with `!analyze <pair>`")
    else:
        await ctx.send("ℹ️ No active session to reset.")

@bot.command(name='setup')
async def setup(ctx, action: str = None):
    """Setup command handler for validation, grid, and alerts"""
    if not action:
        await ctx.send("⚠️ Please specify an action: `validate`, `grid`, or `alerts`\nExample: `!setup validate`")
        return

    user_id = ctx.author.id
    if user_id not in maverick_bot.active_analyses:
        await ctx.send("❌ No active analysis session. Start with `!analyze <pair>`")
        return

    state = maverick_bot.active_analyses[user_id]

    if action.lower() == 'validate':
        await validate_setup(ctx, state)
    elif action.lower() == 'grid':
        await calculate_grid(ctx, state)
    elif action.lower() == 'alerts':
        await setup_alerts(ctx, state)
    else:
        await ctx.send("❌ Invalid action. Use: `validate`, `grid`, or `alerts`")

async def validate_setup(ctx, state: AnalysisState):
    """Validate trading setup"""
    try:
        if not state.analysis:
            await ctx.send("❌ No analysis available. Run analysis first.")
            return

        embed = discord.Embed(
            title="🔍 Setup Validation",
            color=0x4DA3FF
        )

        # Pattern Quality
        pattern_quality = state.analysis.get('setup', {}).get('quality', 'standard')
        embed.add_field(
            name="Pattern Quality",
            value=f"```{pattern_quality.title()}```",
            inline=True
        )

        # Volume Analysis
        volume = state.analysis.get('volume_analysis', {})
        volume_text = f"Trend: {volume.get('trend', 'unknown')}\n"
        volume_text += f"Quality: {volume.get('quality', 'unknown')}"
        embed.add_field(
            name="Volume Analysis",
            value=f"```{volume_text}```",
            inline=True
        )

        # Momentum
        momentum = state.analysis.get('momentum', {})
        momentum_text = f"RSI State: {momentum.get('rsi_state', 'unknown')}\n"
        momentum_text += f"Strength: {momentum.get('strength', 'unknown')}"
        embed.add_field(
            name="Momentum",
            value=f"```{momentum_text}```",
            inline=True
        )

        await ctx.send(embed=embed)

    except Exception as e:
        logger.error(f"Setup validation error: {str(e)}")
        await ctx.send("❌ Error validating setup. Please try again.")

async def calculate_grid(ctx, state: AnalysisState):
    """Calculate trading grid levels"""
    try:
        if not state.analysis:
            await ctx.send("❌ No analysis available. Run analysis first.")
            return

        setup_type = state.analysis.get('setup', {}).get('type')
        if not setup_type:
            await ctx.send("❌ No valid setup detected in analysis.")
            return

        if not state.current_price or not state.atr:
            await ctx.send("❌ Missing price data for grid calculation.")
            return

        # Calculate grid levels
        grid = maverick_bot.grid_calculator.calculate_grid(
            setup_type=setup_type,
            current_price=state.current_price,
            atr=state.atr,
            support_levels=state.supports,
            resistance_levels=state.resistances
        )

        # Create embed for grid levels
        embed = discord.Embed(
            title="📊 Trading Grid",
            description=f"Setup Type: {setup_type}",
            color=0x4DA3FF
        )

        # Add entry levels
        entry_text = "\n".join(f"• {level.price} ({level.description})" 
                              for level in grid['entries'])
        embed.add_field(
            name="Entry Levels",
            value=f"```{entry_text}```",
            inline=False
        )

        # Add stop loss
        stop_text = "\n".join(f"• {level.price} ({level.description})" 
                             for level in grid['stops'])
        embed.add_field(
            name="Stop Loss",
            value=f"```{stop_text}```",
            inline=False
        )

        # Add targets
        target_text = "\n".join(f"• {level.price} ({level.description})" 
                               for level in grid['targets'])
        embed.add_field(
            name="Targets",
            value=f"```{target_text}```",
            inline=False
        )

        await ctx.send(embed=embed)

    except Exception as e:
        logger.error(f"Grid calculation error: {str(e)}")
        await ctx.send("❌ Error calculating grid levels. Please try again.")

async def setup_alerts(ctx, state: AnalysisState):
    """Setup price alerts"""
    try:
        if not state.analysis:
            await ctx.send("❌ No analysis available. Run analysis first.")
            return

        embed = discord.Embed(
            title="⚠️ Price Alerts",
            description="Select alert types to set up:",
            color=0x4DA3FF
        )

        embed.add_field(
            name="Available Alerts",
            value=(
                "1️⃣ Entry Alert\n"
                "2️⃣ Stop Loss Alert\n"
                "3️⃣ Take Profit Alert\n"
                "4️⃣ Custom Level Alert"
            ),
            inline=False
        )

        embed.add_field(
            name="How to Set",
            value="React to this message with the corresponding number to set up each alert type.",
            inline=False
        )

        msg = await ctx.send(embed=embed)
        
        # Add reactions for selection
        reactions = ['1️⃣', '2️⃣', '3️⃣', '4️⃣']
        for reaction in reactions:
            await msg.add_reaction(reaction)

    except Exception as e:
        logger.error(f"Alert setup error: {str(e)}")
        await ctx.send("❌ Error setting up alerts. Please try again.")

@bot.command(name='ask')
async def ask_question(ctx, *, question: str = None):
    """Ask questions about trading, strategy, or current analysis"""
    if not question:
        await ctx.send("❌ Please include your question. Example: `!ask How does the contraction phase work?`")
        return

    try:
        # Get user's current analysis state if any
        user_id = ctx.author.id
        state = maverick_bot.active_analyses.get(user_id)
        
        # Create context-aware prompt
        prompt = f"""As a Maverick Trading Strategy expert, please answer this question:

Question: {question}

Context:
- Strategy focuses on market phases (Contraction, Expansion, Trend)
- Uses institutional order flow analysis
- Grid-based trading approach
{f'- Current analysis shows {state.analysis.get("phase", "Unknown")} phase' if state else ''}

Please provide a clear, educational response focusing on practical application."""

        # Get response from Claude
        response = await maverick_bot.anthropic.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        # Create and send embed response
        embed = discord.Embed(
            title="📚 Maverick Strategy Guide",
            description=response.content[0].text,
            color=0x4DA3FF
        )
        
        # Add context footer if in active analysis
        if state:
            embed.set_footer(text=f"Current Analysis: {state.pair} | Phase: {state.analysis.get('phase', 'Unknown')}")

        await ctx.send(embed=embed)

    except Exception as e:
        logger.error(f"Question handling error: {str(e)}")
        await ctx.send("❌ Error processing your question. Please try again.")

@bot.command(name='guide')
async def show_help(ctx):
    """Show available commands and features"""
    embed = discord.Embed(
        title="🤖 Maverick Bot Commands",
        description="Complete guide to available commands and features",
        color=0x4DA3FF
    )

    # Analysis Commands
    embed.add_field(
        name="📊 Analysis Commands",
        value=(
            "`!analyze <pair>` - Start new analysis\n"
            "`!status` - Check analysis progress\n"
            "`!reset` - Reset current analysis"
        ),
        inline=False
    )

    # Setup Commands
    embed.add_field(
        name="⚙️ Setup Commands",
        value=(
            "`!setup validate` - Validate current setup\n"
            "`!setup grid` - Calculate grid levels\n"
            "`!setup alerts` - Set price alerts"
        ),
        inline=False
    )

    # Learning Commands
    embed.add_field(
        name="📚 Learning & Support",
        value=(
            "`!ask <question>` - Ask about strategy/analysis\n"
            "Example questions:\n"
            "• `!ask How do I identify a contraction phase?`\n"
            "• `!ask What volume patterns indicate institutional buying?`\n"
            "• `!ask Explain the grid system in trend phase`"
        ),
        inline=False
    )

    # Tips
    embed.add_field(
        name="💡 Tips",
        value=(
            "• Upload charts in order: 4H → 1H → 15M\n"
            "• Include RSI and volume indicators\n"
            "• Mark key support/resistance levels\n"
            "• Use `!ask` for strategy questions"
        ),
        inline=False
    )

    await ctx.send(embed=embed)

# Optional: Add these helper commands for quick reference
@bot.command(name='phases')
async def show_phases(ctx):
    """Show information about market phases"""
    embed = discord.Embed(
        title="📈 Market Phases Guide",
        description="Understanding the three market phases in Maverick Strategy",
        color=0x4DA3FF
    )

    # Contraction Phase
    embed.add_field(
        name="📊 Contraction Phase (Setup)",
        value=(
            "• Low institutional volume\n"
            "• Range getting smaller\n"
            "• RSI between 40-60\n"
            "• Grid level formation"
        ),
        inline=False
    )

    # Expansion Phase
    embed.add_field(
        name="📈 Expansion Phase (Play)",
        value=(
            "• Volume surge (2x average)\n"
            "• Range expansion\n"
            "• Clear directional move\n"
            "• Institutional footprint"
        ),
        inline=False
    )

    # Trend Phase
    embed.add_field(
        name="🚀 Trend Phase (Pay-out)",
        value=(
            "• Sustained directional move\n"
            "• Higher highs/lower lows\n"
            "• FOMO volume\n"
            "• Clear grid progression"
        ),
        inline=False
    )

    await ctx.send(embed=embed)

@bot.command(name='setups')
async def show_setups(ctx):
    """Show information about setup types"""
    embed = discord.Embed(
        title="🎯 Trading Setups Guide",
        description="Understanding the four main setup types",
        color=0x4DA3FF
    )

    # Bounce Setup
    embed.add_field(
        name="↩️ Bounce Setup",
        value=(
            "• Quick reversal at level\n"
            "• Volume confirmation\n"
            "• RSI extremes\n"
            "• Min R:R 1.5"
        ),
        inline=True
    )

    # Pullback Setup
    embed.add_field(
        name="⬇️ Pullback Setup",
        value=(
            "• Trend continuation\n"
            "• Decreasing volume\n"
            "• RSI reset\n"
            "• Min R:R 2.0"
        ),
        inline=True
    )

    # Add spacing
    embed.add_field(name="\u200b", value="\u200b", inline=True)

    # Reversal Setup
    embed.add_field(
        name="🔄 Reversal Setup",
        value=(
            "• Strong trend change\n"
            "• RSI divergence\n"
            "• Volume confirmation\n"
            "• Min R:R 2.5"
        ),
        inline=True
    )

    # Breakout Setup
    embed.add_field(
        name="💥 Breakout Setup",
        value=(
            "• Clean level break\n"
            "• Volume surge\n"
            "• Momentum confirm\n"
            "• Min R:R 1.8"
        ),
        inline=True
    )

    await ctx.send(embed=embed)

@bot.event
async def on_message(message):
    """Handle image uploads and commands"""
    if message.author.bot:
        return
        
    # Process commands first
    await bot.process_commands(message)
    
    # Check for image uploads in active analysis
    if message.attachments and message.author.id in maverick_bot.active_analyses:
        await handle_image_upload(message)

async def handle_image_upload(message):
    """Process uploaded chart images"""
    try:
        state = maverick_bot.active_analyses[message.author.id]
        progress_msg = await message.channel.send("🔄 Processing images...")
        
        for attachment in message.attachments:
            # Check file type
            if not any(attachment.filename.lower().endswith(f".{fmt.lower()}") 
                      for fmt in Config.SUPPORTED_FORMATS):
                continue
                
            # Download image
            image_data = await attachment.read()
            
            try:
                # Process image
                b64_data, mime_type = await maverick_bot.process_image(
                    image_data,
                    attachment.filename
                )
                
                # Detect timeframe
                timeframe = detect_timeframe(attachment.filename)
                if timeframe in state.timeframes:
                    state.timeframes[timeframe] = True
                    state.images.append((b64_data, mime_type))
                    
                logger.info(f"Processed image for {state.pair} - {timeframe}")
                
            except Exception as e:
                await message.channel.send(f"⚠️ Error processing {attachment.filename}: {str(e)}")
                continue
        
        # Check if all timeframes are uploaded
        if all(state.timeframes.values()):
            await progress_msg.edit(content="✅ All charts received! Starting analysis...")
            
            try:
                # Run analysis
                analysis = await maverick_bot.analyze_chart(state)
                await send_analysis_results(message.channel, analysis, state)
                
            except Exception as e:
                logger.error(f"Analysis error: {str(e)}")
                await message.channel.send("❌ Analysis failed. Please try again.")
                
        else:
            # Update progress message
            missing = [tf for tf, uploaded in state.timeframes.items() if not uploaded]
            await progress_msg.edit(
                content=f"📊 Waiting for timeframes: {', '.join(missing)}"
            )
            
    except Exception as e:
        logger.error(f"Image handling error: {str(e)}")
        await message.channel.send("❌ Error processing images.")

async def send_analysis_results(channel, analysis: Dict, state: AnalysisState):
    """Send formatted analysis results"""
    try:
        # Main Analysis Embed
        main_embed = discord.Embed(
            title=f"📊 Maverick Analysis - {state.pair}",
            color=0x4DA3FF
        )
        
        # Market Phase
        main_embed.add_field(
            name="Market Phase",
            value=f"```{analysis.get('phase', 'Unknown')}```",
            inline=False
        )
        
        # Price Levels
        levels = analysis.get('levels', {})
        support_levels = '\n'.join(f"• {level}" for level in levels.get('support', []))
        resistance_levels = '\n'.join(f"• {level}" for level in levels.get('resistance', []))
        
        if support_levels or resistance_levels:
            levels_text = ""
            if support_levels:
                levels_text += f"Support:\n{support_levels}\n\n"
            if resistance_levels:
                levels_text += f"Resistance:\n{resistance_levels}"
                
            main_embed.add_field(
                name="Key Levels",
                value=f"```{levels_text}```",
                inline=False
            )
        
        # Setup Classification
        setup_info = analysis.get('setup', {})
        setup_text = (
            f"Type: {setup_info.get('type', 'Unknown')}\n"
            f"Quality: {setup_info.get('quality', 'Standard')}\n"
            f"R:R Ratio: {analysis.get('risk_reward', 'N/A')}"
        )
        
        main_embed.add_field(
            name="Setup Classification",
            value=f"```{setup_text}```",
            inline=False
        )
        
        await channel.send(embed=main_embed)
        
        # Next Steps Embed
        next_steps = discord.Embed(
            title="📋 Next Steps",
            description=(
                "1️⃣ `!setup validate` - Confirm setup requirements\n"
                "2️⃣ `!setup grid` - Calculate grid levels\n"
                "3️⃣ `!setup alerts` - Set price alerts"
            ),
            color=0x4DA3FF
        )
        
        await channel.send(embed=next_steps)
        
    except Exception as e:
        logger.error(f"Error sending results: {str(e)}")
        await channel.send("❌ Error displaying analysis results.")

def detect_timeframe(filename: str) -> Optional[str]:
    """Detect timeframe from filename"""
    filename = filename.upper()
    for tf in Config.TIMEFRAMES:
        if tf in filename:
            return tf
    return None

@bot.event
async def on_reaction_add(reaction, user):
    """Handle alert setup reactions"""
    if user.bot:
        return

    if user.id not in maverick_bot.active_analyses:
        return

    if str(reaction.emoji) in ['1️⃣', '2️⃣', '3️⃣', '4️⃣']:
        state = maverick_bot.active_analyses[user.id]
        alert_type = {
            '1️⃣': 'entry',
            '2️⃣': 'stop',
            '3️⃣': 'target',
            '4️⃣': 'custom'
        }[str(reaction.emoji)]

        try:
            # Add alert
            price = state.current_price  # You might want to adjust this based on alert type
            await maverick_bot.alert_manager.add_alert(
                price=price,
                direction='above' if alert_type in ['entry', 'target'] else 'below',
                message=f"{alert_type.title()} level reached for {state.pair}",
                channel_id=reaction.message.channel.id,
                user_id=user.id
            )

            await reaction.message.channel.send(
                f"✅ {alert_type.title()} alert set at {price}"
            )

        except Exception as e:
            await reaction.message.channel.send(
                f"❌ Error setting {alert_type} alert: {str(e)}"
            )

class MaverickAnalyzer:
    """Advanced Maverick pattern recognition and analysis"""
    
    def __init__(self):
        self.phase_characteristics = {
            MarketPhase.CONTRACTION: {
                "volume": "below_average",
                "range": "< 1.5%",
                "rsi": "40-60",
                "min_touches": 3
            },
            MarketPhase.EXPANSION: {
                "volume": "increasing",
                "momentum": "strong",
                "rsi": "directional",
                "break_type": "clean"
            },
            MarketPhase.TREND: {
                "volume": "sustained",
                "structure": "higher_highs",
                "rsi": "momentum",
                "continuation": "clear"
            }
        }
        
        self.setup_requirements = {
            SetupType.BOUNCE: {
                "required_confirmations": [
                    "RSI oversold on 4H",
                    "Volume expansion",
                    "Clean rejection",
                    "Multiple timeframe alignment"
                ],
                "min_rr": 1.5
            },
            SetupType.PULLBACK: {
                "required_confirmations": [
                    "Clean market structure",
                    "Decreasing volume on pullback",
                    "Previous zone validation",
                    "RSI reset"
                ],
                "min_rr": 2.0
            },
            SetupType.REVERSAL: {
                "required_confirmations": [
                    "RSI divergence",
                    "Structure break",
                    "Volume confirmation",
                    "Multiple timeframe confluence"
                ],
                "min_rr": 2.5
            },
            SetupType.BREAKOUT: {
                "required_confirmations": [
                    "Volume surge",
                    "Clean break of structure",
                    "RSI momentum",
                    "No major resistance ahead"
                ],
                "min_rr": 1.8
            }
        }
    
    def analyze_charts(self, data: pd.DataFrame, timeframes: List[str]) -> Dict:
        """Analyze chart data for patterns and setups"""
        try:
            analysis = {
                'phase': self._detect_phase(data),
                'levels': self._detect_levels(data),
                'setup': self._detect_setup(data),
                'volume_analysis': self._analyze_volume(data),
                'momentum': self._analyze_momentum(data),
                'structure': self._analyze_structure(data)
            }
            return analysis
        except Exception as e:
            logger.error(f"Chart analysis error: {str(e)}")
            return {}

    def _detect_phase(self, data: pd.DataFrame) -> str:
        """Detect current market phase"""
        try:
            # Implement phase detection logic
            return "Unknown"
        except Exception as e:
            logger.error(f"Phase detection error: {str(e)}")
            return "Unknown"

    def _detect_levels(self, data: pd.DataFrame) -> Dict:
        """Detect support and resistance levels"""
        try:
            # Implement level detection logic
            return {'support': [], 'resistance': []}
        except Exception as e:
            logger.error(f"Level detection error: {str(e)}")
            return {'support': [], 'resistance': []}

    def _detect_setup(self, data: pd.DataFrame) -> Dict:
        """Detect trading setup"""
        try:
            # Implement setup detection logic
            return {'type': None, 'quality': 'standard'}
        except Exception as e:
            logger.error(f"Setup detection error: {str(e)}")
            return {'type': None, 'quality': 'standard'}

    def _analyze_volume(self, data: pd.DataFrame) -> Dict:
        """Analyze volume patterns"""
        try:
            # Implement volume analysis logic
            return {'trend': 'stable', 'state': 'average', 'quality': 'medium'}
        except Exception as e:
            logger.error(f"Volume analysis error: {str(e)}")
            return {'trend': 'stable', 'state': 'average', 'quality': 'medium'}

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators"""
        try:
            # Implement momentum analysis logic
            return {'rsi_state': 'neutral', 'strength': 'moderate', 'divergence': None}
        except Exception as e:
            logger.error(f"Momentum analysis error: {str(e)}")
            return {'rsi_state': 'neutral', 'strength': 'moderate', 'divergence': None}

    def _analyze_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure"""
        try:
            # Implement structure analysis logic
            return {'state': 'ranging', 'quality': 'medium', 'trend': None}
        except Exception as e:
            logger.error(f"Structure analysis error: {str(e)}")
            return {'state': 'ranging', 'quality': 'medium', 'trend': None}

class MaverickGridCalculator:
    """Calculates and manages Maverick grid levels"""
    
    def __init__(self):
        # ATR-based grid spacing defaults
        self.grid_settings = Config.GRID_SETTINGS
        
        # Minimum R:R requirements by setup type
        self.min_rr_ratios = {
            'bounce': Decimal('1.5'),
            'pullback': Decimal('2.0'),
            'reversal': Decimal('2.5'),
            'breakout': Decimal('1.8')
        }

    def calculate_grid(
        self,
        setup_type: str,
        current_price: Decimal,
        atr: Decimal,
        support_levels: List[Decimal],
        resistance_levels: List[Decimal]
    ) -> Dict[str, List[GridLevel]]:
        """Calculate grid levels based on setup type and market conditions"""
        try:
            # Determine volatility condition
            volatility = self._assess_volatility(atr, current_price)
            settings = self.grid_settings[volatility]
            
            # Initialize grid containers
            grid = {
                'entries': [],
                'stops': [],
                'targets': []
            }
            
            # Calculate base grid levels based on setup type
            if setup_type == 'bounce':
                grid = self._calculate_bounce_grid(
                    current_price, atr, settings,
                    support_levels, resistance_levels
                )
            elif setup_type == 'pullback':
                grid = self._calculate_pullback_grid(
                    current_price, atr, settings,
                    support_levels, resistance_levels
                )
            elif setup_type == 'breakout':
                grid = self._calculate_breakout_grid(
                    current_price, atr, settings,
                    support_levels, resistance_levels
                )
            elif setup_type == 'reversal':
                grid = self._calculate_reversal_grid(
                    current_price, atr, settings,
                    support_levels, resistance_levels
                )
                
            # Validate R:R ratios
            grid = self._validate_grid_rr(grid, setup_type)
            
            return grid
            
        except Exception as e:
            logger.error(f"Grid calculation error: {traceback.format_exc()}")
            raise

    def _assess_volatility(self, atr: Decimal, price: Decimal) -> str:
        """Assess volatility condition based on ATR"""
        atr_percentage = (atr / price) * 100
        
        if atr_percentage < Decimal('1.0'):
            return 'low_volatility'
        elif atr_percentage > Decimal('3.0'):
            return 'high_volatility'
        return 'normal_volatility'

    def _calculate_bounce_grid(
        self,
        price: Decimal,
        atr: Decimal,
        settings: Dict,
        supports: List[Decimal],
        resistances: List[Decimal]
    ) -> Dict[str, List[GridLevel]]:
        """Calculate grid levels for bounce setup"""
        grid = {'entries': [], 'stops': [], 'targets': []}
        
        # Find nearest support
        support = self._find_nearest_level(supports, price, 'below')
        if not support:
            return grid
            
        # Entry levels
        entry_base = support + (atr * settings['entry_spacing'])
        grid['entries'] = [
            GridLevel(
                price=entry_base,
                type='entry',
                status='active',
                strength=4,
                description='Primary bounce entry'
            ),
            GridLevel(
                price=entry_base - (atr * Decimal('0.25')),
                type='entry',
                status='active',
                strength=3,
                description='Secondary bounce entry'
            )
        ]
        
        # Stop loss
        grid['stops'] = [
            GridLevel(
                price=support - (atr * settings['sl_spacing']),
                type='sl',
                status='active',
                strength=5,
                description='Bounce stop loss'
            )
        ]
        
        # Take profit levels
        resistance = self._find_nearest_level(resistances, price, 'above')
        if resistance:
            grid['targets'] = [
                GridLevel(
                    price=resistance - (atr * Decimal('0.5')),
                    type='tp',
                    status='active',
                    strength=3,
                    description='Take profit 1'
                ),
                GridLevel(
                    price=resistance + (atr * settings['tp_spacing']),
                    type='tp',
                    status='active',
                    strength=4,
                    description='Take profit 2'
                )
            ]
        
        return grid
    
    def _calculate_pullback_grid(
        self,
        price: Decimal,
        atr: Decimal,
        settings: Dict,
        supports: List[Decimal],
        resistances: List[Decimal]
    ) -> Dict[str, List[GridLevel]]:
        """Calculate grid levels for pullback setup"""
        grid = {'entries': [], 'stops': [], 'targets': []}
        
        # Find nearest support and resistance
        support = self._find_nearest_level(supports, price, 'below')
        resistance = self._find_nearest_level(resistances, price, 'above')
        
        if not (support and resistance):
            return grid
            
        # Entry levels
        grid['entries'] = [
            GridLevel(
                price=support + (atr * Decimal('0.2')),
                type='entry',
                status='active',
                strength=4,
                description='Primary pullback entry'
            ),
            GridLevel(
                price=support,
                type='entry',
                status='active',
                strength=3,
                description='Deep pullback entry'
            )
        ]
        
        # Stop loss
        grid['stops'] = [
            GridLevel(
                price=support - (atr * settings['sl_spacing']),
                type='sl',
                status='active',
                strength=5,
                description='Pullback stop loss'
            )
        ]
        
        # Targets
        grid['targets'] = [
            GridLevel(
                price=resistance,
                type='tp',
                status='active',
                strength=3,
                description='Initial target'
            ),
            GridLevel(
                price=resistance + (atr * settings['tp_spacing']),
                type='tp',
                status='active',
                strength=4,
                description='Extended target'
            )
        ]
        
        return grid

    def _calculate_breakout_grid(
        self,
        price: Decimal,
        atr: Decimal,
        settings: Dict,
        supports: List[Decimal],
        resistances: List[Decimal]
    ) -> Dict[str, List[GridLevel]]:
        """Calculate grid levels for breakout setup"""
        grid = {'entries': [], 'stops': [], 'targets': []}
        
        # Find nearest resistance (breakout level)
        breakout_level = self._find_nearest_level(resistances, price, 'below')
        if not breakout_level:
            return grid
        
        # Entries above breakout
        grid['entries'] = [
            GridLevel(
                price=breakout_level + (atr * Decimal('0.2')),
                type='entry',
                status='active',
                strength=4,
                description='Breakout entry'
            ),
            GridLevel(
                price=breakout_level + (atr * Decimal('0.5')),
                type='entry',
                status='active',
                strength=3,
                description='Momentum entry'
            )
        ]
        
        # Stop below breakout
        grid['stops'] = [
            GridLevel(
                price=breakout_level - (atr * Decimal('0.5')),
                type='sl',
                status='active',
                strength=5,
                description='Breakout stop loss'
            )
        ]
        
        # Target at next resistance
        next_resistance = self._find_nearest_level(resistances, price, 'above')
        if next_resistance:
            grid['targets'] = [
                GridLevel(
                    price=next_resistance,
                    type='tp',
                    status='active',
                    strength=3,
                    description='First target'
                ),
                GridLevel(
                    price=next_resistance + (atr * settings['tp_spacing']),
                    type='tp',
                    status='active',
                    strength=4,
                    description='Extended target'
                )
            ]
        
        return grid

    def _calculate_reversal_grid(
        self,
        price: Decimal,
        atr: Decimal,
        settings: Dict,
        supports: List[Decimal],
        resistances: List[Decimal]
    ) -> Dict[str, List[GridLevel]]:
        """Calculate grid levels for reversal setup"""
        grid = {'entries': [], 'stops': [], 'targets': []}
        
        # Find nearest resistance (reversal point)
        reversal_level = self._find_nearest_level(resistances, price, 'below')
        if not reversal_level:
            return grid
        
        # Entries below reversal
        grid['entries'] = [
            GridLevel(
                price=reversal_level - (atr * Decimal('0.3')),
                type='entry',
                status='active',
                strength=4,
                description='Primary reversal entry'
            ),
            GridLevel(
                price=reversal_level - (atr * Decimal('0.6')),
                type='entry',
                status='active',
                strength=3,
                description='Secondary reversal entry'
            )
        ]
        
        # Stop above reversal
        grid['stops'] = [
            GridLevel(
                price=reversal_level + (atr * settings['sl_spacing']),
                type='sl',
                status='active',
                strength=5,
                description='Reversal stop loss'
            )
        ]
        
        # Targets at support levels
        support = self._find_nearest_level(supports, price, 'below')
        if support:
            grid['targets'] = [
                GridLevel(
                    price=support + (atr * Decimal('0.5')),
                    type='tp',
                    status='active',
                    strength=3,
                    description='First target'
                ),
                GridLevel(
                    price=support,
                    type='tp',
                    status='active',
                    strength=4,
                    description='Final target'
                )
            ]
        
        return grid

    def _find_nearest_level(
        self, 
        levels: List[Decimal],
        price: Decimal,
        direction: str
    ) -> Optional[Decimal]:
        """Find nearest price level in given direction"""
        if not levels:
            return None
            
        if direction == 'above':
            above_levels = [l for l in levels if l > price]
            return min(above_levels) if above_levels else None
        else:
            below_levels = [l for l in levels if l < price]
            return max(below_levels) if below_levels else None

    def _validate_grid_rr(
        self,
        grid: Dict[str, List[GridLevel]],
        setup_type: str
    ) -> Dict[str, List[GridLevel]]:
        """Validate and adjust grid levels for minimum R:R requirements"""
        if not (grid['entries'] and grid['stops'] and grid['targets']):
            return grid
            
        min_rr = self.min_rr_ratios.get(setup_type, Decimal('1.5'))
        
        # Calculate current R:R
        entry = grid['entries'][0].price
        stop = grid['stops'][0].price
        target = grid['targets'][-1].price
        
        risk = abs(entry - stop)
        reward = abs(target - entry)
        
        current_rr = reward / risk
        
        # Adjust if needed
        if current_rr < min_rr:
            # Extend target
            required_reward = risk * min_rr
            new_target = entry + (required_reward if entry > stop else -required_reward)
            
            grid['targets'][-1] = GridLevel(
                price=new_target,
                type='tp',
                status='active',
                strength=4,
                description=f'Adjusted take profit (R:R {min_rr})'
            )
            
        return grid

class AlertManager:
    """Manages price alerts and notifications"""
    
    def __init__(self, bot):
        self.bot = bot
        self.alerts: Set[PriceAlert] = set()
        self.check_interval = Config.ALERT_CHECK_INTERVAL
        self.cleanup_interval = Config.ALERT_CLEANUP_INTERVAL
        self.max_alerts_per_user = Config.MAX_ALERTS_PER_USER
        self.max_alert_duration = timedelta(seconds=Config.MAX_ALERT_DURATION)
        
        # Start alert checker
        self.checker_task = asyncio.create_task(self._alert_checker())
        self.cleanup_task = asyncio.create_task(self._cleanup_alerts())
        
    async def add_alert(
        self,
        price: Decimal,
        direction: str,
        message: str,
        channel_id: int,
        user_id: int,
        duration: Optional[timedelta] = None
    ) -> bool:
        """Add new price alert"""
        try:
            # Check user alert limit
            user_alerts = sum(1 for a in self.alerts if a.user_id == user_id)
            if user_alerts >= self.max_alerts_per_user:
                return False
            
            # Set expiration
            if not duration:
                duration = self.max_alert_duration
            expiry = datetime.now() + duration
            
            # Create and add alert
            alert = PriceAlert(
                price=price,
                direction=direction,
                message=message,
                channel_id=channel_id,
                user_id=user_id,
                created_at=datetime.now(),
                expires_at=expiry
            )
            
            self.alerts.add(alert)
            logger.info(f"Added price alert at {price} for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding alert: {str(e)}")
            return False

    async def remove_alert(self, user_id: int, price: Decimal) -> bool:
        """Remove specific price alert"""
        try:
            alert = next(
                (a for a in self.alerts 
                 if a.user_id == user_id and a.price == price),
                None
            )
            
            if alert:
                self.alerts.remove(alert)
                logger.info(f"Removed alert at {price} for user {user_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error removing alert: {str(e)}")
            return False

    async def get_user_alerts(self, user_id: int) -> List[PriceAlert]:
        """Get all active alerts for user"""
        return [a for a in self.alerts if a.user_id == user_id]

    async def _alert_checker(self):
        """Background task to check price alerts"""
        while True:
            try:
                await asyncio.sleep(self.check_interval)
                
                # Get current price (implement price fetching)
                current_price = await self._get_current_price()
                if not current_price:
                    continue
                
                # Check each alert
                triggered = set()
                for alert in self.alerts:
                    if alert.triggered:
                        continue
                        
                    if self._check_alert_condition(alert, current_price):
                        try:
                            # Send alert notification
                            channel = self.bot.get_channel(alert.channel_id)
                            if channel:
                                await channel.send(
                                    f"🚨 **Price Alert**\n"
                                    f"<@{alert.user_id}> {alert.message}\n"
                                    f"Price: {current_price}"
                                )
                            
                            alert.triggered = True
                            triggered.add(alert)
                            
                        except Exception as e:
                            logger.error(f"Error sending alert: {str(e)}")
                
                # Remove triggered alerts
                self.alerts -= triggered
                
            except Exception as e:
                logger.error(f"Alert checker error: {str(e)}")
                await asyncio.sleep(self.check_interval)

    async def _cleanup_alerts(self):
        """Clean up expired alerts"""
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                now = datetime.now()
                
                # Remove expired alerts
                expired = {
                    alert for alert in self.alerts
                    if alert.expires_at <= now
                }
                
                self.alerts -= expired
                
                if expired:
                    logger.info(f"Cleaned up {len(expired)} expired alerts")
                    
            except Exception as e:
                logger.error(f"Alert cleanup error: {str(e)}")

    def _check_alert_condition(self, alert: PriceAlert, current_price: Decimal) -> bool:
        """Check if alert condition is met"""
        if alert.direction == 'above':
            return current_price >= alert.price
        else:
            return current_price <= alert.price

    async def _get_current_price(self) -> Optional[Decimal]:
        """Get current price implementation"""
        # Implement price fetching logic here
        return None

class MaverickSetupValidator:
    """Validates trading setups against Maverick criteria"""
    
    def __init__(self):
        self.setup_scores = {
            "pattern_quality": 0.3,
            "momentum_alignment": 0.2,
            "volume_confirmation": 0.2,
            "structure_validation": 0.3
        }
        
        self.minimum_requirements = {
            SetupType.BOUNCE: {
                "pattern_quality": 0.7,
                "momentum_alignment": 0.8,
                "volume_confirmation": 0.6,
                "structure_validation": 0.7
            },
            SetupType.PULLBACK: {
                "pattern_quality": 0.8,
                "momentum_alignment": 0.7,
                "volume_confirmation": 0.7,
                "structure_validation": 0.8
            },
            SetupType.REVERSAL: {
                "pattern_quality": 0.8,
                "momentum_alignment": 0.8,
                "volume_confirmation": 0.8,
                "structure_validation": 0.8
            },
            SetupType.BREAKOUT: {
                "pattern_quality": 0.7,
                "momentum_alignment": 0.7,
                "volume_confirmation": 0.8,
                "structure_validation": 0.7
            }
        }

@dataclass
class BacktestTrade:
    """Represents a single backtest trade"""
    entry_time: datetime
    entry_price: Decimal
    exit_time: Optional[datetime] = None
    exit_price: Optional[Decimal] = None
    setup_type: str = ""
    position_size: Decimal = Decimal('0')
    stop_loss: Decimal = Decimal('0')
    take_profit: Decimal = Decimal('0')
    pnl: Optional[Decimal] = None
    status: str = "open"  # open, closed, stopped
    grid_levels: List[GridLevel] = None
    timeframe: str = "4H"

@dataclass
class BacktestResult:
    """Contains backtest results and statistics"""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    average_win: Decimal
    average_loss: Decimal
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[BacktestTrade]
    equity_curve: List[float]
    setup_performance: Dict[str, Dict]
    optimization_params: Dict = None

class MaverickBacktester:
    """Backtesting engine for Maverick strategy"""
    
    def __init__(self):
        self.analyzer = MaverickAnalyzer()
        self.grid_calculator = MaverickGridCalculator()
        
        # Default parameters
        self.default_params = {
            'atr_period': Config.ATR_PERIOD,
            'rsi_period': Config.RSI_PERIOD,
            'grid_extension': Decimal('0.5'),
            'min_rr_ratio': Config.MIN_RR_RATIO,
            'position_size': Decimal('0.01'),  # 1% per trade
            'max_trades': 3  # Maximum concurrent trades
        }
        
        # Optimization ranges
        self.optimization_ranges = {
            'atr_period': range(10, 21),
            'rsi_period': range(10, 21),
            'grid_extension': [Decimal('0.3'), Decimal('0.4'), Decimal('0.5'), 
                             Decimal('0.6'), Decimal('0.7')],
            'min_rr_ratio': [Decimal('1.3'), Decimal('1.5'), Decimal('1.8'), 
                            Decimal('2.0'), Decimal('2.2')]
        }

    async def backtest(
        self,
        data: pd.DataFrame,
        timeframes: List[str],
        params: Optional[Dict] = None,
        progress_callback: Optional[callable] = None
    ) -> BacktestResult:
        """Run backtest with given parameters"""
        try:
            # Use provided params or defaults
            test_params = {**self.default_params, **(params or {})}
            
            # Prepare data
            prepared_data = self._prepare_data(data, timeframes, test_params)
            
            # Initialize backtest variables
            trades: List[BacktestTrade] = []
            equity = [Decimal('100000')]  # Start with 100k
            open_trades = []
            
            # Run through each candle
            total_candles = len(prepared_data)
            
            for i, candle in enumerate(prepared_data.itertuples()):
                # Update progress
                if progress_callback and i % 100 == 0:
                    await progress_callback(i / total_candles * 100)
                
                # Check and update open trades
                await self._update_trades(open_trades, candle, trades, equity)
                
                # Look for new setups if we're not at max trades
                if len(open_trades) < test_params['max_trades']:
                    setup = await self._identify_setup(
                        prepared_data.iloc[:i+1],
                        timeframes,
                        test_params
                    )
                    
                    if setup:
                        # Calculate grid levels
                        grid = self.grid_calculator.calculate_grid(
                            setup_type=setup['type'],
                            current_price=Decimal(str(candle.close)),
                            atr=Decimal(str(candle.atr)),
                            support_levels=setup['supports'],
                            resistance_levels=setup['resistances']
                        )
                        
                        # Validate setup
                        if self._validate_setup(setup, grid, test_params):
                            # Open new trade
                            trade = BacktestTrade(
                                entry_time=candle.Index,
                                entry_price=Decimal(str(candle.close)),
                                setup_type=setup['type'],
                                position_size=test_params['position_size'] * equity[-1],
                                stop_loss=grid['stops'][0].price,
                                take_profit=grid['targets'][-1].price,
                                grid_levels=grid['entries'],
                                timeframe=setup['timeframe']
                            )
                            open_trades.append(trade)
            
            # Close any remaining trades
            for trade in open_trades:
                trade.exit_time = prepared_data.index[-1]
                trade.exit_price = Decimal(str(prepared_data.iloc[-1].close))
                trade.status = 'closed'
                trade.pnl = self._calculate_pnl(trade)
                trades.append(trade)
            
            # Calculate results
            results = self._calculate_results(trades, equity)
            return results
            
        except Exception as e:
            logger.error(f"Backtest error: {traceback.format_exc()}")
            raise

    async def optimize(
        self,
        data: pd.DataFrame,
        timeframes: List[str],
        target_metric: str = 'sharpe_ratio',
        population_size: int = 30,
        generations: int = 20,
        progress_callback: Optional[callable] = None
    ) -> Tuple[Dict, BacktestResult]:
        """Optimize strategy parameters using genetic algorithm"""
        try:
            # Initialize population with random parameters
            population = self._initialize_population(population_size)
            best_result = None
            best_params = None
            
            # Run generations
            for gen in range(generations):
                gen_results = []
                
                # Test each parameter set
                for i, params in enumerate(population):
                    # Update progress
                    if progress_callback:
                        progress = (gen * population_size + i) / (generations * population_size) * 100
                        await progress_callback(progress)
                    
                    # Run backtest
                    result = await self.backtest(data, timeframes, params)
                    gen_results.append((params, result))
                
                # Sort by target metric
                gen_results.sort(
                    key=lambda x: getattr(x[1], target_metric),
                    reverse=True
                )
                
                # Update best result
                if not best_result or getattr(gen_results[0][1], target_metric) > getattr(best_result, target_metric):
                    best_params = gen_results[0][0]
                    best_result = gen_results[0][1]
                
                # Create next generation
                population = self._create_next_generation(
                    gen_results,
                    population_size
                )
            
            return best_params, best_result
            
        except Exception as e:
            logger.error(f"Optimization error: {traceback.format_exc()}")
            raise

    async def _identify_setup(
        self,
        data: pd.DataFrame,
        timeframes: List[str],
        params: Dict
    ) -> Optional[Dict]:
        """Identify trading setup in historical data"""
        try:
            # Get latest candle data
            current_idx = len(data) - 1
            window = data.iloc[max(0, current_idx-20):current_idx+1]
            
            # Run pattern analysis
            analysis = self.analyzer.analyze_charts(window, timeframes)
            
            if analysis and analysis.get('setup', {}).get('type'):
                return {
                    'type': analysis['setup']['type'],
                    'supports': [Decimal(str(p)) for p in analysis.get('levels', {}).get('support', [])],
                    'resistances': [Decimal(str(p)) for p in analysis.get('levels', {}).get('resistance', [])],
                    'timeframe': analysis.get('timeframe', '4H')
                }
            return None
            
        except Exception as e:
            logger.error(f"Setup identification error: {str(e)}")
            return None

    def _validate_setup(
        self,
        setup: Dict,
        grid: Dict[str, List[GridLevel]],
        params: Dict
    ) -> bool:
        """Validate setup meets minimum requirements"""
        try:
            # Check R:R ratio
            if not (grid['entries'] and grid['stops'] and grid['targets']):
                return False
                
            entry = grid['entries'][0].price
            stop = grid['stops'][0].price
            target = grid['targets'][-1].price
            
            risk = abs(entry - stop)
            reward = abs(target - entry)
            
            if risk == 0:
                return False
                
            rr_ratio = reward / risk
            
            return rr_ratio >= params['min_rr_ratio']
            
        except Exception as e:
            logger.error(f"Setup validation error: {str(e)}")
            return False
        
    async def _update_trades(
        self,
        open_trades: List[BacktestTrade],
        candle: pd.Series,
        closed_trades: List[BacktestTrade],
        equity: List[float]
    ):
        """Update open trades with new candle data"""
        current_price = Decimal(str(candle.close))
        
        for trade in open_trades[:]:  # Copy list for safe removal
            # Check stop loss
            if candle.low <= trade.stop_loss:
                trade.exit_price = trade.stop_loss
                trade.exit_time = candle.Index
                trade.status = 'stopped'
                trade.pnl = self._calculate_pnl(trade)
                closed_trades.append(trade)
                open_trades.remove(trade)
                equity.append(float(equity[-1] + float(trade.pnl)))
                continue
            
            # Check take profit
            if candle.high >= trade.take_profit:
                trade.exit_price = trade.take_profit
                trade.exit_time = candle.Index
                trade.status = 'closed'
                trade.pnl = self._calculate_pnl(trade)
                closed_trades.append(trade)
                open_trades.remove(trade)
                equity.append(float(equity[-1] + float(trade.pnl)))
    
    def _prepare_data(
        self,
        data: pd.DataFrame,
        timeframes: List[str],
        params: Dict
    ) -> pd.DataFrame:
        """Prepare data for backtesting"""
        df = data.copy()
        
        # Add technical indicators
        df['atr'] = self._calculate_atr(df, params['atr_period'])
        df['rsi'] = self._calculate_rsi(df, params['rsi_period'])
        
        # Add support/resistance levels
        df = self._add_levels(df)
        
        return df

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range"""
        high = data['high']
        low = data['low']
        close = data['close'].shift(1)
        
        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def _calculate_rsi(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _add_levels(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add support and resistance levels"""
        data['support'] = data['low'].rolling(20).min()
        data['resistance'] = data['high'].rolling(20).max()
        return data

    def _initialize_population(self, size: int) -> List[Dict]:
        """Initialize random population for optimization"""
        import random
        
        population = []
        for _ in range(size):
            params = {}
            for param, value_range in self.optimization_ranges.items():
                if isinstance(value_range, range):
                    params[param] = random.choice(value_range)
                else:
                    params[param] = random.choice(value_range)
            population.append(params)
        return population

    def _create_next_generation(
        self,
        results: List[Tuple[Dict, BacktestResult]],
        population_size: int
    ) -> List[Dict]:
        """Create next generation using genetic algorithm"""
        import random
        
        # Keep top 20% performers
        next_gen = [r[0] for r in results[:int(population_size * 0.2)]]
        
        # Create rest through crossover and mutation
        while len(next_gen) < population_size:
            # Select parents
            parent1 = random.choice(results[:int(population_size * 0.4)])[0]
            parent2 = random.choice(results[:int(population_size * 0.4)])[0]
            
            # Crossover
            child = self._crossover(parent1, parent2)
            
            # Mutation
            child = self._mutate(child)
            
            next_gen.append(child)
            
        return next_gen

    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Perform parameter crossover"""
                
        child = {}
        for param in self.optimization_ranges.keys():
            if random.random() < 0.5:
                child[param] = parent1[param]
            else:
                child[param] = parent2[param]
        return child

    def _mutate(self, params: Dict) -> Dict:
        """Mutate parameters randomly"""
        import random
        
        mutated = params.copy()
        for param, value_range in self.optimization_ranges.items():
            if random.random() < 0.1:  # 10% mutation rate
                if isinstance(value_range, range):
                    mutated[param] = random.choice(value_range)
                else:
                    mutated[param] = random.choice(value_range)
        return mutated

    def _calculate_results(
        self,
        trades: List[BacktestTrade],
        equity: List[float]
    ) -> BacktestResult:
        """Calculate comprehensive backtest results"""
        try:
            winning_trades = [t for t in trades if t.pnl > 0]
            losing_trades = [t for t in trades if t.pnl <= 0]
            
            setup_performance = self._calculate_setup_performance(trades)
            
            return BacktestResult(
                total_trades=len(trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades),
                win_rate=len(winning_trades) / len(trades) if trades else 0,
                average_win=sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else Decimal('0'),
                average_loss=sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else Decimal('0'),
                profit_factor=self._calculate_profit_factor(trades),
                max_drawdown=self._calculate_max_drawdown(equity),
                sharpe_ratio=self._calculate_sharpe_ratio(equity),
                trades=trades,
                equity_curve=equity,
                setup_performance=setup_performance
            )
        except Exception as e:
            logger.error(f"Error calculating results: {str(e)}")
            raise

    def _calculate_setup_performance(self, trades: List[BacktestTrade]) -> Dict:
        """Calculate performance metrics by setup type"""
        setup_stats = {}
        
        for setup_type in ['bounce', 'pullback', 'reversal', 'breakout']:
            setup_trades = [t for t in trades if t.setup_type == setup_type]
            if not setup_trades:
                continue
                
            winning = [t for t in setup_trades if t.pnl > 0]
            
            setup_stats[setup_type] = {
                'total_trades': len(setup_trades),
                'win_rate': len(winning) / len(setup_trades),
                'average_rr': self._calculate_average_rr(setup_trades),
                'profit_factor': self._calculate_profit_factor(setup_trades),
                'average_bars': self._calculate_average_bars(setup_trades)
            }
            
        return setup_stats

    def _calculate_profit_factor(self, trades: List[BacktestTrade]) -> float:
        """Calculate profit factor"""
        gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
        
        return float(gross_profit / gross_loss) if gross_loss else 0

    def _calculate_max_drawdown(self, equity: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = equity[0]
        max_dd = 0
        
        for value in equity:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd

    def _calculate_sharpe_ratio(self, equity: List[float]) -> float:
        """Calculate Sharpe ratio"""
        returns = pd.Series(equity).pct_change().dropna()
        if len(returns) < 2:
            return 0
            
        return float((returns.mean() * 252) / (returns.std() * math.sqrt(252)))

    def _calculate_average_rr(self, trades: List[BacktestTrade]) -> float:
        """Calculate average risk/reward ratio"""
        if not trades:
            return 0
            
        rr_ratios = []
        for trade in trades:
            risk = abs(trade.entry_price - trade.stop_loss)
            if risk == 0:
                continue
            reward = abs(trade.exit_price - trade.entry_price)
            rr_ratios.append(float(reward / risk))
            
        return sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0

    def _calculate_average_bars(self, trades: List[BacktestTrade]) -> float:
        """Calculate average trade duration in bars"""
        if not trades:
            return 0
            
        durations = []
        for trade in trades:
            if trade.exit_time and trade.entry_time:
                duration = (trade.exit_time - trade.entry_time).total_seconds()
                durations.append(duration)
                
        return sum(durations) / len(durations) if durations else 0

# Main execution
def main():
    """Main execution function"""
    try:
        # Validate configuration
        Config.validate()
        
        # Initialize Discord bot
        logger.info("Starting Maverick Trading Bot...")
        
        # Run the bot
        bot.run(Config.DISCORD_TOKEN)
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
