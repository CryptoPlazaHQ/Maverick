import os
import sys
import json
import logging
import asyncio
import aiohttp
import discord
from discord.ext import commands
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
from anthropic import Anthropic
from anthropic import AsyncAnthropic, RateLimitError
from anthropic import (
    AsyncAnthropic,
    BadRequestError,
    AuthenticationError,
    PermissionDeniedError,
    NotFoundError,
    UnprocessableEntityError,
    RateLimitError,
    InternalServerError,
    APIConnectionError
)
from dotenv import load_dotenv
import traceback
import base64
import re
import io
from PIL import Image
import signal


# Ensure proper event loop policy for Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Enhanced Logging Configuration
def setup_logging() -> logging.Logger:
    """
    Configure enhanced logging with both file and console handlers
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    # Create formatter with detailed information
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Configure logger
    logger = logging.getLogger('maverick_bot')
    logger.setLevel(logging.DEBUG)
    
    # Console handler (INFO and above)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # File handler (DEBUG and above) with daily rotation
    file_handler = logging.FileHandler(
        f'logs/maverick_bot_{datetime.now().strftime("%Y%m%d")}.log',
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

# Initialize logger
logger = setup_logging()

# Load environment variables
load_dotenv()

class Config:
    """Configuration class with enhanced validation and documentation"""
    @classmethod
    def validate(cls) -> None:
        """Validate configuration settings"""
        errors = []
        
        # Check API keys and credits
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY environment variable is missing")
        else:
            # Test Anthropic API access
            try:
                client = Anthropic(api_key=cls.ANTHROPIC_API_KEY)
                # Add a simple test request
                client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}]
                )
            except Exception as e:
                errors.append(f"Anthropic API error: {str(e)}")
        
        if errors:
            raise ValueError("\n".join(errors))
        

    # API Keys and Tokens
    DISCORD_TOKEN = os.getenv('DISCORD_BOT_TOKEN')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')


    # Image Processing Settings
    IMAGE_SETTINGS = {
        'max_size': 1024 * 1024 * 2,  # 2MB (Claude's recommended limit)
        'max_dimension': 1024,         # Maximum dimension in pixels
        'allowed_formats': {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        },
        'format_settings': {
            'jpeg': {
                'quality': 85,
                'optimize': True
            },
            'png': {
                'optimize': True,
                'compress_level': 6
            },
            'webp': {
                'quality': 85,
                'method': 4
            },
            'gif': {
                'optimize': True
            }
        }
    }
    
    # Analysis Settings
    ANALYSIS_SETTINGS = {
        'timeframes': ['4H', '1H', '15M'],
        'max_cached_analyses': 10,
        'analysis_timeout': 300,  # 5 minutes
        'max_retries': 3,
        'chunk_size': 256 * 1024  # 256KB chunks for processing
    }
    
    CLEANUP_SETTINGS = {
        'session_timeout': 3600,  # 1 hour
        'max_active_sessions': 50,
        'memory_threshold': 0.8,  # 80% memory usage trigger
        'cleanup_interval': 300  # Run cleanup every 5 minutes
    }


    # Trading Settings
    TRADING_SETTINGS = {
        'atr_threshold': 3.5,
        'low_volatility': {
            'take_profit': 4.0,
            'stop_loss': 6.0,
            'grid_profit': [2.0, 3.0]
        },
        'high_volatility': {
            'take_profit': 6.0,
            'stop_loss': 9.0,
            'grid_profit': 3.0
        }
    }
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration settings and raise detailed errors if invalid
        
        Raises:
            ValueError: With specific validation error messages
        """
        errors = []
        
        # Check API keys
        if not cls.DISCORD_TOKEN:
            errors.append("DISCORD_BOT_TOKEN environment variable is missing")
        if not cls.ANTHROPIC_API_KEY:
            errors.append("ANTHROPIC_API_KEY environment variable is missing")
            
        # Validate image settings
        if cls.IMAGE_SETTINGS['max_size'] > 2 * 1024 * 1024:  # 2MB limit
            errors.append("Image max_size exceeds Claude's 2MB limit")
        if cls.IMAGE_SETTINGS['max_dimension'] > 1024:
            errors.append("Image max_dimension exceeds recommended 1024px limit")
            
        if errors:
            raise ValueError("\n".join(errors))

# Custom Exception Classes
class MaverickBotError(Exception):
    """Base exception class for Maverick Bot"""
    pass

class ValidationError(MaverickBotError):
    """Raised when validation fails"""
    pass

class AnalysisError(MaverickBotError):
    """Raised when analysis operations fail"""
    pass

class ImageProcessingError(MaverickBotError):
    """Raised when image processing fails"""
    pass

# Data Classes
@dataclass
class ChartMetadata:
    """Stores metadata about chart images"""
    timeframe: str
    timestamp: datetime
    pair: str
    indicators: List[str]
    dimensions: Tuple[int, int]
    file_size: int
    format: str
    
    def __str__(self) -> str:
        """String representation for logging purposes"""
        return (
            f"Chart[{self.timeframe}] - {self.pair} "
            f"({self.dimensions[0]}x{self.dimensions[1]}px, "
            f"{self.file_size/1024:.1f}KB, {self.format})"
        )

@dataclass
class Confirmation:
    """Stores setup confirmation details"""
    description: str
    status: bool
    value: Optional[float] = None
    timestamp: Optional[datetime] = None

@dataclass
class GridLevel:
    """Stores grid trading level information"""
    price: Decimal
    type: str  # 'support' or 'resistance'
    strength: int  # 1-5
    confirmation_count: int
    exit_percentage: Optional[Decimal] = None

# Enums
class SetupType(Enum):
    """Trading setup classification types"""
    BOUNCE_BACK = "Bounce Back"
    PULLBACK = "Pullback"
    REVERSAL = "Reversal"
    BREAKOUT = "Breakout"

class TradeStatus(Enum):
    """Trading position status types"""
    PENDING = "Pending Entry"
    PARTIAL = "Partially Filled"
    ACTIVE = "Active"
    CLOSED = "Closed"


class ChartImageValidator:
    def __init__(self, config: Config):
        # Standardize attribute names
        self.max_file_size = 5 * 1024 * 1024  # 5MB per API docs
        self.max_dimension = 1568  # Maximum dimension in pixels
        self.min_dimension = 200   # Minimum dimension
        
        # Supported formats from Claude documentation
        self.supported_formats = {
            'JPEG': 'image/jpeg',
            'PNG': 'image/png',
            'GIF': 'image/gif',
            'WEBP': 'image/webp'
        }
        
        # Optimal dimensions for token efficiency
        self.optimal_dimensions = {
            '1:1': (1092, 1092),
            '3:4': (951, 1268),
            '2:3': (896, 1344),
            '9:16': (819, 1456),
            '1:2': (784, 1568)
        }
        
        logger.info(
            f"ChartImageValidator initialized with Claude's specifications:\n"
            f"- Max file size: {self.max_file_size/1024/1024:.1f}MB\n"
            f"- Max dimension: {self.max_dimension}px\n"
            f"- Min dimension: {self.min_dimension}px"
        )

    async def validate_and_process_image(
        self,
        image_url: str,
        timeframe: str
    ) -> Tuple[str, ChartMetadata, str]:
        """
        Process image from URL with validation and optimization
        
        Args:
            image_url: URL of the image to process
            timeframe: Trading timeframe for the chart
            
        Returns:
            Tuple containing (base64_data, metadata, mime_type)
            
        Raises:
            ValidationError: If image validation fails
            ImageProcessingError: If processing fails
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as response:
                    if response.status != 200:
                        raise ValidationError(f"Failed to fetch image: {response.status}")
                    
                    image_data = await response.read()
                    
                    # Check file size
                    if len(image_data) > self.max_file_size:
                        raise ValidationError(
                            f"Image exceeds {self.max_file_size/1024/1024:.1f}MB limit"
                        )
                    
                    # Process and validate image
                    image = Image.open(io.BytesIO(image_data))
                    
                    # Validate format
                    if image.format not in self.supported_formats:
                        raise ValidationError(
                            f"Unsupported format: {image.format}. "
                            f"Supported: {', '.join(self.supported_formats.keys())}"
                        )
                    
                    # Process image
                    processed_image = self._process_image(
                        image,
                        image.format.lower()
                    )
                    
                    # Convert to bytes
                    output = io.BytesIO()
                    processed_image.save(
                        output,
                        format=image.format,
                        quality=85,
                        optimize=True
                    )
                    output.seek(0)
                    
                    # Get final image data
                    final_data = output.getvalue()
                    
                    # Create metadata
                    metadata = ChartMetadata(
                        timeframe=timeframe,
                        timestamp=datetime.now(),
                        pair="unknown",  # Set later
                        indicators=[],    # Detected during analysis
                        dimensions=processed_image.size,
                        file_size=len(final_data),
                        format=image.format.lower()
                    )
                    
                    # Convert to base64
                    img_base64 = base64.b64encode(final_data).decode('utf-8')
                    
                    return img_base64, metadata, self.supported_formats[image.format]

        except aiohttp.ClientError as e:
            raise ValidationError(f"Failed to download image: {str(e)}")
        except Image.DecompressionBombError:
            raise ValidationError("Image is too large to process safely")
        except Exception as e:
            logger.error(f"Image processing error: {traceback.format_exc()}")
            raise ImageProcessingError(f"Failed to process image: {str(e)}")

    def _process_image(self, image: Image.Image, format: str) -> Image.Image:
        """
        Process image based on format while preserving quality
        
        Args:
            image: PIL Image object
            format: Image format (jpeg, png, gif, webp)
            
        Returns:
            Processed PIL Image object
            
        Raises:
            ImageProcessingError: If processing fails
        """
        try:
            width, height = image.size
            
            # Check minimum dimension
            if min(width, height) < self.min_dimension:
                raise ValidationError(
                    f"Image too small ({width}x{height}). "
                    f"Minimum dimension: {self.min_dimension}px"
                )
            
            # Resize if needed
            if max(width, height) > self.max_dimension:
                scale = self.max_dimension / max(width, height)
                new_size = (int(width * scale), int(height * scale))
                image = image.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(
                    f"Resized {format.upper()}: {width}x{height} -> "
                    f"{new_size[0]}x{new_size[1]}"
                )
            
            # Format-specific processing
            if format in ['jpeg', 'jpg']:
                if image.mode != 'RGB':
                    image = image.convert('RGB')
            elif format == 'png':
                if image.mode == 'RGBA':
                    background = Image.new('RGBA', image.size, (255, 255, 255))
                    image = Image.alpha_composite(background, image)
                    image = image.convert('RGB')
            elif format == 'webp':
                pass  # WebP handles both RGB and RGBA
            elif format == 'gif':
                if 'duration' in image.info:
                    logger.warning("Animated GIF detected - using first frame")
                if image.mode not in ['RGB', 'RGBA']:
                    image = image.convert('RGBA')
            
            return image
            
        except Exception as e:
            raise ImageProcessingError(f"Image processing failed: {str(e)}")


class VisionAnalyzer:
    def __init__(self, anthropic_client: AsyncAnthropic):
        # Keep your existing client initialization - it's correct per SDK
        self.anthropic = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            timeout=20.0,  # 20 seconds
            max_retries=2  # default is 2
        )
        
        # Keep your existing settings
        self.chunk_size = Config.ANALYSIS_SETTINGS['chunk_size']
        self.max_images_per_request = 100  # Documented limit
        
        # Add enhanced rate limiting from my version
        self.rate_limits = {
            'rpm': 50,          # Requests per minute
            'tpm': 40000,       # Tokens per minute
            'tpd': 1000000      # Tokens per day
        }
        
        # Add enhanced tracking
        self._request_timestamps: List[datetime] = []
        self._token_usage = {
            'minute': 0,
            'day': 0,
            'last_reset': datetime.now()
        }
        
        logger.info(
            f"VisionAnalyzer initialized with documented settings:\n"
            f"- Max retries: {self.anthropic.max_retries}\n"
            f"- Timeout: {self.anthropic.timeout}s\n"
            f"- Rate limits: {self.rate_limits['rpm']} RPM, {self.rate_limits['tpm']} TPM"
        )

    async def _respect_rate_limit(self) -> None:
        """
        Enhanced rate limit management combining both implementations
        """
        current_time = datetime.now()
        minute_ago = current_time - timedelta(minutes=1)
        
        # Clean old timestamps
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if ts > minute_ago
        ]
        
        # Check RPM limit
        if len(self._request_timestamps) >= self.rate_limits['rpm']:
            sleep_time = (self._request_timestamps[0] - minute_ago).total_seconds()
            logger.warning(f"Rate limit approaching, sleeping for {sleep_time:.2f}s")
            await asyncio.sleep(sleep_time + 0.1)  # Add small buffer
            
        # Add timestamp for new request
        self._request_timestamps.append(current_time)
        
        # Reset token usage if needed
        if (current_time - self._token_usage['last_reset']).total_seconds() >= 60:
            self._token_usage['minute'] = 0
            self._token_usage['last_reset'] = current_time

    # Keep your existing analyze_charts method but add enhanced error handling
    async def analyze_charts(
        self, 
        pair: str, 
        images: List[Tuple[str, ChartMetadata, str]]
    ) -> Dict:
        """
        Perform comprehensive chart analysis using Claude's vision capabilities
        """
        try:
            logger.info(f"Starting analysis of {len(images)} images for {pair}")
            all_responses = []
            
            for i, (img_data, metadata, mime_type) in enumerate(images, 1):
                try:
                    await self._respect_rate_limit()
                    
                    logger.info(f"Analyzing image {i} of {len(images)}")
                    
                    response = await self.anthropic.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=[{
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": mime_type,
                                        "data": img_data
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": self._create_vision_prompt(pair, [metadata])
                                }
                            ]
                        }]
                    )
                    
                    # Track token usage
                    response_tokens = len(response.content[0].text.split())  # Approximate
                    self._token_usage['minute'] += response_tokens
                    self._token_usage['day'] += response_tokens
                    
                    all_responses.append(response.content[0].text)
                    logger.info(f"Successfully analyzed image {i}")
                    
                except RateLimitError:
                    logger.warning("Rate limit hit, implementing backoff")
                    await asyncio.sleep(2)
                    continue
                    
                except BadRequestError as e:
                    logger.error(f"Invalid request: {str(e)}")
                    raise AnalysisError(f"Analysis request invalid: {str(e)}")
                    
                except AuthenticationError:
                    logger.error("Authentication failed")
                    raise AnalysisError(
                        "❌ API authentication failed. Please check API key configuration."
                    )
                    
                except Exception as e:
                    logger.error(f"Error processing image {i}: {traceback.format_exc()}")
                    continue

            if not all_responses:
                raise AnalysisError("No valid analysis results received")

            # Process responses
            analysis_result = self._parse_vision_response(
                self._combine_responses(all_responses)
            )
            
            analysis_result["metadata"] = {
                "analysis_time": datetime.now().isoformat(),
                "timeframes_analyzed": [meta.timeframe for _, meta, _ in images],
                "image_count": len(images),
                "confidence_score": self._calculate_confidence_score(analysis_result),
                "token_usage": {
                    "minute": self._token_usage['minute'],
                    "day": self._token_usage['day']
                }
            }
            
            return analysis_result

        except Exception as e:
            logger.error(f"Analysis failed: {traceback.format_exc()}")
            raise AnalysisError(f"Failed to analyze charts: {str(e)}")
        

    def _create_vision_prompt(self, pair: str, metadata: List[ChartMetadata]) -> str:
        """
        Create detailed prompt for Maverick Strategy vision analysis
        
        Args:
            pair: Trading pair being analyzed
            metadata: List of metadata for the images being analyzed
            
        Returns:
            Formatted prompt string for Claude
        """
        prompt = f"""Analyze these {pair} charts using the Maverick Trading Strategy framework.

Required Analysis Points:

1. Pattern Identification:
- Key support/resistance levels (provide exact price levels)
- Price action patterns and structures
- Volume analysis and accumulation/distribution zones
- RSI divergences, momentum, and trend strength
- High-probability reversal zones

2. Multi-timeframe Analysis:"""

        # Add timeframe-specific requirements
        for meta in metadata:
            if meta.timeframe == "4H":
                prompt += """
- 4H: Primary trend direction and market structure
  * Major support/resistance levels
  * RSI conditions and divergences
  * Volume profile analysis"""
            elif meta.timeframe == "1H":
                prompt += """
- 1H: Pattern confirmation and setup validation
  * Pattern completion status
  * RSI momentum analysis
  * Volume confirmation signals"""
            elif meta.timeframe == "15M":
                prompt += """
- 15M: Entry precision and execution timing
  * Immediate support/resistance
  * RSI trigger conditions
  * Volume trigger signals"""

        # Add setup classification requirements
        prompt += """

3. Setup Classification:
- Identify potential setup type:
  * Bounce Back: RSI conditions, support validation, volume expansion
  * Pullback: Trend continuation, retracement levels, volume contraction
  * Reversal: Structure break, RSI divergence, volume confirmation
  * Breakout: Level breach, momentum confirmation, volume surge

- High-quality trait validation:
  * Multiple timeframe confluence
  * Pattern reliability score
  * Volume confirmation strength
  * RSI alignment across timeframes
  * Risk-reward optimization

4. Risk Assessment:
- Entry zones with specific price levels
- Stop loss placement with technical justification
- Target projections based on structure
- Risk-reward ratios for different scenarios
- Grid trading parameters based on volatility

Please provide:
1. Exact price levels for all key zones
2. Rate confidence (1-5) for each identified pattern
3. Clear confluence points between timeframes
4. Quality assessment of setup based on confirmations
5. Risk-reward calculations

Format your analysis with clear sections and bullet points."""

        return prompt

    
    async def _make_request(
    self, 
    prompt: str, 
    images: List[Tuple[str, ChartMetadata, str]]
) -> str:
        """
        Make a request to Claude's API using the SDK
        
        Args:
            prompt: Analysis prompt for Claude
            images: List of image data tuples
            
        Returns:
            Claude's analysis response
            
        Raises:
            AnalysisError: If the request fails
        """
        try:
            # Log request details
            logger.debug(
                f"Making API request\n"
                f"Prompt length: {len(prompt)} chars\n"
                f"Number of images: {len(images)}"
            )

            # Create message content starting with images
            message_content = []
            
            # Add images first (following documented structure)
            for img_data, _, mime_type in images:
                message_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": img_data
                    }
                })
            
            # Add the text prompt last
            message_content.append({
                "type": "text",
                "text": prompt
            })

            # Make request using SDK with documented message structure
            response = await self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": message_content
                }]
            )
            
            logger.info("Received response from Claude API")
            return response.content[0].text

        except BadRequestError as e:
            logger.error(f"Bad request error: {str(e)}")
            raise AnalysisError(f"Invalid request format: {str(e)}")
        
        except AuthenticationError:
            logger.error("Authentication failed")
            raise AnalysisError(
                "❌ API authentication failed. Please check API key configuration."
            )
            
        except PermissionDeniedError:
            logger.error("Permission denied")
            raise AnalysisError(
                "❌ Permission denied. Please verify API access permissions."
            )
            
        except NotFoundError:
            logger.error("Resource not found")
            raise AnalysisError(
                "❌ Requested resource not found. Please verify API endpoint."
            )
            
        except UnprocessableEntityError as e:
            logger.error(f"Unprocessable entity: {str(e)}")
            raise AnalysisError(
                f"❌ Invalid request data: {str(e)}"
            )
            
        except RateLimitError:
            logger.error("Rate limit exceeded")
            raise AnalysisError(
                "❌ Rate limit exceeded. Please try again later."
            )
            
        except InternalServerError:
            logger.error("Anthropic server error")
            raise AnalysisError(
                "❌ Service temporarily unavailable. Please try again later."
            )
            
        except APIConnectionError:
            logger.error("Connection error")
            raise AnalysisError(
                "❌ Connection to API failed. Please try again later."
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {traceback.format_exc()}")
            raise AnalysisError(f"Analysis failed: {str(e)}")

    def _chunk_images(
        self, 
        images: List[Tuple[str, ChartMetadata, str]]
    ) -> List[List[Tuple[str, ChartMetadata, str]]]:
        """
        Split images into smaller chunks to prevent timeouts
        
        Args:
            images: List of image data tuples
            
        Returns:
            List of image chunks
        """
        chunks = []
        current_chunk = []
        current_size = 0
        
        for img_data, metadata, mime_type in images:
            img_size = len(img_data)
            
            # Start new chunk if current one is too large or has max images
            if (current_size + img_size > self.chunk_size or 
                len(current_chunk) >= self.max_images_per_request):
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
                
            current_chunk.append((img_data, metadata, mime_type))
            current_size += img_size

        if current_chunk:
            chunks.append(current_chunk)
            
        logger.info(
            f"Split {len(images)} images into {len(chunks)} chunks "
            f"(max {self.chunk_size/1024:.1f}KB per chunk)"
        )
        return chunks

    def _combine_responses(self, responses: List[str]) -> str:
        """
        Combine multiple analysis responses intelligently
        
        Args:
            responses: List of response strings from Claude
            
        Returns:
            Combined and formatted response string
        """
        sections = {
            "Pattern Recognition": [],
            "Multi-timeframe Analysis": [],
            "Setup Classification": [],
            "Risk Assessment": []
        }
        
        for response in responses:
            current_section = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Detect section headers
                for section in sections.keys():
                    if section in line:
                        current_section = section
                        break
                
                # Add content to appropriate section
                if current_section and line not in sections[current_section]:
                    sections[current_section].append(line)
        
        # Combine sections
        combined = []
        for section, content in sections.items():
            if content:
                combined.append(f"\n{section}:")
                combined.extend(content)
        
        return "\n".join(combined)

    def _parse_vision_response(self, response: str) -> Dict:
        """
        Parse Claude's vision analysis response into structured data
        
        Args:
            response: Combined response string
            
        Returns:
            Dictionary containing structured analysis data
            
        Raises:
            AnalysisError: If parsing fails
        """
        analysis_result = {
            "patterns": [],
            "levels": {
                "support": [],
                "resistance": []
            },
            "setup": {
                "type": None,
                "quality": "standard",
                "confirmations": []
            },
            "timeframes": {
                "4H": {},
                "1H": {},
                "15M": {}
            },
            "risk_assessment": {
                "entry_zones": [],
                "stops": [],
                "targets": [],
                "rr_ratio": None
            }
        }
        
        try:
            current_section = None
            current_timeframe = None
            
            for line in response.split('\n'):
                line = line.strip()
                if not line:
                    continue
                
                # Section detection
                if "Pattern Recognition:" in line:
                    current_section = "patterns"
                elif "Multi-timeframe Analysis:" in line:
                    current_section = "timeframes"
                elif "Setup Classification:" in line:
                    current_section = "setup"
                elif "Risk Assessment:" in line:
                    current_section = "risk"
                    
                # Timeframe detection
                elif any(tf in line for tf in ["4H:", "1H:", "15M:"]):
                    current_timeframe = line[:3]
                    
                # Content processing
                elif line.startswith(('•', '-', '*')):
                    content = line[1:].strip()
                    self._process_analysis_line(
                        content,
                        current_section,
                        current_timeframe,
                        analysis_result
                    )
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Response parsing error: {traceback.format_exc()}")
            raise AnalysisError(f"Failed to parse vision analysis: {str(e)}")

    def _process_analysis_line(
        self,
        content: str,
        section: str,
        timeframe: str,
        result: Dict
    ) -> None:
        """
        Process individual lines of the analysis response
        
        Args:
            content: Line content to process
            section: Current section being processed
            timeframe: Current timeframe context
            result: Analysis result dictionary to update
        """
        try:
            content_lower = content.lower()
            
            if section == "patterns":
                if "support" in content_lower:
                    self._extract_price_level(content, result["levels"]["support"])
                elif "resistance" in content_lower:
                    self._extract_price_level(content, result["levels"]["resistance"])
                else:
                    result["patterns"].append({
                        "description": content,
                        "confidence": self._estimate_pattern_confidence(content)
                    })
                    
            elif section == "timeframes" and timeframe:
                if timeframe not in result["timeframes"]:
                    result["timeframes"][timeframe] = {}
                    
                if "rsi" in content_lower:
                    result["timeframes"][timeframe]["rsi"] = self._extract_numeric_value(content)
                elif "volume" in content_lower:
                    result["timeframes"][timeframe]["volume"] = content
                elif "trend" in content_lower:
                    result["timeframes"][timeframe]["trend"] = content
                    
            elif section == "setup":
                for setup_type in ["bounce", "pullback", "reversal", "breakout"]:
                    if setup_type in content_lower:
                        result["setup"]["type"] = setup_type
                        break
                        
                if "confirmation" in content_lower:
                    result["setup"]["confirmations"].append(content)
                elif "high quality" in content_lower:
                    result["setup"]["quality"] = "high"
                    
            elif section == "risk":
                if "entry" in content_lower:
                    result["risk_assessment"]["entry_zones"].append(content)
                elif "stop" in content_lower:
                    result["risk_assessment"]["stops"].append(content)
                elif "target" in content_lower:
                    result["risk_assessment"]["targets"].append(content)
                elif "r:r" in content_lower or "risk/reward" in content_lower:
                    result["risk_assessment"]["rr_ratio"] = self._extract_numeric_value(content)
                    
        except Exception as e:
            logger.warning(f"Error processing line '{content}': {str(e)}")

    def _extract_price_level(self, content: str, levels_list: List) -> None:
        """
        Extract price level information from content
        
        Args:
            content: Content string containing price information
            levels_list: List to append extracted level information
        """
        try:
            numbers = re.findall(r'[\d.]+', content)
            if numbers:
                price = float(numbers[0])
                strength = 3  # Default strength
                
                # Adjust strength based on description
                if any(word in content.lower() for word in ["strong", "major", "key"]):
                    strength += 1
                if "multiple" in content.lower() or "tested" in content.lower():
                    strength += 1
                
                levels_list.append({
                    "price": price,
                    "description": content,
                    "strength": min(strength, 5)
                })
                
        except Exception as e:
            logger.warning(f"Error extracting price level from '{content}': {str(e)}")

    def _extract_numeric_value(self, content: str) -> Optional[float]:
        """
        Extract numeric value from content
        
        Args:
            content: Content string to extract number from
            
        Returns:
            Extracted float value or None if not found
        """
        try:
            numbers = re.findall(r'[\d.]+', content)
            return float(numbers[0]) if numbers else None
        except:
            return None

    def _estimate_pattern_confidence(self, content: str) -> float:
        """
        Estimate confidence level for pattern recognition
        
        Args:
            content: Pattern description content
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence = 0.7  # Base confidence
        
        content_lower = content.lower()
        # Adjust based on certainty words
        if any(word in content_lower for word in ["clear", "strong", "confirmed"]):
            confidence += 0.2
        elif any(word in content_lower for word in ["potential", "possible", "might"]):
            confidence -= 0.2
            
        return round(min(max(confidence, 0.1), 1.0), 2)

    def _calculate_confidence_score(self, analysis_result: Dict) -> float:
        """
        Calculate overall confidence score for the analysis
        
        Args:
            analysis_result: Complete analysis result dictionary
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score_factors = []
        
        # Pattern confidence
        pattern_scores = [p.get("confidence", 0) for p in analysis_result["patterns"]]
        if pattern_scores:
            score_factors.append(sum(pattern_scores) / len(pattern_scores))
            
        # Setup quality
        if analysis_result["setup"]["quality"] == "high":
            score_factors.append(1.0)
        else:
            score_factors.append(0.5)
            
        # Confirmation count
        conf_count = len(analysis_result["setup"]["confirmations"])
        score_factors.append(min(conf_count * 0.2, 1.0))
        
        # Risk/reward ratio
        if analysis_result["risk_assessment"]["rr_ratio"]:
            rr_score = min(analysis_result["risk_assessment"]["rr_ratio"] * 0.2, 1.0)
            score_factors.append(rr_score)
            
        return round(sum(score_factors) / max(len(score_factors), 1), 2)
    

class AnalysisState:
    """
    Maintains state for ongoing trading analysis session
    """
    def __init__(self):
        self.current_step = 0
        self.pair = ""
        self.timestamps = []
        self.patterns = []
        self.confirmations: List[Confirmation] = []
        self.setup_type: Optional[SetupType] = None
        self.entry_price: Optional[Decimal] = None
        self.stop_loss: Optional[Decimal] = None
        self.targets: List[Decimal] = []
        self.is_high_quality: bool = False
        self.grid_levels: List[GridLevel] = []
        self.last_update = datetime.now()

    def update_state(self, **kwargs) -> None:
        """
        Update analysis state with new information
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = datetime.now()
        logger.debug(f"Analysis state updated: {', '.join(kwargs.keys())}")

class SetupValidator:
    """
    Validates trading setups against required conditions
    """
    def __init__(self):
        self.high_quality_threshold = 7  # Minimum confirmations for High Quality
        self.setup_requirements = {
            SetupType.BOUNCE_BACK: {
                "confirmations": [
                    "RSI 4H crosses above 30",
                    "Double bottom in 15m",
                    "Volume above average",
                    "Reversal candle in 1H"
                ],
                "min_rr_ratio": Decimal('1.5'),
                "confluence_required": 3
            },
            SetupType.PULLBACK: {
                "confirmations": [
                    "RSI 4H above 35",
                    "EMA 50 15m as support",
                    "Decreasing volume on retrace",
                    "Minimum 2 bullish candles in 1H"
                ],
                "min_rr_ratio": Decimal('1.8'),
                "confluence_required": 3
            },
            SetupType.REVERSAL: {
                "confirmations": [
                    "RSI divergence on 4H",
                    "Break of market structure",
                    "Volume confirmation",
                    "Higher highs forming on 1H"
                ],
                "min_rr_ratio": Decimal('2.0'),
                "confluence_required": 4
            },
            SetupType.BREAKOUT: {
                "confirmations": [
                    "Clear resistance break",
                    "Volume expansion",
                    "RSI momentum",
                    "Pullback retest"
                ],
                "min_rr_ratio": Decimal('1.5'),
                "confluence_required": 3
            }
        }
        
        logger.info(f"SetupValidator initialized with {len(self.setup_requirements)} setup types")
        
    async def validate_setup(self, 
                           setup_type: SetupType, 
                           analysis_result: dict,
                           current_price: Decimal) -> Dict:
        """
        Validate setup against required confirmations
        
        Args:
            setup_type: Type of setup to validate
            analysis_result: Analysis results from vision analysis
            current_price: Current price for calculations
            
        Returns:
            Dictionary containing validation results
            
        Raises:
            ValidationError: If setup type is invalid
        """
        try:
            requirements = self.setup_requirements.get(setup_type)
            if not requirements:
                raise ValidationError(f"Invalid setup type: {setup_type}")

            # Validate confirmations
            confirmations = []
            for req in requirements["confirmations"]:
                status = self._check_confirmation(req, analysis_result)
                confirmations.append(Confirmation(
                    description=req,
                    status=status,
                    timestamp=datetime.now()
                ))

            # Calculate confirmation score
            confirmed_count = sum(1 for conf in confirmations if conf.status)
            
            # Check for high quality conditions
            is_high_quality = (
                confirmed_count >= self.high_quality_threshold and
                self._validate_risk_reward(
                    analysis_result, 
                    current_price, 
                    requirements["min_rr_ratio"]
                )
            )

            validation_result = {
                "setup_type": setup_type.value,
                "confirmations": confirmations,
                "confirmation_score": f"{confirmed_count}/{len(requirements['confirmations'])}",
                "is_high_quality": is_high_quality,
                "missing_confirmations": [
                    conf.description for conf in confirmations if not conf.status
                ],
                "risk_reward_valid": self._validate_risk_reward(
                    analysis_result, 
                    current_price,
                    requirements["min_rr_ratio"]
                )
            }

            logger.info(
                f"Setup validation completed: {setup_type.value}, "
                f"Score: {validation_result['confirmation_score']}, "
                f"High Quality: {is_high_quality}"
            )

            return validation_result

        except Exception as e:
            logger.error(f"Setup validation error: {traceback.format_exc()}")
            raise ValidationError(f"Failed to validate setup: {str(e)}")

    def _check_confirmation(self, requirement: str, analysis_result: dict) -> bool:
        """
        Check if a specific confirmation requirement is met
        
        Args:
            requirement: Required confirmation description
            analysis_result: Analysis results dictionary
            
        Returns:
            Boolean indicating if requirement is met
        """
        requirement_lower = requirement.lower()
        
        # Check patterns
        for pattern in analysis_result.get("patterns", []):
            if isinstance(pattern, dict):
                pattern_desc = pattern.get("description", "").lower()
            else:
                pattern_desc = str(pattern).lower()
            
            if any(term in pattern_desc for term in requirement_lower.split()):
                return True
                
        # Check confluence
        for conf in analysis_result.get("confluence", []):
            if any(term in conf.lower() for term in requirement_lower.split()):
                return True
                
        # Check high quality traits
        for trait in analysis_result.get("high_quality_traits", []):
            if any(term in trait.lower() for term in requirement_lower.split()):
                return True
                
        return False

    def _validate_risk_reward(self, 
                            analysis_result: dict, 
                            current_price: Decimal,
                            min_rr: Decimal) -> bool:
        """
        Validate risk-reward ratio meets minimum requirement
        
        Args:
            analysis_result: Analysis results dictionary
            current_price: Current price for calculations
            min_rr: Minimum required risk-reward ratio
            
        Returns:
            Boolean indicating if risk-reward is valid
        """
        try:
            stops = analysis_result.get("risk_assessment", {}).get("stops", [])
            targets = analysis_result.get("risk_assessment", {}).get("targets", [])
            
            if not stops or not targets:
                return False
                
            # Extract numeric values
            stop_prices = self._extract_prices(stops)
            target_prices = self._extract_prices(targets)
            
            if not stop_prices or not target_prices:
                return False
                
            # Calculate risk-reward ratio
            stop_loss = min(abs(Decimal(str(price)) - current_price) for price in stop_prices)
            take_profit = max(abs(Decimal(str(price)) - current_price) for price in target_prices)
            
            rr_ratio = take_profit / stop_loss
            
            logger.debug(
                f"Risk-Reward calculation: {rr_ratio:.2f} "
                f"(Required: {min_rr})"
            )
            
            return rr_ratio >= min_rr
            
        except Exception as e:
            logger.error(f"Risk-reward validation error: {traceback.format_exc()}")
            return False

    def _extract_prices(self, level_list: List[str]) -> List[float]:
        """
        Extract price values from text descriptions
        
        Args:
            level_list: List of price level descriptions
            
        Returns:
            List of extracted price values
        """
        prices = []
        for level in level_list:
            matches = re.findall(r"[\d.]+", level)
            try:
                prices.extend(float(match) for match in matches)
            except ValueError:
                continue
        return prices

class SetupAnalysis:
    """
    Coordinates trading setup analysis and validation
    """
    def __init__(self, anthropic_client: Anthropic):
        self.anthropic = anthropic_client
        self.validator = SetupValidator()
        self.pattern_analyzer = PatternAnalysis(anthropic_client)
        
        logger.info("SetupAnalysis initialized")

    async def analyze_setup(self,
                          pair: str,
                          images: List[Tuple[str, ChartMetadata, str]],
                          current_price: Decimal) -> Dict:
        """
        Analyze and validate trading setup
        
        Args:
            pair: Trading pair being analyzed
            images: List of tuples containing (base64_data, metadata, mime_type)
            current_price: Current price for calculations
            
        Returns:
            Dictionary containing analysis and validation results
            
        Raises:
            ValidationError: If analysis fails
        """
        try:
            logger.info(f"Starting setup analysis for {pair}")
            
            # Get initial pattern analysis
            analysis_result = await self.pattern_analyzer.analyze_images(images, pair)
            
            # Determine setup type
            setup_type = self._determine_setup_type(analysis_result)
            if not setup_type:
                raise ValidationError("Could not determine setup type from analysis")
            
            # Validate setup
            validation_result = await self.validator.validate_setup(
                setup_type,
                analysis_result,
                current_price
            )
            
            # Combine results
            final_result = {
                **validation_result,
                "analysis": analysis_result,
                "metadata": {
                    "analysis_time": datetime.now().isoformat(),
                    "pair": pair,
                    "current_price": str(current_price),
                    "image_count": len(images)
                }
            }
            
            logger.info(
                f"Setup analysis completed: {setup_type.value}, "
                f"High Quality: {validation_result['is_high_quality']}"
            )
            
            return final_result
            
        except Exception as e:
            logger.error(f"Setup analysis error: {traceback.format_exc()}")
            raise ValidationError(f"Failed to analyze setup: {str(e)}")

    def _determine_setup_type(self, analysis_result: dict) -> Optional[SetupType]:
        """
        Determine setup type from analysis results
        
        Args:
            analysis_result: Analysis results dictionary
            
        Returns:
            SetupType enum or None if not determined
        """
        setup_str = analysis_result.get("setup", {}).get("type", "").upper()
        
        for setup_type in SetupType:
            if setup_type.name in setup_str or setup_type.value.upper() in setup_str:
                return setup_type
                
        return None

class PatternAnalysis:
    """
    Analyzes trading patterns and market structure
    """
    def __init__(self, anthropic_client: Anthropic):
        self.anthropic = anthropic_client
        self.vision_analyzer = VisionAnalyzer(anthropic_client)
        
        logger.info("PatternAnalysis initialized")

    async def analyze_images(self, 
                           images: List[Tuple[str, ChartMetadata, str]], 
                           pair: str) -> Dict:
        """
        Analyze trading patterns from chart images
        
        Args:
            images: List of tuples containing (base64_data, metadata, mime_type)
            pair: Trading pair being analyzed
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            logger.info(f"Starting pattern analysis for {pair}")
            
            # Get vision analysis
            analysis_result = await self.vision_analyzer.analyze_charts(pair, images)
            
            # Enhance analysis
            enhanced_result = self._enhance_analysis(analysis_result)
            
            # Calculate confidence scores
            enhanced_result["confidence_scores"] = self._calculate_confidence_scores(
                enhanced_result
            )
            
            logger.info("Pattern analysis completed successfully")
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {traceback.format_exc()}")
            raise AnalysisError(f"Failed to analyze patterns: {str(e)}")

    def _enhance_analysis(self, analysis_result: Dict) -> Dict:
        """
        Enhance raw analysis with additional insights
        
        Args:
            analysis_result: Raw analysis results
            
        Returns:
            Enhanced analysis dictionary
        """
        try:
            enhanced = analysis_result.copy()
            
            # Add pattern categorization
            if enhanced.get("patterns"):
                enhanced["patterns"] = self._categorize_patterns(
                    enhanced["patterns"]
                )
            
            # Add confluence analysis
            enhanced["confluence"] = self._analyze_confluence(enhanced)
            
            # Add trend strength
            enhanced["trend_strength"] = self._calculate_trend_strength(enhanced)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Analysis enhancement error: {str(e)}")
            return analysis_result

    def _categorize_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """
        Categorize patterns by type and significance
        
        Args:
            patterns: List of pattern dictionaries
            
        Returns:
            List of categorized pattern dictionaries
        """
        categorized = []
        for pattern in patterns:
            try:
                pattern_type = self._determine_pattern_type(pattern["description"])
                significance = self._calculate_pattern_significance(
                    pattern["description"],
                    pattern.get("confidence", 0.5)
                )
                
                categorized.append({
                    **pattern,
                    "type": pattern_type,
                    "significance": significance
                })
                
            except Exception as e:
                logger.warning(f"Pattern categorization error: {str(e)}")
                categorized.append(pattern)
                
        return categorized

    def _determine_pattern_type(self, description: str) -> str:
        """
        Determine pattern type from description
        
        Args:
            description: Pattern description
            
        Returns:
            Pattern type classification
        """
        desc_lower = description.lower()
        
        if any(word in desc_lower for word in ["support", "resistance", "level"]):
            return "structure"
        elif any(word in desc_lower for word in ["divergence", "rsi", "momentum"]):
            return "momentum"
        elif any(word in desc_lower for word in ["volume", "accumulation"]):
            return "volume"
        elif any(word in desc_lower for word in ["trend", "moving average"]):
            return "trend"
        else:
            return "price_action"

    def _calculate_pattern_significance(
        self, 
        description: str,
        confidence: float
    ) -> int:
        """
        Calculate pattern significance score
        
        Args:
            description: Pattern description
            confidence: Pattern confidence score
            
        Returns:
            Significance score (1-5)
        """
        base_score = int(confidence * 3)
        description_lower = description.lower()
        
        # Adjust based on keywords
        if any(word in description_lower for word in ["major", "strong", "key", "critical"]):
            base_score += 1
        if any(word in description_lower for word in ["confirmed", "validated", "tested"]):
            base_score += 1
        if any(word in description_lower for word in ["weak", "minor", "potential"]):
            base_score -= 1
            
        # Ensure score is within valid range
        return min(max(base_score, 1), 5)

    def _analyze_confluence(self, analysis: Dict) -> Dict:
        """
        Analyze confluences between different analysis components
        
        Args:
            analysis: Complete analysis dictionary
            
        Returns:
            Dictionary containing confluence analysis
        """
        try:
            confluence = {
                "price_levels": [],
                "momentum": [],
                "volume": [],
                "timeframe_alignment": []
            }
            
            # Check price level confluences
            if analysis.get("levels"):
                confluence["price_levels"] = self._find_price_level_confluence(
                    analysis["levels"]
                )
            
            # Check momentum confluences
            if analysis.get("timeframes"):
                confluence["momentum"] = self._find_momentum_confluence(
                    analysis["timeframes"]
                )
            
            # Check volume confluences
            if "patterns" in analysis:
                confluence["volume"] = self._find_volume_confluence(
                    analysis["patterns"],
                    analysis.get("timeframes", {})
                )
            
            # Check timeframe alignments
            if analysis.get("timeframes"):
                confluence["timeframe_alignment"] = self._check_timeframe_alignment(
                    analysis["timeframes"]
                )
            
            return confluence
            
        except Exception as e:
            logger.error(f"Confluence analysis error: {traceback.format_exc()}")
            return {}

    def _find_price_level_confluence(self, levels: Dict) -> List[Dict]:
        """
        Find confluent price levels across different timeframes
        
        Args:
            levels: Dictionary containing support and resistance levels
            
        Returns:
            List of confluent price levels with metadata
        """
        confluent_levels = []
        
        try:
            all_levels = []
            # Combine support and resistance levels
            for level_type in ["support", "resistance"]:
                for level in levels.get(level_type, []):
                    if isinstance(level, dict) and "price" in level:
                        all_levels.append({
                            **level,
                            "type": level_type
                        })
            
            # Sort levels by price
            all_levels.sort(key=lambda x: x["price"])
            
            # Find confluent levels (within 0.5% range)
            current_confluence = []
            for level in all_levels:
                if not current_confluence:
                    current_confluence = [level]
                else:
                    price_diff_percent = abs(
                        level["price"] - current_confluence[0]["price"]
                    ) / current_confluence[0]["price"] * 100
                    
                    if price_diff_percent <= 0.5:
                        current_confluence.append(level)
                    else:
                        if len(current_confluence) > 1:
                            confluent_levels.append({
                                "price": sum(l["price"] for l in current_confluence) / len(current_confluence),
                                "strength": max(l.get("strength", 1) for l in current_confluence),
                                "count": len(current_confluence),
                                "types": [l["type"] for l in current_confluence]
                            })
                        current_confluence = [level]
            
            # Add last confluence group if exists
            if len(current_confluence) > 1:
                confluent_levels.append({
                    "price": sum(l["price"] for l in current_confluence) / len(current_confluence),
                    "strength": max(l.get("strength", 1) for l in current_confluence),
                    "count": len(current_confluence),
                    "types": [l["type"] for l in current_confluence]
                })
            
            return confluent_levels
            
        except Exception as e:
            logger.error(f"Price level confluence error: {str(e)}")
            return []

    def _find_momentum_confluence(self, timeframes: Dict) -> List[Dict]:
        """
        Find momentum confluences across timeframes
        
        Args:
            timeframes: Dictionary containing timeframe analysis
            
        Returns:
            List of momentum confluences
        """
        confluences = []
        
        try:
            # Check RSI alignment
            rsi_values = {}
            for tf, data in timeframes.items():
                if isinstance(data, dict) and "rsi" in data:
                    rsi_values[tf] = data["rsi"]
            
            if len(rsi_values) >= 2:
                # Check for oversold confluence
                oversold_tfs = [
                    tf for tf, rsi in rsi_values.items() 
                    if isinstance(rsi, (int, float)) and rsi < 30
                ]
                if oversold_tfs:
                    confluences.append({
                        "type": "oversold",
                        "timeframes": oversold_tfs,
                        "strength": len(oversold_tfs)
                    })
                
                # Check for overbought confluence
                overbought_tfs = [
                    tf for tf, rsi in rsi_values.items() 
                    if isinstance(rsi, (int, float)) and rsi > 70
                ]
                if overbought_tfs:
                    confluences.append({
                        "type": "overbought",
                        "timeframes": overbought_tfs,
                        "strength": len(overbought_tfs)
                    })
            
            return confluences
            
        except Exception as e:
            logger.error(f"Momentum confluence error: {str(e)}")
            return []

    def _find_volume_confluence(self, patterns: List[Dict], timeframes: Dict) -> List[Dict]:
        """
        Find volume confluences across patterns and timeframes
        
        Args:
            patterns: List of identified patterns
            timeframes: Dictionary containing timeframe analysis
            
        Returns:
            List of volume confluences
        """
        confluences = []
        
        try:
            # Extract volume patterns
            volume_patterns = [
                p for p in patterns 
                if p.get("type") == "volume"
            ]
            
            # Extract volume conditions from timeframes
            volume_conditions = {}
            for tf, data in timeframes.items():
                if isinstance(data, dict) and "volume" in data:
                    volume_conditions[tf] = data["volume"]
            
            # Find confluences
            if volume_patterns and volume_conditions:
                for pattern in volume_patterns:
                    matching_tfs = []
                    pattern_desc = pattern["description"].lower()
                    
                    for tf, condition in volume_conditions.items():
                        condition_lower = condition.lower()
                        if (
                            ("increase" in pattern_desc and "increase" in condition_lower) or
                            ("decrease" in pattern_desc and "decrease" in condition_lower) or
                            ("above average" in pattern_desc and "above average" in condition_lower)
                        ):
                            matching_tfs.append(tf)
                    
                    if matching_tfs:
                        confluences.append({
                            "type": "volume",
                            "pattern": pattern["description"],
                            "timeframes": matching_tfs,
                            "strength": len(matching_tfs)
                        })
            
            return confluences
            
        except Exception as e:
            logger.error(f"Volume confluence error: {str(e)}")
            return []

    def _check_timeframe_alignment(self, timeframes: Dict) -> List[Dict]:
        """
        Check for trend alignments across timeframes
        
        Args:
            timeframes: Dictionary containing timeframe analysis
            
        Returns:
            List of timeframe alignments
        """
        alignments = []
        
        try:
            # Extract trends
            trends = {}
            for tf, data in timeframes.items():
                if isinstance(data, dict) and "trend" in data:
                    trends[tf] = data["trend"].lower()
            
            if len(trends) >= 2:
                # Check bullish alignment
                bullish_tfs = [
                    tf for tf, trend in trends.items()
                    if "bullish" in trend or "uptrend" in trend
                ]
                if len(bullish_tfs) >= 2:
                    alignments.append({
                        "type": "bullish",
                        "timeframes": bullish_tfs,
                        "strength": len(bullish_tfs)
                    })
                
                # Check bearish alignment
                bearish_tfs = [
                    tf for tf, trend in trends.items()
                    if "bearish" in trend or "downtrend" in trend
                ]
                if len(bearish_tfs) >= 2:
                    alignments.append({
                        "type": "bearish",
                        "timeframes": bearish_tfs,
                        "strength": len(bearish_tfs)
                    })
            
            return alignments
            
        except Exception as e:
            logger.error(f"Timeframe alignment error: {str(e)}")
            return []

    def _calculate_trend_strength(self, analysis: Dict) -> Dict:
        """
        Calculate overall trend strength across timeframes
        
        Args:
            analysis: Complete analysis dictionary
            
        Returns:
            Dictionary containing trend strength metrics
        """
        try:
            trend_metrics = {
                "primary": "neutral",
                "strength": 0,
                "confidence": 0.0,
                "supporting_factors": []
            }
            
            # Analyze timeframe trends
            if analysis.get("timeframes"):
                trend_metrics.update(
                    self._analyze_timeframe_trends(analysis["timeframes"])
                )
            
            # Add pattern confirmations
            if analysis.get("patterns"):
                trend_metrics.update(
                    self._analyze_trend_patterns(
                        analysis["patterns"],
                        trend_metrics["primary"]
                    )
                )
            
            return trend_metrics
            
        except Exception as e:
            logger.error(f"Trend strength calculation error: {str(e)}")
            return {
                "primary": "neutral",
                "strength": 0,
                "confidence": 0.0,
                "supporting_factors": []
            }

    def _calculate_confidence_scores(self, analysis: Dict) -> Dict:
        """
        Calculate confidence scores for different aspects of analysis
        
        Args:
            analysis: Complete analysis dictionary
            
        Returns:
            Dictionary containing confidence scores
        """
        try:
            scores = {
                "overall": 0.0,
                "pattern_recognition": 0.0,
                "momentum": 0.0,
                "volume": 0.0,
                "trend": 0.0
            }
            
            # Pattern recognition confidence
            if analysis.get("patterns"):
                pattern_scores = [
                    p.get("significance", 0) / 5.0 
                    for p in analysis["patterns"]
                ]
                scores["pattern_recognition"] = sum(pattern_scores) / len(pattern_scores)
            
            # Momentum confidence
            if analysis.get("confluence", {}).get("momentum"):
                momentum_count = len(analysis["confluence"]["momentum"])
                scores["momentum"] = min(momentum_count * 0.25, 1.0)
            
            # Volume confidence
            if analysis.get("confluence", {}).get("volume"):
                volume_count = len(analysis["confluence"]["volume"])
                scores["volume"] = min(volume_count * 0.25, 1.0)
            
            # Trend confidence
            if analysis.get("trend_strength", {}).get("confidence"):
                scores["trend"] = analysis["trend_strength"]["confidence"]
            
            # Overall confidence
            valid_scores = [s for s in scores.values() if s > 0]
            if valid_scores:
                scores["overall"] = sum(valid_scores) / len(valid_scores)
            
            return scores
            
        except Exception as e:
            logger.error(f"Confidence score calculation error: {str(e)}")
            return {
                "overall": 0.0,
                "pattern_recognition": 0.0,
                "momentum": 0.0,
                "volume": 0.0,
                "trend": 0.0
            }
        

# Initialize Discord bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

class TradingAnalysisBot:
    """
    Main bot class handling Discord interactions and analysis coordination
    """
    
    def __init__(self):
        # Initialize with AsyncAnthropic using documented best practices
        self.anthropic = AsyncAnthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY"),
            timeout=20.0,    # 20 seconds timeout
            max_retries=2    # default retry count
        )
        
        # Initialize analyzers with configured client
        self.vision_analyzer = VisionAnalyzer(self.anthropic)
        self.image_validator = ChartImageValidator(Config)
        self.setup_analyzer = SetupAnalysis(self.anthropic)
        
        # State management
        self.active_analyses: Dict[int, AnalysisState] = {}
        self.start_time = datetime.now()
        self.cleanup_task = None
        self.api_enabled = True
        
        # Service limits based on SDK documentation
        self.api_limits = {
            'rpm': 50,          # Requests per minute for Claude 3.5 Sonnet
            'tpm': 40000,       # Tokens per minute
            'tpd': 1000000,     # Tokens per day
            'max_images': 100,  # Maximum images per request (updated)
            'max_image_size': 5 * 1024 * 1024  # 5MB per image
        }
        
        logger.info(
            f"Trading Analysis Bot initialized:\n"
            f"- Timeout: 20.0s\n"
            f"- Max retries: 2\n"
            f"- Rate limits: {self.api_limits['rpm']} RPM, "
            f"{self.api_limits['tpm']} TPM"
        )

    async def start(self):
        """Initialize async components"""
        try:
            # Test API connection
            await self._test_api_connection()
            
            # Start cleanup task
            self.cleanup_task = asyncio.create_task(self._start_cleanup_loop())
            logger.info("Cleanup task started")
            
        except Exception as e:
            logger.error(f"Initialization error: {traceback.format_exc()}")
            raise

    async def _test_api_connection(self):
        """Test API connection and authentication"""
        try:
            # Simple test request
            await self.anthropic.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [{"type": "text", "text": "test"}]
                }]
            )
            logger.info("API connection test successful")
            
        except BadRequestError as e:
            logger.error(f"Bad request error: {str(e)}")
            raise AnalysisError(f"Invalid request format: {str(e)}")
            
        except AuthenticationError:
            logger.error("Authentication failed")
            raise AnalysisError(
                "❌ API authentication failed. Please check API key configuration."
            )
            
        except PermissionDeniedError:
            logger.error("Permission denied")
            raise AnalysisError(
                "❌ Permission denied. Please verify API access permissions."
            )
            
        except NotFoundError:
            logger.error("Resource not found")
            raise AnalysisError(
                "❌ Requested resource not found. Please verify API endpoint."
            )
            
        except UnprocessableEntityError as e:
            logger.error(f"Unprocessable entity: {str(e)}")
            raise AnalysisError(
                f"❌ Invalid request data: {str(e)}"
            )
            
        except RateLimitError:
            logger.error("Rate limit exceeded")
            raise AnalysisError(
                "❌ Rate limit exceeded. Please try again later."
            )
            
        except InternalServerError:
            logger.error("Anthropic server error")
            raise AnalysisError(
                "❌ Service temporarily unavailable. Please try again later."
            )
            
        except APIConnectionError:
            logger.error("Connection error")
            raise AnalysisError(
                "❌ Connection to API failed. Please try again later."
            )
            
        except Exception as e:
            logger.error(f"Unexpected error: {traceback.format_exc()}")
            raise AnalysisError(f"API test failed: {str(e)}")

    async def cleanup(self):
        """Cleanup resources on shutdown"""
        if self.cleanup_task and not self.cleanup_task.cancelled():
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info("Cleanup task stopped")
    

    async def check_status(self) -> bool:
        """Check if all components are properly initialized"""
        try:
            # Test anthropic client
            if not self.anthropic.api_key:
                raise ValueError("Anthropic API key not set")
                
            # Test image processing with appropriate dimensions
            test_image = Image.new('RGB', (400, 300))  # Create larger test image
            test_bytes = io.BytesIO()
            test_image.save(test_bytes, format='PNG')
            test_bytes = test_bytes.getvalue()
            
            # Basic validation test without network calls
            try:
                image = Image.open(io.BytesIO(test_bytes))
                processed_image = self.image_validator._process_image(image, 'png')
                if not processed_image:
                    raise ValueError("Image processing test failed")
                logger.info("Image processing test successful")
            except Exception as e:
                raise ValueError(f"Image processing test failed: {str(e)}")
                
            # Test cleanup task
            if not self.cleanup_task or self.cleanup_task.done():
                raise ValueError("Cleanup task not running")
                
            logger.info("All component checks passed successfully")
            return True
                
        except Exception as e:
            logger.error(f"Status check failed: {str(e)}")
            return False
        
    async def _start_cleanup_loop(self):
        """Start the periodic cleanup loop"""
        try:
            while True:
                await self.cleanup_old_sessions()
                await asyncio.sleep(Config.CLEANUP_SETTINGS['cleanup_interval'])
        except Exception as e:
            logger.error(f"Cleanup loop error: {traceback.format_exc()}")

    async def cleanup_old_sessions(self):
        """Cleanup old analysis sessions"""
        try:
            current_time = datetime.now()
            cleaned_count = 0
            
            # Cleanup old sessions
            for user_id, state in list(self.active_analyses.items()):
                if (current_time - state.last_update).total_seconds() > Config.CLEANUP_SETTINGS['session_timeout']:
                    del self.active_analyses[user_id]
                    cleaned_count += 1
            
            # Check if we're over the max sessions limit
            if len(self.active_analyses) > Config.CLEANUP_SETTINGS['max_active_sessions']:
                # Sort by last update and remove oldest
                sorted_sessions = sorted(
                    self.active_analyses.items(),
                    key=lambda x: x[1].last_update
                )
                to_remove = len(self.active_analyses) - Config.CLEANUP_SETTINGS['max_active_sessions']
                
                for user_id, _ in sorted_sessions[:to_remove]:
                    del self.active_analyses[user_id]
                    cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} old analysis sessions")
                
        except Exception as e:
            logger.error(f"Session cleanup error: {traceback.format_exc()}")

    async def handle_analysis_request(
        self,
        ctx: commands.Context,
        pair: str
    ) -> None:
        """
        Handle initial analysis request and guide user through the process
        
        Args:
            ctx: Discord context
            pair: Trading pair to analyze
        """
        try:
            # Create analysis session
            user_id = ctx.author.id
            self.active_analyses[user_id] = AnalysisState()
            self.active_analyses[user_id].update_state(pair=pair)
            
            # Create instruction embed
            embed = discord.Embed(
                title=f"📊 Chart Analysis - {pair}",
                description="Please provide trading chart images for analysis",
                color=0x00ff00
            )
            
            embed.add_field(
                name="Required Timeframes",
                value=(
                    "• 4H (Primary Analysis)\n"
                    "• 1H (Pattern Confirmation)\n"
                    "• 15M (Entry Timing)"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Image Requirements",
                value=(
                    "• Supported formats: JPEG, PNG, GIF, WebP\n"
                    "• Maximum size: 2MB per image\n"
                    "• Include RSI indicator\n"
                    "• Show volume data"
                ),
                inline=False
            )
            
            embed.add_field(
                name="Next Steps",
                value=(
                    "1. Upload your chart images\n"
                    "2. Wait for analysis completion\n"
                    "3. Follow setup validation prompts"
                ),
                inline=False
            )
            
            await ctx.send(embed=embed)
            
            logger.info(f"Analysis session started for user {user_id} - {pair}")
            
        except Exception as e:
            logger.error(f"Error starting analysis: {traceback.format_exc()}")
            await ctx.send("❌ Failed to start analysis session.")

    async def handle_image_upload(
    self,
    message: discord.Message,
    attachments: List[discord.Attachment]
) -> None:
        """
        Handle uploaded images for analysis
        
        Args:
            message: Discord message object
            attachments: List of uploaded attachments
        """
        if not self.api_enabled:
            await message.channel.send(
                "⚠️ **API Currently Disabled**\n"
                "The image analysis feature is temporarily unavailable. "
                "Please try again later."
            )
            return
        
        try:
            user_id = message.author.id
            if user_id not in self.active_analyses:
                await message.channel.send(
                    "⚠️ No active analysis session. Start with `!analyze <pair>` first."
                )
                return
                
            state = self.active_analyses[user_id]
            
            # Process images
            progress_msg = await message.channel.send("🔄 Processing images...")
            processed_images = []
            
            for attachment in attachments:
                try:
                    # Detect timeframe from filename
                    timeframe = self._detect_timeframe(attachment.filename)
                    
                    # Process image
                    img_data, metadata, mime_type = await self.image_validator.validate_and_process_image(
                        attachment.url,
                        timeframe
                    )
                    
                    processed_images.append((img_data, metadata, mime_type))
                    logger.info(
                        f"Processed image: {attachment.filename} "
                        f"({metadata.dimensions[0]}x{metadata.dimensions[1]}px)"
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Error processing {attachment.filename}: "
                        f"{traceback.format_exc()}"
                    )
                    await message.channel.send(f"⚠️ Error processing {attachment.filename}")
                    continue
            
            if not processed_images:
                await progress_msg.edit(content="❌ No valid images to analyze.")
                return
                
            # Update progress
            await progress_msg.edit(
                content="🔄 Images processed. Starting analysis..."
            )
            
            # Perform analysis
            try:
                analysis_result = await self.vision_analyzer.analyze_charts(
                    state.pair,
                    processed_images
                )
                
                # Update state and send results
                state.update_state(
                    patterns=analysis_result.get("patterns", []),
                    is_high_quality=analysis_result.get("setup", {}).get("quality") == "high"
                )
                
                await self._send_analysis_results(message, analysis_result)
                
            except AnalysisError as e:
                if "credit balance is too low" in str(e):
                    await message.channel.send(
                        "❌ **API Credit Limit Reached**\n"
                        "The bot's Anthropic API credits have been exhausted. "
                        "Please contact the bot administrator to upgrade the API plan."
                    )
                    logger.error("Analysis failed due to API credit limit")
                else:
                    await message.channel.send(f"❌ Analysis error: {str(e)}")
                    logger.error(f"Analysis error: {traceback.format_exc()}")
                
        except Exception as e:
            logger.error(f"Image handling error: {traceback.format_exc()}")
            await message.channel.send("❌ Error processing images.")

    async def _send_analysis_results(self, message: discord.Message, analysis_result: Dict) -> None:
        try:
            # Main Analysis Embed with Status Overview
            main_embed = discord.Embed(
                title=f"📊 Analysis Results - {self.active_analyses[message.author.id].pair}",
                color=0x00ff00
            )

            # Analysis Quality Check
            missing_elements = []
            warnings = []
            
            # Check Pattern Analysis
            patterns = analysis_result.get("patterns", [])
            if not patterns:
                missing_elements.append("Pattern Analysis")
                warnings.append("• Add more visible price action patterns")
            else:
                patterns_text = []
                for pattern in patterns:
                    confidence = pattern.get("confidence", 0) * 100
                    emoji = "🟢" if confidence > 70 else "🟡" if confidence > 50 else "🔴"
                    patterns_text.append(f"{emoji} {pattern['description']} ({confidence:.0f}%)")
                
                main_embed.add_field(
                    name="📈 Pattern Analysis",
                    value="```" + "\n".join(patterns_text) + "```",
                    inline=False
                )

            await message.channel.send(embed=main_embed)

            # Setup Classification with Enhanced Information
            setup_embed = discord.Embed(
                title="🎯 Setup Classification & Quality Check",
                color=0x00ff00 if not missing_elements else 0xffaa00
            )

            setup_info = analysis_result.get("setup", {})
            if not isinstance(setup_info, dict):
                setup_info = {}

            setup_type = setup_info.get('type')
            setup_type_display = setup_type.title() if isinstance(setup_type, str) else 'Not Identified'
            
            quality_status = '🌟 High' if setup_info.get('quality') == 'high' else '⚡ Standard'
            confidence_score = analysis_result.get('metadata', {}).get('confidence_score', 0) * 100

            setup_status = [
                f"Type: {setup_type_display}",
                f"Quality: {quality_status}",
                f"Confidence: {confidence_score:.0f}%",
                f"Analysis Coverage: {100 - len(missing_elements)*10:.0f}%"
            ]

            setup_embed.add_field(
                name="Setup Status",
                value="```" + "\n".join(setup_status) + "```",
                inline=False
            )

            # Add Missing Information Alert if needed
            if missing_elements:
                # Solution 2: Using format() method instead of f-string
                message_parts = [
                    "⚠️ **Missing Information Alert**",
                    "The following elements need attention:\n• {}".format("\n• ".join(missing_elements)),
                    "**Suggested Improvements:**",
                    *warnings  # Unpack warnings list
                ]
                            
                # Join with newlines
                missing_info = "\n\n".join(message_parts)
                
                setup_embed.add_field(
                    name="Analysis Completion Status",
                    value=missing_info,
                    inline=False
                )

            await message.channel.send(embed=setup_embed)

            # Enhanced Next Steps with Context
            next_steps = discord.Embed(
                title="📋 Recommended Next Steps",
                color=0x00ff00
            )

            if missing_elements:
                next_steps.description = (
                    "To improve analysis quality:\n"
                    "1️⃣ `!resetanalysis` - Start fresh with suggested improvements\n"
                    "2️⃣ `!help charts` - View chart requirements guide\n"
                    "3️⃣ Upload additional charts with missing timeframes"
                )
            else:
                next_steps.description = (
                    "Analysis complete! Choose your next action:\n"
                    "1️⃣ `!validatesetup <type> <price>` - Validate setup\n"
                    "2️⃣ `!viewanalysis` - View detailed analysis\n"
                    "3️⃣ `!alerts` - Set price alerts"
                )

            await message.channel.send(embed=next_steps)

        except Exception as e:
            logger.error(f"Error sending analysis results: {traceback.format_exc()}")
            await message.channel.send(
                "❌ Error displaying analysis results. Please try again or contact support."
            )

    def _detect_timeframe(self, filename: str) -> str:
        """
        Detect timeframe from filename
        
        Args:
            filename: Name of the uploaded file
            
        Returns:
            Detected timeframe or 'unknown'
        """
        filename_lower = filename.lower()
        for tf in ['15m', '1h', '4h']:
            if tf in filename_lower:
                return tf.upper()
        return 'unknown'

# Global variable for bot instance
trading_bot = None

# Bot event handlers
@bot.event
async def on_ready():
    """Bot startup event"""
    try:
        global trading_bot
        logger.info(f'{bot.user} has connected to Discord!')
        
        # Initialize trading bot if not already initialized
        if trading_bot is None:
            trading_bot = TradingAnalysisBot()
            await trading_bot.start()
            
            status_ok = await trading_bot.check_status()
            if not status_ok:
                logger.error("Bot status check failed")
                await trading_bot.cleanup()  # Ensure cleanup is called
                await bot.close()
                return
                
            logger.info("Bot initialization complete")
    except Exception as e:
        logger.error(f"Error during bot initialization: {traceback.format_exc()}")
        if trading_bot:
            await trading_bot.cleanup()
        await bot.close()

@bot.event
async def on_command_error(ctx, error):
    """Global error handler for bot commands"""
    if isinstance(error, commands.errors.CommandNotFound):
        await ctx.send("❌ Unknown command. Use `!help` to see available commands.")
    elif isinstance(error, commands.errors.MissingPermissions):
        await ctx.send("❌ You don't have permission to use this command.")
    elif isinstance(error, commands.errors.CommandOnCooldown):
        await ctx.send(f"⏳ Command on cooldown. Try again in {error.retry_after:.1f}s")
    else:
        logger.error(f"Command error: {traceback.format_exc()}")
        await ctx.send(f"❌ An error occurred: {str(error)}")

# Bot Commands
@bot.command(name='analyze')
@commands.cooldown(1, 60, commands.BucketType.user)
async def analyze(ctx, pair: str):
    """Start the analysis process for a trading pair"""
    global trading_bot
    if trading_bot is not None:
        await trading_bot.handle_analysis_request(ctx, pair)
    else:
        await ctx.send("❌ Bot is not fully initialized yet. Please try again in a moment.")

@bot.command(name='toggleapi')
@commands.is_owner()  # Only bot owner can use this command
async def toggle_api(ctx):
    """Toggle the API on/off"""
    global trading_bot
    if trading_bot:
        trading_bot.api_enabled = not trading_bot.api_enabled
        status = "enabled" if trading_bot.api_enabled else "disabled"
        await ctx.send(f"🔄 API is now {status}")
    else:
        await ctx.send("❌ Bot not properly initialized")

@bot.event
async def on_message(message):
    """Handle message events, including image uploads"""
    if message.author.bot:
        return

    await bot.process_commands(message)
    
    # Handle image uploads for active analysis sessions
    global trading_bot
    if message.attachments and trading_bot is not None:
        await trading_bot.handle_image_upload(message, message.attachments)

# Main execution
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info("Shutdown signal received")
    if trading_bot is not None:
        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run cleanup
            loop.run_until_complete(trading_bot.cleanup())
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
    sys.exit(0)

if __name__ == "__main__":
    try:
        logger.info("Starting Trading Analysis Bot...")
        
        # Validate environment
        Config.validate()
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start the bot
        logger.info("Connecting to Discord...")
        bot.run(Config.DISCORD_TOKEN)
        
    except Exception as e:
        logger.critical(f"Critical error starting bot: {traceback.format_exc()}")
        sys.exit(1)
    finally:
        if 'trading_bot' in globals() and trading_bot is not None:
            try:
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(trading_bot.cleanup())
            except Exception as e:
                logger.error(f"Error during final cleanup: {str(e)}")
        logger.info("Bot shutdown complete")