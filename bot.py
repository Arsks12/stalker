# Combined imports
import aiohttp
import asyncio
import json
import logging
import logging.handlers
import os
import pytz
import re
import requests
import signal
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from fuzzywuzzy import fuzz
from pathlib import Path
from pyrogram import Client, filters
from pyrogram.errors import FloodWait
from pyrogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup
from typing import Optional, Tuple, Dict, Any, List, Union

# Combined code

# From file: bot/config/settings.py

# Bot configuration
BOT_TOKEN = "7178497534:AAHFmq8svOSR__RB4FnE4ZSOX9I6MXB0094"
API_ID = "29153378"
API_HASH = "4e48910b21d246e4c447edde28a214e6"

# Chat configuration
ALLOWED_CHAT_ID = -1002202897360  # The group where bot is allowed to work
BOT_OWNER_ID = 5896930301  # Bot owner's user ID

# Proxy settings
PROXY = "172.191.74.198:8080"
USE_PROXY = False

def get_proxy_dict(proxy: str = PROXY) -> Dict[str, str]:
    """Create proxy dictionary for requests"""
    if not USE_PROXY:
        return {}
    return {
        "http": f"http://{proxy}",
        "https": f"http://{proxy}"
    }

# Recording settings
RECORDINGS_DIR = "recordings"
MIN_VIABLE_SIZE = 1024 * 1024  # 1MB
STALL_TIMEOUT = 30  # Seconds before considering stream stalled

# UI settings
CHANNELS_PER_PAGE = 10
ERROR_MESSAGE_DELETE_DELAY = 30
RESULT_MESSAGE_DELETE_DELAY = 30

# FFmpeg settings
FFMPEG_HEADERS = (
    'Icy-MetaData: 1\r\n'
    'User-Agent: Lavf53.32.100\r\n'
    'Accept-Encoding: identity\r\n'
    'Connection: Keep-Alive\r\n'
)

# Cache settings
CHANNEL_CACHE_EXPIRY = 24  # Hours
CDN_CACHE_EXPIRY = 4  # Hours

# From file: bot/config/logging_config.py

def setup_logging(log_dir: str = "logs"):
    """Set up logging configuration"""
    
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    
    # Create handlers
    # File handler with daily rotation
    log_file = os.path.join(
        log_dir,
        f"bot_{datetime.now().strftime('%Y-%m-%d')}.log"
    )
    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_file,
        when='midnight',
        interval=1,
        backupCount=7  # Keep logs for 7 days
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set levels for specific loggers
    logging.getLogger('pyrogram').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)
    
    # Create logger for our bot
    bot_logger = logging.getLogger('bot')
    bot_logger.setLevel(logging.DEBUG)
    
    return bot_logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name"""
    return logging.getLogger(f"bot.{name}")

# From file: bot/utils/helpers.py

def print_colored(text: str, color: str) -> str:
    """Convert colored text to Telegram-friendly format"""
    emoji_map = {
        "green": "✅",
        "red": "❌",
        "blue": "ℹ️",
        "yellow": "⚠️",
        "cyan": "🔷"
    }
    return f"{emoji_map.get(color.lower(), '')} {text}"

def format_bytes(size: float) -> str:
    """Format bytes to human readable string"""
    power = 1024
    n = 0
    units = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.1f} {units[n]}B"

def get_ist_time() -> datetime:
    """Get current time in IST timezone"""
    ist = pytz.timezone('Asia/Kolkata')
    return datetime.now(ist)

def ensure_dir(path: str):
    """Ensure directory exists"""
    if not os.path.exists(path):
        os.makedirs(path)

def clean_filename(name: str) -> str:
    """Clean filename of invalid characters"""
    return "".join(c for c in name if c.isalnum() or c in (' ', '-', '_', '[', ']')) 

# From file: bot/stalkers/base.py

logger = logging.getLogger(__name__)

class BaseStalker:
    """Base class for all stalker implementations"""
    
    def __init__(self):
        self.token: Optional[str] = None
        self.session: Optional[requests.Session] = None
        self.base_url: str = ""
        
    async def initialize(self) -> bool:
        """Initialize stalker with required setup"""
        raise NotImplementedError
        
    async def get_token(self) -> Optional[str]:
        """Get authentication token"""
        raise NotImplementedError
        
    async def get_subscription(self) -> bool:
        """Verify subscription status"""
        raise NotImplementedError
        
    async def get_stream_url(self, channel_id: str) -> Optional[str]:
        """Get stream URL for channel"""
        raise NotImplementedError
        
    async def get_channel_list(self) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """Get list of channels and groups"""
        raise NotImplementedError
        
    async def search_channels(self, query: str) -> List[Dict]:
        """Search channels by name"""
        raise NotImplementedError
        
    def cleanup(self):
        """Cleanup resources"""
        if self.session:
            self.session.close()
            self.session = None
        self.token = None

# From file: bot/stalkers/starshare.py


logger = logging.getLogger(__name__)

class StarShareStalker(BaseStalker):
    """StarShare implementation of stalker"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "http://starshare.live:8080"
        self.cache_manager = CacheManager()
        
    async def initialize(self) -> bool:
        """Initialize StarShare session"""
        try:
            self.session = requests.Session()
            cookies = {
                'mac': "00:1A:79:41:48:27",
                'timezone': 'Asia/Calcutta',
                'adid': '00a71c7bffca3e7a1fd6f4f116d49796'
            }
            self.session.cookies.update(cookies)
            if USE_PROXY:
                self.session.proxies = get_proxy_dict(PROXY)
                
            # Get token and verify subscription
            self.token = await self.get_token()
            if not self.token:
                logger.error("Failed to get token")
                return False
                
            if not await self.get_subscription():
                logger.error("Invalid subscription")
                self.token = None
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error initializing StarShare: {e}")
            return False
            
    async def get_token(self) -> Optional[str]:
        """Get StarShare authentication token"""
        try:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (QtEmbedded; U; Linux; C) AppleWebKit/533.3 (KHTML, like Gecko) MAG200 stbapp ver: 2 rev: 250 Safari/533.3',
                'X-User-Agent': 'Model: MAG270; Link: WiFi',
                'Accept': '*/*',
            })
            
            url = f"{self.base_url}/portal.php?type=stb&action=handshake&token=&prehash=13f0848280789b506aeb024ec95eb65cc07b6ce7"
            
            res = self.session.get(url, timeout=10, allow_redirects=False)
            
            if res.status_code == 200:
                return json.loads(res.text)['js']['token']
                
            logger.error(f"Token request failed: {res.status_code}")
            return None
                
        except Exception as e:
            logger.error(f"Error getting token: {e}")
            return None
            
    async def get_subscription(self) -> bool:
        """Verify StarShare subscription"""
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                'User-Agent': 'Mozilla/5.0 (QtEmbedded; U; Linux; C) AppleWebKit/533.3 (KHTML, like Gecko) MAG200 stbapp ver: 2 rev: 250 Safari/533.3',
                'X-User-Agent': 'Model: MAG270; Link: WiFi',
                'Accept': '*/*'
            }
            
            # Get profile first
            profile_url = f"{self.base_url}/portal.php?type=stb&action=get_profile&sn=061856N353978&auth_second_step=1"
            profile_res = self.session.get(profile_url, headers=headers, timeout=10)
            
            if profile_res.status_code != 200:
                logger.error("Failed to fetch profile")
                return False
                
            # Get account info
            account_url = f"{self.base_url}/portal.php?type=account_info&action=get_main_info"
            res = self.session.get(account_url, headers=headers, timeout=10)
            
            if res.status_code == 200 and res.text.strip():
                data = json.loads(res.text)
                logger.info(
                    f"Account info - MAC: {data['js']['mac']}, "
                    f"Expiry: {data['js']['phone']}"
                )
                return True
                
            logger.error("Failed to fetch subscription info")
            return False
                
        except Exception as e:
            logger.error(f"Error checking subscription: {e}")
            return False
            
    async def get_stream_url(self, channel_id: str) -> Optional[str]:
        """Get stream URL for channel"""
        # Check cache first
        cached_url = self.cache_manager.get_cdn_url(channel_id)
        if cached_url:
            return cached_url
            
        try:
            url = f"{self.base_url}/server/load.php"
            params = {
                'type': 'itv',
                'action': 'create_link',
                'cmd': f'ffmpeg http://localhost/ch/{channel_id}',
                'series': '',
                'forced_storage': '0',
                'disable_ad': '0',
                'download': '0',
                'force_ch_link_check': '0',
                'JsHttpRequest': '1-xml'
            }
            
            headers = {
                'Authorization': f'Bearer {self.token}',
                'User-Agent': 'Mozilla/5.0 (QtEmbedded; U; Linux; C) AppleWebKit/533.3 (KHTML, like Gecko) MAG200 stbapp ver: 2 rev: 250 Safari/533.3',
                'X-User-Agent': 'Model: MAG270; Link: WiFi',
                'Accept': '*/*'
            }
            
            res = self.session.get(url, params=params, headers=headers)
            
            if res.status_code == 200:
                data = res.json()
                if 'js' in data and 'cmd' in data['js']:
                    initial_url = data['js']['cmd']
                    stream_url = initial_url[7:] if initial_url.startswith('ffmpeg ') else initial_url

                    # Follow redirect to get CDN URL
                    headers = {
                        'Icy-MetaData': '1',
                        'User-Agent': 'Lavf53.32.100',
                        'Accept-Encoding': 'identity',
                        'Connection': 'Keep-Alive'
                    }

                    res = self.session.get(stream_url, headers=headers, allow_redirects=False)
                    
                    final_url = res.headers.get('Location') if res.status_code == 302 else stream_url
                    if final_url:
                        # Cache the URL
                        self.cache_manager.save_cdn_url(channel_id, final_url)
                        return final_url
                    
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
        return None
        
    async def get_channel_list(self) -> Tuple[Optional[List[Dict]], Optional[Dict]]:
        """Get channel list and groups"""
        # Check cache first
        cached_data = self.cache_manager.get_channels()
        if cached_data:
            return cached_data['channels'], cached_data['groups']
            
        try:
            headers = {
                "Authorization": f"Bearer {self.token}",
                'User-Agent': 'Mozilla/5.0 (QtEmbedded; U; Linux; C) AppleWebKit/533.3 (KHTML, like Gecko) MAG200 stbapp ver: 2 rev: 250 Safari/533.3',
                'X-User-Agent': 'Model: MAG270; Link: WiFi',
                'Accept': '*/*'
            }
            
            # Get genres (channel groups) first
            url_genre = f"{self.base_url}/portal.php?type=itv&action=get_genres"
            res_genre = self.session.get(url_genre, headers=headers, timeout=10)
            
            group_info = {}
            if res_genre.status_code == 200:
                # Process genre data
                genres = json.loads(res_genre.text)['js']
                for group in genres:
                    try:
                        group_info[int(group['id'])] = group['title']
                    except ValueError:
                        continue
                
                # Get channel list
                channel_url = f"{self.base_url}/portal.php?type=itv&action=get_all_channels"
                res = self.session.get(channel_url, headers=headers, timeout=10)
                
                if res.status_code == 200:
                    channels = json.loads(res.text)["js"]["data"]
                    logger.info(f"Found {len(channels)} channels")
                    
                    # Cache the data
                    self.cache_manager.save_channels(channels, group_info)
                    return channels, group_info
                    
            logger.error("Failed to fetch channel data")
            return None, None
                
        except Exception as e:
            logger.error(f"Error getting channel list: {e}")
            return None, None
            
    async def search_channels(self, query: str) -> List[Dict]:
        """Search channels by name"""
        try:
            channels, _ = await self.get_channel_list()
            if not channels:
                return []
                
            query = query.lower()
            return [
                channel for channel in channels
                if query in channel.get('name', '').lower()
            ]
            
        except Exception as e:
            logger.error(f"Error searching channels: {e}")
            return []
            
    def cleanup(self):
        """Cleanup resources"""
        super().cleanup()
        self.cache_manager.cleanup_expired()

# From file: bot/cache/manager.py


logger = logging.getLogger(__name__)

class CacheManager:
    """Unified cache manager for both CDN and channel data"""
    
    def __init__(self):
        self.cdn_cache_file = "cdn_cache.json"
        self.channel_cache_file = "channel_cache.json"
        self.cdn_cache = self._load_cache(self.cdn_cache_file)
        self.channel_cache = self._load_cache(self.channel_cache_file)
        
    def _load_cache(self, cache_file: str) -> dict:
        """Load cache from file"""
        try:
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading cache {cache_file}: {e}")
            return {}
            
    def _save_cache(self, cache_file: str, data: dict):
        """Save cache to file"""
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=4)
            logger.info(f"Cache saved: {cache_file}")
        except Exception as e:
            logger.error(f"Error saving cache {cache_file}: {e}")
            
    def _is_expired(self, timestamp: str, expiry_hours: int) -> bool:
        """Check if cache entry is expired"""
        try:
            cache_time = datetime.fromisoformat(timestamp)
            return datetime.now() - cache_time > timedelta(hours=expiry_hours)
        except Exception:
            return True
            
    # CDN Cache Methods
    def get_cdn_url(self, channel_id: str) -> Optional[str]:
        """Get CDN URL from cache"""
        if channel_id in self.cdn_cache:
            entry = self.cdn_cache[channel_id]
            if not self._is_expired(entry['timestamp'], CDN_CACHE_EXPIRY):
                return entry['url']
        return None
        
    def save_cdn_url(self, channel_id: str, url: str):
        """Save CDN URL to cache"""
        self.cdn_cache[channel_id] = {
            'url': url,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache(self.cdn_cache_file, self.cdn_cache)
        
    def remove_cdn_url(self, channel_id: str):
        """Remove CDN URL from cache"""
        if channel_id in self.cdn_cache:
            del self.cdn_cache[channel_id]
            self._save_cache(self.cdn_cache_file, self.cdn_cache)
            
    # Channel Cache Methods
    def get_channels(self) -> Optional[Dict[str, Any]]:
        """Get channel data from cache"""
        if self.channel_cache:
            if not self._is_expired(
                self.channel_cache.get('timestamp', ''),
                CHANNEL_CACHE_EXPIRY
            ):
                return self.channel_cache
        return None
        
    def save_channels(self, channels: List[Dict], groups: Dict):
        """Save channel data to cache"""
        self.channel_cache = {
            'channels': channels,
            'groups': groups,
            'timestamp': datetime.now().isoformat()
        }
        self._save_cache(self.channel_cache_file, self.channel_cache)
        
    def clear_all(self):
        """Clear all caches"""
        self.cdn_cache = {}
        self.channel_cache = {}
        if os.path.exists(self.cdn_cache_file):
            os.remove(self.cdn_cache_file)
        if os.path.exists(self.channel_cache_file):
            os.remove(self.channel_cache_file)
            
    def cleanup_expired(self):
        """Remove expired entries from caches"""
        # Clean CDN cache
        expired_cdn = [
            channel_id for channel_id, entry in self.cdn_cache.items()
            if self._is_expired(entry['timestamp'], CDN_CACHE_EXPIRY)
        ]
        for channel_id in expired_cdn:
            del self.cdn_cache[channel_id]
            
        # Clean channel cache
        if self.channel_cache and self._is_expired(
            self.channel_cache.get('timestamp', ''),
            CHANNEL_CACHE_EXPIRY
        ):
            self.channel_cache = {}
            
        # Save cleaned caches
        self._save_cache(self.cdn_cache_file, self.cdn_cache)
        self._save_cache(self.channel_cache_file, self.channel_cache)
        
    async def get_channel_list(self, session: aiohttp.ClientSession, base_url: str, token: str) -> Tuple[List[Dict], Dict]:
        """Get channel list from API"""
        try:
            async with session.get(
                f"{base_url}/channels",
                headers={"Authorization": f"Bearer {token}"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    channels = data.get('channels', [])
                    groups = data.get('groups', {})
                    self.save_channels(channels, groups)
                    return channels, groups
                else:
                    logger.error(f"Failed to get channels: {response.status}")
                    return [], {}
        except Exception as e:
            logger.error(f"Error getting channel list: {e}")
            return [], {}
            
    async def search_channels(self, channels: List[Dict], query: str) -> List[Dict]:
        """Search channels using fuzzy matching"""
        try:
            results = []
            for channel in channels:
                name = channel.get('name', '').lower()
                score = fuzz.partial_ratio(query.lower(), name)
                if score > 60:  # Adjust threshold as needed
                    results.append(channel)
            return sorted(
                results,
                key=lambda x: fuzz.partial_ratio(query.lower(), x.get('name', '').lower()),
                reverse=True
            )
        except Exception as e:
            logger.error(f"Error searching channels: {e}")
            return []
            
    async def get_stream_url(self, session: aiohttp.ClientSession, base_url: str, channel_id: str, token: str) -> Optional[str]:
        """Get stream URL from API"""
        try:
            # Check cache first
            cached_url = self.get_cdn_url(channel_id)
            if cached_url:
                return cached_url
                
            # Get from API if not in cache
            async with session.get(
                f"{base_url}/stream/{channel_id}",
                headers={"Authorization": f"Bearer {token}"}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    url = data.get('url')
                    if url:
                        self.save_cdn_url(channel_id, url)
                        return url
                else:
                    logger.error(f"Failed to get stream URL: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Error getting stream URL: {e}")
            return None

# From file: bot/utils/state.py

@dataclass
class UserState:
    """User state with recording information"""
    
    # Stalker instance
    stalker: Optional[StarShareStalker] = None
    
    # Channel data
    channels: List[Dict] = field(default_factory=list)
    current_page: int = 0
    selected_channel: Optional[Dict] = None
    
    # Message tracking
    last_message_id: Optional[int] = None
    original_message_id: Optional[int] = None
    
    # Recording state
    recording_in_progress: bool = False
    current_process: Optional[asyncio.subprocess.Process] = None
    recording_start_time: Optional[datetime] = None
    
    # File paths
    output_file: Optional[str] = None
    thumbnail_file: Optional[str] = None
    
    def __post_init__(self):
        """Initialize stalker if not provided"""
        if self.stalker is None:
            self.stalker = StarShareStalker()
    
    async def initialize(self) -> bool:
        """Initialize stalker session"""
        return await self.stalker.initialize()
    
    async def close(self):
        """Close stalker and cleanup"""
        if self.stalker:
            self.stalker.cleanup()
            self.stalker = None
    
    def reset(self):
        """Reset state to initial values"""
        self.channels = []
        self.current_page = 0
        self.selected_channel = None
        self.last_message_id = None
        self.original_message_id = None
        self.recording_in_progress = False
        self.current_process = None
        self.recording_start_time = None
        self.output_file = None
        self.thumbnail_file = None
        
    def start_recording(self):
        """Mark recording as started"""
        self.recording_in_progress = True
        self.recording_start_time = datetime.now()
        
    def stop_recording(self):
        """Mark recording as stopped"""
        self.recording_in_progress = False
        self.current_process = None
        self.recording_start_time = None
        
    @property
    def recording_duration(self) -> Optional[int]:
        """Get current recording duration in seconds"""
        if self.recording_start_time and self.recording_in_progress:
            return int((datetime.now() - self.recording_start_time).total_seconds())
        return None

user_states: Dict[int, UserState] = {} 

# From file: bot/utils/keyboard.py

class KeyboardManager:
    """Unified manager for all keyboard operations"""
    
    @staticmethod
    def create_channel_keyboard(channels: List[Dict], current_page: int) -> InlineKeyboardMarkup:
        """Create paginated keyboard for channel selection"""
        keyboard = []
        start_idx = current_page * CHANNELS_PER_PAGE
        end_idx = start_idx + CHANNELS_PER_PAGE
        
        # Add channel buttons
        for idx, channel in enumerate(channels[start_idx:end_idx], start=start_idx):
            keyboard.append([
                InlineKeyboardButton(
                    channel.get('name', 'Unknown Channel'),
                    callback_data=f"select_{idx}"
                )
            ])
        
        # Add navigation buttons
        nav_buttons = []
        if current_page > 0:
            nav_buttons.append(
                InlineKeyboardButton("⬅️ Previous", callback_data="prev_page")
            )
        if end_idx < len(channels):
            nav_buttons.append(
                InlineKeyboardButton("Next ➡️", callback_data="next_page")
            )
        
        if nav_buttons:
            keyboard.append(nav_buttons)
        
        return InlineKeyboardMarkup(keyboard)
        
    @staticmethod
    def create_duration_keyboard() -> InlineKeyboardMarkup:
        """Create keyboard with duration options"""
        durations = [
            ("30 Seconds", 30),
            ("1 Minute", 60),
            ("5 Minutes", 300),
            ("10 Minutes", 600),
            ("30 Minutes", 1800),
            ("1 Hour", 3600),
            ("2 Hours", 7200)
        ]
        
        keyboard = []
        # Create two buttons per row
        for i in range(0, len(durations), 2):
            row = []
            for label, seconds in durations[i:i+2]:
                row.append(
                    InlineKeyboardButton(
                        label,
                        callback_data=f"duration_{seconds}"
                    )
                )
            keyboard.append(row)
        
        return InlineKeyboardMarkup(keyboard)
        
    @staticmethod
    def parse_callback_data(data: str) -> tuple:
        """Parse callback data into action and value"""
        if data.startswith("select_"):
            return "select", int(data.split("_")[1])
        elif data.startswith("duration_"):
            return "duration", int(data.split("_")[1])
        elif data in ["prev_page", "next_page"]:
            return "navigation", data
        return "unknown", None 

# From file: bot/utils/message.py

logger = logging.getLogger(__name__)

class MessageHandler:
    """Unified handler for all message operations"""
    
    def __init__(self, client: Client):
        self.client = client
        
    async def edit_message(
        self,
        message: Message,
        text: str,
        reply_markup: Optional[Union[InlineKeyboardMarkup, ReplyKeyboardMarkup]] = None
    ) -> Optional[Message]:
        """Safely edit message with flood wait handling"""
        try:
            return await message.edit_text(text, reply_markup=reply_markup)
        except FloodWait as e:
            await asyncio.sleep(e.value)
            try:
                return await message.edit_text(text, reply_markup=reply_markup)
            except Exception as e:
                logger.error(f"Error editing message after flood wait: {e}")
                return None
        except Exception as e:
            logger.error(f"Error editing message: {e}")
            return None
            
    async def send_message(
        self,
        chat_id: int,
        text: str,
        reply_to_message_id: Optional[int] = None,
        reply_markup: Optional[Union[InlineKeyboardMarkup, ReplyKeyboardMarkup]] = None
    ) -> Optional[Message]:
        """Safely send message with flood wait handling"""
        try:
            return await self.client.send_message(
                chat_id=chat_id,
                text=text,
                reply_to_message_id=reply_to_message_id,
                reply_markup=reply_markup
            )
        except FloodWait as e:
            await asyncio.sleep(e.value)
            try:
                return await self.client.send_message(
                    chat_id=chat_id,
                    text=text,
                    reply_to_message_id=reply_to_message_id,
                    reply_markup=reply_markup
                )
            except Exception as e:
                logger.error(f"Error sending message after flood wait: {e}")
                return None
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            return None
            
    async def get_username(self, user_id: int) -> Optional[str]:
        """Safely fetch username for a given user ID"""
        try:
            user = await self.client.get_users(user_id)
            return user.username or user.first_name
        except FloodWait as e:
            await asyncio.sleep(e.value)
            try:
                user = await self.client.get_users(user_id)
                return user.username or user.first_name
            except Exception as e:
                logger.error(f"Error getting username after flood wait: {e}")
                return None
        except Exception as e:
            logger.error(f"Error getting username: {e}")
            return None
            
    async def delete_after_delay(self, message: Message, delay_seconds: int = 30):
        """Delete a message after specified delay"""
        try:
            await asyncio.sleep(delay_seconds)
            await message.delete()
        except Exception as e:
            logger.error(f"Error deleting message: {e}")
            
    async def update_status(
        self,
        message: Message,
        status: str,
        progress: Optional[float] = None,
        size: Optional[float] = None,
        error: Optional[str] = None,
        duration: Optional[str] = None,
        resolution: Optional[str] = None
    ) -> Optional[Message]:
        """Update status message with current progress"""
        try:
            text = f"🎥 Status: {status}\n"
            
            if duration is not None:
                text += f"⏱ Recording: {duration}\n"
                
            if progress is not None:
                text += f"⏳ Progress: {progress:.1f}%\n"
                
            if size is not None:
                text += f"📦 Size: {size:.1f} MB\n"
                
            if resolution is not None:
                text += f"📺 Resolution: {resolution}\n"
                
            if error:
                text += f"❌ Error: {error}\n"
                
            return await self.edit_message(message, text)
            
        except Exception as e:
            logger.error(f"Error updating status message: {e}")
            return None
            
    async def send_error(
        self,
        chat_id: int,
        error: str,
        reply_to_message_id: Optional[int] = None,
        delete_after: Optional[int] = 30
    ) -> None:
        """Send error message and optionally delete it after delay"""
        try:
            error_msg = await self.send_message(
                chat_id,
                f"❌ Error: {error}",
                reply_to_message_id
            )
            if error_msg and delete_after:
                asyncio.create_task(self.delete_after_delay(error_msg, delete_after))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")
            
    def format_duration(self, seconds: int) -> str:
        """Format duration in seconds to human readable string"""
        return str(timedelta(seconds=seconds)).split('.')[0]
        
    def format_bytes(self, size: float) -> str:
        """Format bytes to human readable string"""
        power = 1024
        n = 0
        units = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
        while size > power:
            size /= power
            n += 1
        return f"{size:.1f} {units[n]}B"

# From file: bot/utils/video.py

logger = logging.getLogger(__name__)

async def safe_edit_message(message, text):
    """Safely edit message with flood wait handling"""
    try:
        await message.edit_text(text)
    except FloodWait as e:
        await asyncio.sleep(e.value)
        try:
            await message.edit_text(text)
        except Exception as e:
            logger.error(f"Error editing message after flood wait: {e}")
    except Exception as e:
        logger.error(f"Error editing message: {e}")

async def get_video_metadata(file_path: str, max_retries: int = 3) -> Dict[str, Any]:
    """Extract video metadata with retries"""
    metadata = {"duration": None, "width": None, "height": None}
    retry_delay = 1

    # Get duration
    for retry in range(max_retries):
        try:
            duration_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', file_path]
            duration_process = await asyncio.create_subprocess_exec(*duration_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            duration_output, _ = await duration_process.communicate()
            duration_str = duration_output.decode().strip()
            if duration_str and duration_str != 'N/A':
                metadata["duration"] = int(float(duration_str))
                break
        except Exception as e:
            logger.warning(f"Duration check attempt {retry + 1} failed: {e}")
            if retry < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    # Get resolution
    retry_delay = 1
    for retry in range(max_retries):
        try:
            probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'json', file_path]
            probe_process = await asyncio.create_subprocess_exec(*probe_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            probe_out, _ = await probe_process.communicate()
            resolution_data = json.loads(probe_out.decode())
            metadata["width"] = resolution_data['streams'][0]['width']
            metadata["height"] = resolution_data['streams'][0]['height']
            break
        except Exception as e:
            logger.warning(f"Resolution check attempt {retry + 1} failed: {e}")
            if retry < max_retries - 1:
                await asyncio.sleep(retry_delay)
                retry_delay *= 2

    return metadata

def format_bytes(size):
    """Format bytes to human readable string"""
    power = 1024
    n = 0
    units = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while size > power:
        size /= power
        n += 1
    return f"{size:.1f} {units[n]}B"

def get_user_mention(message) -> str:
    """Get user mention from message"""
    if message.reply_to_message and message.reply_to_message.from_user:
        user = message.reply_to_message.from_user
        return f"@{user.username}" if user.username else user.first_name
    return "Unknown User"

async def generate_thumbnail(input_file: str, output_file: str, width: int, height: int) -> Optional[str]:
    """Generate video thumbnail"""
    try:
        thumb_cmd = ['ffmpeg', '-ss', '00:00:01', '-i', input_file, '-vframes', '1', '-vf', f'scale={width}:{height}', '-y', output_file]
        thumb_process = await asyncio.create_subprocess_exec(*thumb_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        await thumb_process.communicate()
        return output_file if os.path.exists(output_file) else None
    except Exception as e:
        logger.warning(f"Error generating thumbnail: {e}")
        return None

async def process_and_upload_video(client, message, input_file: str, output_file: str, safe_name: str, channel_name: str, elapsed_time: float, is_partial: bool = False):
    """Process and upload video file"""
    try:
        # Get video metadata
        metadata = await get_video_metadata(output_file)
        width = metadata.get("width", 1920)
        height = metadata.get("height", 1080)
        duration = metadata.get("duration", int(elapsed_time))

        # Generate thumbnail
        thumbnail_file = await generate_thumbnail(
            output_file,
            os.path.join(os.path.dirname(output_file), f"{safe_name}_thumb.jpg"),
            width,
            height
        )

        # Upload video
        await upload_video(
            client,
            message.chat.id,
            str(Path(output_file).absolute()),
            str(Path(thumbnail_file).absolute()) if thumbnail_file else None,
            message,
            width,
            height,
            channel_name,
            duration,
            message.id,
            is_partial=is_partial
        )

        # Clean up files
        for file in [output_file, thumbnail_file]:
            if file and os.path.exists(file):
                try:
                    os.remove(file)
                except Exception as e:
                    logger.error(f"Error cleaning up file {file}: {e}")

    except Exception as e:
        logger.error(f"Processing error: {e}")
        await safe_edit_message(message, f"Processing error: {str(e)}")

async def update_status_message(message, recording_status: Optional[Dict] = None, upload_status: Optional[Dict] = None):
    """Update status message with recording and upload progress"""
    try:
        if recording_status:
            status_text = (
                f"🎥 Status: Recording in progress\n"
                f"⏱ Recording: {recording_status['elapsed']}/{recording_status['duration']}\n"
                f"📦 Size: {recording_status['size']:.1f} MB\n"
                f"👤 Requested by: {recording_status['user']}"
            )
        else:
            status_text = ""

        if upload_status:
            upload_text = (
                f"📤 Uploading:\n"
                f"{upload_status['filename']}\n"
                f"⏳ Progress: {upload_status['progress']:.1f}%\n"
                f"💫 Speed: {upload_status['speed']}/s\n"
                f"{'▰' * int(upload_status['progress'] / 10)}{'▱' * (10 - int(upload_status['progress'] / 10))}"
            )
            status_text = f"{status_text}\n\n{upload_text}" if status_text else upload_text

        await safe_edit_message(message, status_text)
    except Exception as e:
        logger.warning(f"Error updating status: {e}")

async def upload_video(client, chat_id, output_file, thumb_file, status_msg, width, height, channel_name, duration, reply_to_message_id=None, is_partial=False):
    """Handle video upload with progress tracking"""
    try:
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")

        file_size = os.path.getsize(output_file)
        if file_size == 0:
            raise ValueError("Output file is empty")

        # Get IST timezone
        ist = pytz.timezone('Asia/Kolkata')
        ist_time = datetime.now(ist)
        
        # Extract the base filename without the "to" part
        filename = os.path.basename(output_file)
        if " to" in filename:
            base_filename = filename.split(" to")[0]
            # Add the IST formatted end time
            filename = f"{base_filename} to {ist_time.strftime('%I-%M-%S %p')}].mkv"

        last_update_time = time.time()
        start_time = time.time()

        async def progress(current, total):
            nonlocal last_update_time, start_time
            try:
                current_time = time.time()
                if current_time - last_update_time >= 5 or current == total:
                    try:
                        # Safely convert to float and handle None values
                        current_float = float(current if current is not None else 0)
                        total_float = float(total if total is not None else 1)  # Avoid division by zero
                        
                        progress_pct = (current_float * 100.0) / total_float if total_float > 0 else 0
                        elapsed_time = max(0.1, current_time - start_time)  # Avoid division by zero
                        speed = current_float / elapsed_time if elapsed_time > 0 else 0
                    except (TypeError, ValueError):
                        # If conversion fails, use safe defaults
                        progress_pct = 0
                        speed = 0
                    
                    await update_status_message(
                        status_msg,
                        upload_status={
                            'filename': filename,
                            'progress': progress_pct,
                            'speed': format_bytes(speed),
                        }
                    )
                    last_update_time = current_time
            except Exception as e:
                logger.error(f"Progress calculation error: {e}")
                # Continue upload even if progress update fails
                pass

        # Upload with retries
        max_retries = 3
        retry_delay = 5
        for retry in range(max_retries):
            try:
                await client.send_video(
                    chat_id=chat_id,
                    video=output_file,
                    caption=f"`{filename}`",
                    duration=duration,
                    supports_streaming=True,
                    thumb=thumb_file,
                    file_name=filename,
                    progress=progress,
                    width=width,
                    height=height,
                    reply_to_message_id=reply_to_message_id
                )
                # Only clear status for complete uploads, not partial ones
                if not is_partial:
                    await safe_edit_message(status_msg, "✅ Upload complete")
                    await asyncio.sleep(3)
                    await status_msg.delete()
                break
            except Exception as e:
                if retry < max_retries - 1:
                    logger.warning(f"Upload attempt {retry + 1} failed: {e}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise

    except Exception as e:
        logger.error(f"Upload error: {e}")
        if not is_partial:  # Only show error for complete uploads
            error_msg = (
                "❌ Recording Interrupted\n\n"
                f"Error: {str(e)}\n"
                "Partial upload may be available above.\n"
                f"👤 Requested by: {get_user_mention(status_msg)}"
            )
            await safe_edit_message(status_msg, error_msg)

async def record_stream(stream_url: str, duration: int, message, channel_name: str, channel_id: str = None, base_url: str = None, test_interrupt: bool = True) -> Optional[Tuple[str, str, int, int, int]]:
    """Record stream using FFmpeg with advanced error handling and recovery"""
    try:
        logger.info(f"Starting recording for channel: {channel_name}")
        client = message._client
        main_dir = Path("recordings")
        main_dir.mkdir(exist_ok=True)
        
        ist = pytz.timezone('Asia/Kolkata')
        start_time = datetime.now(ist)
        safe_name = re.sub(r'[<>:"/\\|?*]', '', channel_name)
        temp_filename = f"[Thor-RF] {safe_name} [{start_time.strftime('%d-%m-%y')}] [{start_time.strftime('%I-%M-%S %p')} to"

        async def try_record(current_url: str, temp_filename: str, elapsed_total: float = 0) -> Tuple[bool, Optional[str], float]:
            output_file = os.path.join(main_dir, temp_filename + "_temp.mkv")
            headers = (
                'Icy-MetaData: 1\r\n'
                'User-Agent: Lavf53.32.100\r\n'
                'Accept-Encoding: identity\r\n'
                'Connection: Keep-Alive\r\n'
            )
            
            cmd = [
                'ffmpeg', '-hide_banner',
                '-reconnect', '1',
                '-reconnect_streamed', '1',
                '-reconnect_delay_max', '20',
                '-headers', headers,
                '-i', current_url,
                '-c:v', 'copy', '-c:a', 'copy', '-c:s', 'copy',
                '-map', '0',
                '-y',
                '-loglevel', 'warning',
                '-analyzeduration', '20M',
                '-probesize', '20M',
                output_file
            ]

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            start_time = time.time()
            last_size = 0
            stall_start = None
            last_update_time = 0
            min_viable_size = 1024 * 1024  # 1MB
            last_test_interrupt = 0  # Track when we last did a test interrupt

            try:
                while process.returncode is None:
                    current_time = time.time()
                    elapsed_time = float(current_time - start_time)
                    total_elapsed = elapsed_total + elapsed_time

                    # Test interruption every 30 seconds if enabled
                    if test_interrupt and elapsed_time - last_test_interrupt >= 30:
                        logger.info("Test interruption triggered")
                        await safe_edit_message(message, "Test interruption - simulating stream error")
                        last_test_interrupt = elapsed_time
                        
                        process.send_signal(signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            process.kill()
                            await process.wait()
                        
                        if os.path.exists(output_file) and os.path.getsize(output_file) >= min_viable_size:
                            await asyncio.sleep(2)  # Ensure file is fully written
                            
                            end_time = datetime.now(ist)
                            temp_final_filename = f"{temp_filename} {end_time.strftime('%I-%M-%S %p')}].mkv"
                            temp_final_output_file = os.path.join(main_dir, temp_final_filename)
                            os.rename(output_file, temp_final_output_file)
                            
                            # Start background processing and upload
                            asyncio.create_task(process_and_upload_video(
                                client,
                                message,
                                output_file,
                                temp_final_output_file,
                                safe_name,
                                channel_name,
                                elapsed_time,
                                is_partial=True
                            ))
                            logger.info("Started background processing and upload")
                            
                        return False, "Continuing with test interruption", total_elapsed

                    if total_elapsed >= float(duration):
                        logger.info("Recording duration reached")
                        process.send_signal(signal.SIGTERM)
                        try:
                            await asyncio.wait_for(process.wait(), timeout=5.0)
                        except asyncio.TimeoutError:
                            process.kill()
                        return True, None, total_elapsed

                    if os.path.exists(output_file):
                        current_size = os.path.getsize(output_file)
                        size = current_size / (1024 * 1024)

                        # Check for stalled recording
                        if current_size == last_size:
                            if stall_start is None:
                                stall_start = current_time
                            elif current_time - stall_start >= 30:  # stall timeout
                                await safe_edit_message(message, "Recording stalled - attempting to restart with fresh URL")
                                process.kill()
                                return False, "Recording stalled", total_elapsed
                        else:
                            stall_start = None
                            last_size = current_size

                        # Update status message
                        if current_time - last_update_time >= 5:
                            await update_status_message(
                                message,
                                recording_status={
                                    'elapsed': str(timedelta(seconds=int(total_elapsed))),
                                    'duration': str(timedelta(seconds=duration)),
                                    'size': size,
                                    'user': get_user_mention(message)
                                }
                            )
                            last_update_time = current_time

                    await asyncio.sleep(1)

                # Process ended, check for errors
                stdout, stderr = await process.communicate()
                error_text = stderr.decode().lower()
                
                if any(err in error_text for err in ['404', 'connection refused', 'timeout', 'end of file']):
                    return False, error_text, total_elapsed

                if os.path.exists(output_file) and os.path.getsize(output_file) >= min_viable_size:
                    return True, None, total_elapsed
                    
                return False, "Recording failed", total_elapsed

            except asyncio.CancelledError:
                logger.info("Recording interrupted")
                process.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()

                if os.path.exists(output_file) and os.path.getsize(output_file) >= min_viable_size:
                    await asyncio.sleep(2)
                    return False, "Recording interrupted", total_elapsed
                return False, "Recording interrupted", total_elapsed

            except Exception as e:
                logger.error(f"Recording error: {e}")
                process.kill()
                await process.wait()
                return False, str(e), total_elapsed

        # Start recording loop
        elapsed_total = 0
        current_url = stream_url
        fresh_url_tried = False  # Track if we've already tried getting a fresh URL

        while elapsed_total < duration:
            success, error, elapsed = await try_record(current_url, temp_filename, elapsed_total)
            elapsed_total = elapsed

            if not success:
                # If it's a test interruption or stall, just continue with current URL
                if error in ["Continuing with test interruption", "Recording stalled"]:
                    continue

                # For real failures, try to get fresh URL only once
                if channel_id and base_url and not fresh_url_tried:
                    await safe_edit_message(message, "Stream failed, attempting to get fresh URL...")
                    try:
                        stalker = StarShareStalker()
                        if await stalker.initialize():
                            fresh_url = await stalker.get_stream_url(channel_id)
                            if fresh_url:
                                await safe_edit_message(message, "Retrying with fresh URL...")
                                current_url = fresh_url
                                fresh_url_tried = True  # Mark that we have tried a fresh URL
                                continue
                    except Exception as e:
                        logger.error(f"Error getting fresh URL: {e}")

                # If we have already tried a fresh URL or could not get one, stop recording
                await safe_edit_message(message, f"Recording failed: {error}")
                return None, None, None, None, None

        # Process final recording
        output_file = os.path.join(main_dir, temp_filename + "_temp.mkv")
        if output_file and os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            end_time = datetime.now(ist)
            final_filename = f"{temp_filename} {end_time.strftime('%I-%M-%S %p')}].mkv"
            final_output_file = os.path.join(main_dir, final_filename)
            os.rename(output_file, final_output_file)

            metadata = await get_video_metadata(final_output_file)
            width = metadata.get("width", 1920)
            height = metadata.get("height", 1080)
            duration = metadata.get("duration", int((end_time - start_time).total_seconds()))

            thumbnail_file = await generate_thumbnail(
                final_output_file,
                os.path.join(main_dir, f"{safe_name}_thumb.jpg"),
                width,
                height
            )

            return final_output_file, thumbnail_file, width, height, duration

        return None, None, None, None, None

    except Exception as e:
        logger.exception(f"Recording error: {e}")
        await safe_edit_message(message, f"Recording error: {str(e)}")
        return None, None, None, None, None

# From file: bot/utils/cleanup.py

logger = logging.getLogger(__name__)

class CleanupManager:
    """Unified manager for all cleanup operations"""
    
    def __init__(self, app: Client, user_states: Dict[int, UserState], allowed_chat_id: int):
        self.app = app
        self.user_states = user_states
        self.allowed_chat_id = allowed_chat_id
        self.message_handler = MessageHandler(app)
        
    async def cleanup_recording(self, user_id: int, state: UserState):
        """Clean up a specific recording"""
        try:
            if state.recording_in_progress:
                logger.info(f"Cleaning up recording for user {user_id}")
                if hasattr(state, 'current_process') and state.current_process:
                    state.current_process.send_signal(signal.SIGTERM)
                    
                    # Update status message
                    if hasattr(state, 'last_message_id'):
                        try:
                            message = await self.app.get_messages(
                                self.allowed_chat_id,
                                state.last_message_id
                            )
                            if message and message.reply_to_message:
                                username = message.reply_to_message.from_user.username
                                user_mention = f"@{username}" if username else message.reply_to_message.from_user.first_name
                                await self.message_handler.edit_message(
                                    message,
                                    "⚠️ Recording Interrupted\n\n"
                                    "Bot was stopped manually.\n"
                                    "Partial recording may be available above.\n"
                                    f"👤 Requested by: {user_mention}"
                                )
                        except Exception as e:
                            logger.error(f"Error updating status message during cleanup: {e}")
                            
        except Exception as e:
            logger.error(f"Error cleaning up recording for user {user_id}: {e}")
            
    async def cleanup_files(self, output_file: Optional[str] = None, thumb_file: Optional[str] = None):
        """Clean up recording files"""
        try:
            for file in [output_file, thumb_file]:
                if file and os.path.exists(file):
                    try:
                        os.remove(file)
                        logger.info(f"Removed file: {file}")
                    except Exception as e:
                        logger.error(f"Error removing file {file}: {e}")
        except Exception as e:
            logger.error(f"Error cleaning up files: {e}")
            
    async def cleanup_all(self):
        """Clean up all active recordings and resources"""
        try:
            # Clean up all active recordings
            for user_id, state in self.user_states.items():
                await self.cleanup_recording(user_id, state)
                
            # Wait for cleanup to complete
            await asyncio.sleep(2)
            
            # Stop the client
            logger.info("Stopping Pyrogram client...")
            await self.app.stop()
            
            # Stop the event loop
            logger.info("Stopping event loop...")
            loop = asyncio.get_event_loop()
            loop.stop()
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            logger.info("Cleanup completed")
            
    async def handle_shutdown(self, signum: int):
        """Handle graceful shutdown on signals"""
        logger.info(f"Received signal {signum}, performing cleanup...")
        await self.cleanup_all() 

# From file: bot/__main__.py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Initialize bot
app = Client(
    "stream_recorder_bot",
    api_id=API_ID,
    api_hash=API_HASH,
    bot_token=BOT_TOKEN
)

# Initialize handlers and managers
message_handler = MessageHandler(app)
keyboard_manager = KeyboardManager()
cleanup_manager = CleanupManager(app, user_states, ALLOWED_CHAT_ID)
cache_manager = CacheManager()

async def cleanup_user_sessions():
    """Cleanup all user sessions"""
    for user_id, state in user_states.items():
        try:
            await state.close()
        except Exception as e:
            logger.error(f"Error closing session for user {user_id}: {e}")
    user_states.clear()

@app.on_message(filters.command(["channel", "search"]))
async def channel_command(client: Client, message: Message):
    """Handle /channel and /search commands"""
    # Allow in specified group chat OR for bot owner
    if message.chat.id != ALLOWED_CHAT_ID and message.from_user.id != BOT_OWNER_ID:
        return
        
    try:
        # Get search query from command
        if len(message.command) < 2:
            error_msg = await message_handler.send_message(
                message.chat.id,
                "Please provide a search term\nExample: `/channel sports`",
                message.id
            )
            asyncio.create_task(message_handler.delete_after_delay(error_msg))
            await message.delete()
            return
            
        query = " ".join(message.command[1:])
        user_id = message.from_user.id
        
        # Initialize user state
        if user_id not in user_states:
            user_states[user_id] = UserState()
        state = user_states[user_id]
        state.original_message_id = message.id
        state.recording_in_progress = False

        status_msg = await message_handler.send_message(
            message.chat.id,
            "🔍 Loading channels...",
            message.id
        )
        
        # Initialize stalker if needed
        if not await state.initialize():
            await message_handler.edit_message(status_msg, "❌ Failed to initialize stalker")
            await message.delete()
            return
        
        # Get channel list from cache or API
        channels, groups = await state.stalker.get_channel_list()
        if not channels:
            await message_handler.edit_message(status_msg, "❌ Failed to get channels")
            await message.delete()
            return

        # Search channels
        state.channels = await state.stalker.search_channels(query)
        
        if not state.channels:
            await message_handler.edit_message(status_msg, "❌ No channels found")
            await message.delete()
            return
            
        # Show results
        username = await message_handler.get_username(message.from_user.id)
        display_name = username or "Unknown User"
        
        result_message = await message_handler.edit_message(
            status_msg,
            f"🔍 Found {len(state.channels)} channels matching '{query}'\n"
            f"👤 Requested by: {display_name}",
            reply_markup=keyboard_manager.create_channel_keyboard(state.channels, 0)
        )
        
        if not result_message:
            await message_handler.send_error(
                message.chat.id,
                "Failed to update message",
                message.id
            )
            await message.delete()
            return
        
        state.last_message_id = result_message.id
        
        # Auto-delete after timeout
        async def delete_messages():
            await asyncio.sleep(30)
            try:
                await result_message.delete()
                if not state.recording_in_progress:
                    await message.delete()
                    if user_id in user_states:
                        del user_states[user_id]
            except Exception as e:
                logger.error(f"Error deleting messages: {e}")
        
        asyncio.create_task(delete_messages())
        
    except Exception as e:
        logger.exception(f"Error in channel command for user {user_id}")
        await message_handler.send_error(
            message.chat.id,
            str(e),
            message.id
        )
        await message.delete()

@app.on_callback_query()
async def handle_callback(client: Client, callback_query: CallbackQuery):
    """Handle callback queries from inline keyboard"""
    # Allow in specified group chat OR for bot owner
    if callback_query.message.chat.id != ALLOWED_CHAT_ID and callback_query.from_user.id != BOT_OWNER_ID:
        return
        
    try:
        user_id = callback_query.from_user.id
        message_id = callback_query.message.id
        
        # Get the original requester's state
        original_user_id = None
        for uid, state in user_states.items():
            if state.last_message_id == message_id:
                original_user_id = uid
                break
                
        if original_user_id is None:
            user_states[user_id] = UserState()
            original_user_id = user_id
            
        state = user_states[original_user_id]
        data = callback_query.data

        await callback_query.answer()
        
        if data == "prev_page" or data == "next_page":
            if data == "prev_page":
                state.current_page -= 1
            else:
                state.current_page += 1
                
            result_message = await message_handler.edit_message(
                callback_query.message,
                f"🔍 Channel List\n👤 Requested by: {callback_query.from_user.first_name}",
                reply_markup=keyboard_manager.create_channel_keyboard(state.channels, state.current_page)
            )
            
            if not result_message:
                await message_handler.send_error(
                    callback_query.message.chat.id,
                    "Failed to update message"
                )
                return
            
            async def delete_result_message():
                await asyncio.sleep(30)
                try:
                    await result_message.delete()
                    if user_id in user_states:
                        del user_states[user_id]
                except Exception as e:
                    logger.error(f"Error deleting result message: {e}")
            
            asyncio.create_task(delete_result_message())
            
        elif data.startswith("select_"):
            try:
                if not state.channels:
                    await callback_query.answer("Please search for channels again", show_alert=True)
                    await callback_query.message.delete()
                    return
                    
                idx = int(data.split("_")[1])
                if idx >= len(state.channels):
                    await callback_query.answer("Channel list has been refreshed. Please search again", show_alert=True)
                    await callback_query.message.delete()
                    return
                    
                state.selected_channel = state.channels[idx]
                
                result_message = await message_handler.edit_message(
                    callback_query.message,
                    f"Selected: {state.selected_channel.get('name')}\n"
                    f"Select recording duration:\n"
                    f"👤 Requested by: {callback_query.from_user.first_name}",
                    reply_markup=keyboard_manager.create_duration_keyboard()
                )
                
                if not result_message:
                    await message_handler.send_error(
                        callback_query.message.chat.id,
                        "Failed to update message"
                    )
                    return
                
                asyncio.create_task(message_handler.delete_after_delay(result_message))
                
            except (IndexError, ValueError) as e:
                logger.error(f"Invalid channel selection for user {user_id}: {e}")
                await callback_query.answer("Please search for channels again", show_alert=True)
                
        elif data.startswith("duration_"):
            try:
                state.recording_in_progress = True
                
                duration = int(data.split("_")[1])
                channel = state.selected_channel
                channel_name = channel.get('name', 'Unknown')
                cmd = channel.get('cmd', '')
                channel_id = cmd.split('/')[-1] if cmd else None
                
                await callback_query.message.delete()
                
                if not channel_id:
                    await message_handler.send_error(
                        callback_query.message.chat.id,
                        "Invalid channel ID"
                    )
                    return
                    
                username = await message_handler.get_username(callback_query.from_user.id)
                display_name = username or "Unknown User"
                
                status_msg = await message_handler.send_message(
                    callback_query.message.chat.id,
                    f"🎥 Preparing to record: {channel_name}\n"
                    f"⏱ Duration: {duration}s\n"
                    f"👤 Requested by: {display_name}",
                    state.original_message_id
                )

                # Get stream URL from stalker
                stream_url = await state.stalker.get_stream_url(channel_id)
                
                if not stream_url:
                    state.recording_in_progress = False
                    return

                output_file, thumb_file, width, height, actual_duration = await record_stream(
                    stream_url,
                    duration,
                    status_msg,
                    channel_name,
                    channel_id,
                    state.stalker.base_url
                )
                
                if output_file and os.path.exists(output_file):
                    try:
                        await upload_video(
                            client,
                            callback_query.message.chat.id,
                            str(Path(output_file).absolute()),
                            str(Path(thumb_file).absolute()) if os.path.exists(thumb_file) else None,
                            status_msg,
                            width,
                            height,
                            channel_name,
                            actual_duration,
                            state.original_message_id
                        )
                    finally:
                        # Cleanup files
                        try:
                            if os.path.exists(output_file):
                                os.remove(output_file)
                            if thumb_file and os.path.exists(thumb_file):
                                os.remove(thumb_file)
                        except Exception as e:
                            logger.error(f"Error cleaning up files: {e}")
                            
            except Exception as e:
                logger.exception(f"Error in duration callback for user {user_id}")
                await message_handler.send_error(
                    callback_query.message.chat.id,
                    str(e)
                )
                state.recording_in_progress = False
                
    except Exception as e:
        logger.exception(f"Error in callback handler: {e}")
        await message_handler.send_error(
            callback_query.message.chat.id,
            str(e)
        )

if __name__ == "__main__":
    # Set up signal handlers for graceful shutdown
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop = asyncio.get_event_loop()
        loop.add_signal_handler(
            sig,
            lambda s=sig: asyncio.create_task(
                cleanup_manager.handle_shutdown(s)
            )
        )
    
    try:
        # Start the bot
        app.run()
    finally:
        # Cleanup
        loop = asyncio.get_event_loop()
        loop.run_until_complete(cleanup_user_sessions())
