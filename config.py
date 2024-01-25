# config.py
from decouple import config, Csv
import logging

# Discord Bot Settings
DISCORD_TOKEN = config('DISCORD_TOKEN')  # Bot token from the Discord developer portal
COMMAND_PREFIX = config('COMMAND_PREFIX', default='!')  # Prefix for bot commands

# Logging Configuration
LOG_LEVEL = config('LOG_LEVEL', default=logging.INFO)
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = config('LOG_FILE', default='bot.log')

# Database Configuration
DATABASE_URL = config('DATABASE_URL')  # Database connection string

# Caching Mechanism
CACHE_TYPE = config('CACHE_TYPE', default='redis')
CACHE_REDIS_URL = config('CACHE_REDIS_URL', default='redis://localhost:6379/0')
CACHE_DEFAULT_TIMEOUT = config('CACHE_DEFAULT_TIMEOUT', default=300, cast=int)

# Rate Limiting
DISCORD_API_RATE_LIMIT = config('DISCORD_API_RATE_LIMIT', default=15, cast=int)  # Requests per interval
RATE_LIMIT_INTERVAL = config('RATE_LIMIT_INTERVAL', default=15, cast=int)  # Interval in seconds

# AI Model Configuration
OPENAI_API_KEY = config('OPENAI_API_KEY')
GPT_MODEL = config('GPT_MODEL', default='text-davinci-003')

# Security Settings
REQUEST_SIGNING_SECRET = config('REQUEST_SIGNING_SECRET')
ADMIN_USER_IDS = config('ADMIN_USER_IDS', cast=Csv(cast=int))  # List of admin user IDs

# Feature Toggles
FEATURE_UNFURL_TWITTER_LINKS = config('FEATURE_UNFURL_TWITTER_LINKS', default=True, cast=bool)
FEATURE_AUTO_RESPONDER = config('FEATURE_AUTO_RESPONDER', default=True, cast=bool)

# Link Unfurling Settings
UNFURL_TWITTER_PATTERN = 'https://twitter.com/'
UNFURL_VXTWITTER_DOMAIN = 'https://vxtwitter.com/'

# Extension and Plugin Management
PLUGINS_FOLDER = config('PLUGINS_FOLDER', default='plugins')

# Internationalization (i18n) Settings
LOCALE_PATH = config('LOCALE_PATH', default='locales')
DEFAULT_LOCALE = config('DEFAULT_LOCALE', default='en')

# Ensure critical configurations are present
REQUIRED_CONFIGS = [
    'DISCORD_TOKEN', 'DATABASE_URL', 'OPENAI_API_KEY', 'REQUEST_SIGNING_SECRET'
]
for config_name in REQUIRED_CONFIGS:
    if not globals().get(config_name):
        raise ValueError(f"Missing required configuration: {config_name}")

# Logging setup
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT, filename=LOG_FILE, filemode='a')

# Additional settings and custom configurations can be added below as necessary
