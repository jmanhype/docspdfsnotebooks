import interactions
import re
import uuid
import logging
import os
from dotenv import load_dotenv
from interactions import Client, SlashContext, slash_command, slash_option, Task, IntervalTrigger, listen, OptionType
from link_transformer import GenericLinkUnfurler
from knowledge_base import KnowledgeBase
from message_fetcher import MessageFetcher
from nlp_handler import NLPHandler
from db_manager import DatabaseManager
from search_handler import SearchHandler

# Load environment variables
load_dotenv()

# Constants and Configurations
TOKEN = os.getenv('DISCORD_TOKEN')
if not TOKEN:
    raise ValueError("DISCORD_TOKEN environment variable is required")

TWITTER_REGEX = re.compile(r'https?://twitter\.com/\w+/status/\d+')

# Database connection string
DSN = os.getenv('DATABASE_URL', 'postgresql://localhost:5432/data_disco')

# Database and Elasticsearch parameters
db_params = {
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'data_disco'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', '5432'))
}

es_params = {
    'hosts': [os.getenv('ELASTICSEARCH_URL', 'https://localhost:9200')],
    'basic_auth': (
        os.getenv('ELASTICSEARCH_USER', 'elastic'),
        os.getenv('ELASTICSEARCH_PASSWORD')
    ),
    'verify_certs': os.getenv('ELASTICSEARCH_VERIFY_CERTS', 'False').lower() == 'true'
}

# Validate required credentials
if not db_params['password']:
    raise ValueError("DB_PASSWORD environment variable is required")
if not es_params['basic_auth'][1]:
    raise ValueError("ELASTICSEARCH_PASSWORD environment variable is required")


# Initialize bot and slash commands
bot = Client(token=TOKEN)

# Initialize components
url_unfurler = GenericLinkUnfurler()
knowledge_base = KnowledgeBase(es_params, db_params)
search_handler = SearchHandler(bot)
nlp_handler = NLPHandler(search_handler, knowledge_base)
db_manager = DatabaseManager(db_params)
message_fetcher = MessageFetcher(bot, db_manager)

# Setup logging
logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord_bot_debug.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

# Event Handler: Bot is ready
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    logger.info("Bot is now online.")
    await db_manager.connect()  # Ensure this is successful before proceeding
    if db_manager.pool:
        logger.info("Connected to database.")
        # Now it's safe to start tasks that use the database
        update_knowledge_base.start()
        fetch_all_history.start()
    else:
        logger.error("Failed to connect to database. Tasks not started.")


# Event Handler: Message received
@bot.event
async def on_message(message):
    if message.author == bot.user:
        return
    twitter_links = TWITTER_REGEX.findall(message.content)
    for link in twitter_links:
        unfurled_link = url_unfurler.unfurl(link)
    nlp_result = await nlp_handler.retrieve_and_generate(message.content)

# Corrected Slash Command: Fetch History
@slash_command(name="fetch_history", description="Fetch the history of a channel")
@slash_option(name="channel_id", description="The ID of the channel", opt_type=OptionType.STRING, required=True)
async def fetch_history(ctx: SlashContext, channel_id: str):
    try:
        channel_id = int(channel_id)
    except ValueError:
        await ctx.send("Invalid channel ID.", ephemeral=True)
        return

    channel = bot.get_channel(channel_id)  # Removed 'await' here
    if not channel:
        await ctx.send("Channel not found.", ephemeral=True)
        return

    history = await message_fetcher.fetch_and_store_messages(channel_id)
    if history:
        formatted_history = "\n".join([f"{msg.author}: {msg.content}" for msg in history])
        await ctx.send(f"History: {formatted_history}")
    else:
        await ctx.send("No history found or an error occurred.")


# Slash Command: Ask Question
@slash_command(name="ask_question", description="Ask a question to the bot")
@slash_option(name="question", description="Question to ask", opt_type=OptionType.STRING, required=True)
async def ask_question(ctx: SlashContext, question: str):
    search_results = await knowledge_base.search_documents(question)
    await ctx.send(f"Search results: {search_results}")

# Task: Update Knowledge Base
@Task.create(IntervalTrigger(hours=24))
async def update_knowledge_base():
    document_id = str(uuid.uuid4())
    content = "New content for the knowledge base."
    await knowledge_base.store_document(document_id, content)

# Task: Fetch All History
@Task.create(IntervalTrigger(hours=24))
async def fetch_all_history():
    guild_id = 'your_guild_id_here'
    for channel in await bot.get_guild_channels(guild_id):
        if isinstance(channel, interactions.ChannelType.TEXT):
            await message_fetcher.fetch_and_store_messages(channel.id)



# Error Handling
@bot.event
async def on_command_error(ctx, error):
    await ctx.send(f"An error occurred: {error}")

bot.start()
