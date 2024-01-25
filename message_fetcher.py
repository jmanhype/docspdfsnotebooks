from interactions import Client
from db_manager import DatabaseManager

class MessageFetcher:
    def __init__(self, bot: Client, db_manager: DatabaseManager):
        self.bot = bot
        self.db_manager = db_manager

    async def fetch_and_store_messages(self, channel_id: int, limit: int = 100):
        channel = self.bot.get_channel(channel_id)
        if not channel:
            raise ValueError("Channel not found.")

        fetched_messages = []  # List to store the fetched messages

        try:
            async for message in channel.history(limit=limit):
                if message.content:
                    await self.db_manager.store_message(
                        message_id=message.id,
                        channel_id=message.channel.id,
                        user_id=message.author.id,
                        content=message.content,
                        timestamp=message.created_at.isoformat()
                    )
                    fetched_messages.append(message)  # Add the message to the list
        except Exception as e:
            print(f"An error occurred while fetching messages: {e}")
            return None  # Return None or handle the error differently

        return fetched_messages  # Return the list of fetched messages
