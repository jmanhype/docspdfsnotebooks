import asyncpg
from typing import List, Any, Optional
from contextlib import asynccontextmanager
import logging
import json



# Configure logger
logger = logging.getLogger('database_manager')
logger.setLevel(logging.INFO)

# Stream handler for logging to console
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# File handler for logging to a file
file_handler = logging.FileHandler('database_manager.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


class DatabaseManager:
    def __init__(self, dsn: str):
        self.dsn = "postgresql://postgres:Geraldine1@localhost:5432/data_disco"
        self.pool = None

    async def connect(self):
        try:
            # Use the explicit DSN here
            self.pool = await asyncpg.create_pool(dsn=self.dsn)
            logger.info("Successfully connected to the database.")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")

    @asynccontextmanager
    async def get_connection(self):
        assert self.pool is not None, "Database connection pool not initialized."
        async with self.pool.acquire() as connection:
            yield connection

    async def fetch_messages(self, channel_id: str, limit: int = 100) -> List[Any]:
        async with self.get_connection() as conn:
            stmt = await conn.prepare('SELECT * FROM messages WHERE channel_id=$1 ORDER BY timestamp DESC LIMIT $2')
            return await stmt.fetch(channel_id, limit)

    async def store_message(self, message_id: str, channel_id: str, user_id: str, content: str, timestamp: str):
        async with self.get_connection() as conn:
            await conn.execute('''
                INSERT INTO messages(message_id, channel_id, user_id, content, timestamp)
                VALUES($1, $2, $3, $4, $5)
            ''', message_id, channel_id, user_id, content, timestamp)

    async def fetch_link_data(self, link: str) -> Optional[dict]:
        async with self.get_connection() as conn:
            return await conn.fetchrow('SELECT * FROM link_data WHERE link=$1', link)

    async def store_link_data(self, link: str, data: dict):
        async with self.get_connection() as conn:
            await conn.execute('''
                INSERT INTO link_data(link, data) VALUES($1, $2)
                ON CONFLICT (link) DO UPDATE SET data = $2
            ''', link, json.dumps(data))

    async def disconnect(self):
        if self.pool:
            await self.pool.close()
            logger.info("Database connection closed.")


# Example usage
# db_manager = DatabaseManager(dsn="postgresql://postgres:Geraldine1@localhost:5432/data_disco")
# await db_manager.connect()
# messages = await db_manager.fetch_messages(channel_id="123456789")
# await db_manager.store_message("123", "123456789", "987654321", "Hello, World!", "2023-11-06T12:00:00Z")
# await db_manager.disconnect()