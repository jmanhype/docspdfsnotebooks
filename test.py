import asyncpg

async def test_connection():
    conn = await asyncpg.connect(dsn="postgresql://postgres:Geraldine1@localhost:5432/data_disco")
    print("Successfully connected to the database")
    await conn.close()

import asyncio
asyncio.run(test_connection())
