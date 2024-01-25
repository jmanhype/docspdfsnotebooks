import discord
from discord.ext import commands
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import asyncpg
import logging

logger = logging.getLogger('discord')
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(filename='discord_bot.log', encoding='utf-8', mode='w')
handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s:%(name)s: %(message)s'))
logger.addHandler(handler)

class SearchHandler(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.tfidf_vectorizer = TfidfVectorizer()
        self.db_params = {
            'user': 'postgres',
            'password': 'Geraldine1',
            'database': 'data_disco',
            'host': 'localhost',
            'port': 5432
        }
        self.documents = []
        self.document_vectors = None

    async def load_documents(self):
        conn = await asyncpg.connect(**self.db_params)
        rows = await conn.fetch('SELECT content FROM documents')
        await conn.close()
        return [row['content'] for row in rows] if rows else []

    async def search(self, ctx, query, max_results=10):
        if not self.documents:
            await self.load_documents()
            self.document_vectors = self.tfidf_vectorizer.fit_transform(self.documents)

        query_vector = self.tfidf_vectorizer.transform([query])
        cos_similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(cos_similarities)[-max_results:]
        results = [(self.documents[i], cos_similarities[i]) for i in reversed(top_indices)]

        if results:
            response = "\n".join(f'{result[0]} (Score: {result[1]:.2f})' for result in results)
        else:
            response = "No results found."

        await ctx.send(response)

    @commands.command(help="Searches the messages.")
    async def search_messages(self, ctx, *, query):
        await self.search(ctx, query)

    @commands.Cog.listener()
    async def on_ready(self):
        print(f'{self.bot.user} has connected to Discord!')


# In your main bot file, you would add the SearchHandler cog like this:
# bot.add_cog(SearchHandler(bot))
# And start your bot
