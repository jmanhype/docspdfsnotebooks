import asyncio
import asyncpg
from elasticsearch import AsyncElasticsearch
from sentence_transformers import SentenceTransformer
import uuid
import sys
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    filename='bot.log',  # remove this to log to console
                    filemode='w')
logger = logging.getLogger(__name__)

# Set event loop policy to avoid SSL issues on Windows
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

class KnowledgeBase:
    def __init__(self, es_params, db_params):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.es_params = es_params
        self.db_params = db_params
        self.db_pool = None

    async def connect_to_db(self):
        try:
            self.db_pool = await asyncpg.create_pool(**self.db_params)
            logger.info("Database connection established.")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}")

    async def close_db_connection(self):
        """Close the PostgreSQL connection pool."""
        if self.db_pool:
            await self.db_pool.close()

    async def store_document(self, document_id, content):
        """Asynchronously store a document in the database."""
        embedding = self.model.encode(content).tolist()
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                'INSERT INTO documents (id, content, embedding) VALUES ($1, $2, $3)',
                document_id, content, embedding
            )

    async def search_documents(self, query):
        """Asynchronously search for documents in Elasticsearch."""
        query_vector = self.model.encode(query).tolist()
        es_query = {
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": query_vector}
                    }
                }
            }
        }
        async with AsyncElasticsearch(**self.es_params) as es:
            response = await es.search(index='knowledge_base', body=es_query)
            return response['hits']['hits']

# Example standalone usage of the KnowledgeBase class
async def main():
    # Load configuration from environment variables
    es_params = {
        'hosts': [os.getenv('ELASTICSEARCH_URL', 'https://localhost:9200')],
        'basic_auth': (
            os.getenv('ELASTICSEARCH_USER', 'elastic'),
            os.getenv('ELASTICSEARCH_PASSWORD')
        ),
        'verify_certs': os.getenv('ELASTICSEARCH_VERIFY_CERTS', 'False').lower() == 'true'
    }
    db_params = {
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD'),
        'database': os.getenv('DB_NAME', 'data_disco'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432'))
    }

    # Validate required credentials
    if not db_params['password']:
        raise ValueError("DB_PASSWORD environment variable is required")
    if not es_params['basic_auth'][1]:
        raise ValueError("ELASTICSEARCH_PASSWORD environment variable is required")

    kb = KnowledgeBase(es_params, db_params)
    await kb.connect_to_db()
    document_id = str(uuid.uuid4())
    await kb.store_document(document_id, 'This is the content of the document.')
    results = await kb.search_documents('search query')
    print(results)
    await kb.close_db_connection()

if __name__ == "__main__":
    asyncio.run(main())
