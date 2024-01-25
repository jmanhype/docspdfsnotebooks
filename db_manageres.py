from elasticsearch import AsyncElasticsearch
from typing import List, Any, Optional

class ElasticsearchManager:
    def __init__(self, hosts: List[str]):
        self.es = AsyncElasticsearch(hosts=hosts)

    async def index_message(self, index: str, message: dict):
        # Index a message document into Elasticsearch
        await self.es.index(index=index, document=message)

    async def fetch_messages(self, index: str, query: dict, size: int = 100) -> List[Any]:
        # Search for messages in Elasticsearch
        response = await self.es.search(index=index, body=query, size=size)
        return [hit["_source"] for hit in response["hits"]["hits"]]

    async def get_message(self, index: str, id: str) -> Optional[dict]:
        # Get a specific message by ID
        try:
            response = await self.es.get(index=index, id=id)
            return response["_source"]
        except Exception as e:
            # Handle exceptions, e.g., document not found
            return None

    async def update_message(self, index: str, id: str, update_body: dict):
        # Update a specific message by ID
        await self.es.update(index=index, id=id, body={"doc": update_body})

    async def delete_message(self, index: str, id: str):
        # Delete a specific message by ID
        await self.es.delete(index=index, id=id)

    async def close(self):
        # Close the connection to Elasticsearch
        await self.es.close()

# Example usage
# es_manager = ElasticsearchManager(hosts=["http://localhost:9200"])
# await es_manager.index_message("messages", {"user": "john_doe", "content": "Hello, world!", "timestamp": "2023-11-06T12:00:00Z"})
# messages = await es_manager.fetch_messages("messages", {"query": {"match_all": {}}})
# await es_manager.close()
