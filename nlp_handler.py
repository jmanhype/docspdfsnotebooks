from ctransformers import AutoModelForCausalLM
from search_handler import SearchHandler
from knowledge_base import KnowledgeBase

class NLPHandler:
    def __init__(self, search_handler: SearchHandler, knowledge_base: KnowledgeBase):
        self.search_handler = search_handler
        self.knowledge_base = knowledge_base
        # Initialize the model using ctransformers
        self.model = AutoModelForCausalLM.from_pretrained(
            "TheBloke/Mistral-7B-v0.1-GGUF", 
            model_file="mistral-7b-v0.1.Q2_K.gguf", 
            model_type="mistral", 
            gpu_layers=0  # Set the number of layers for GPU
        )

    async def retrieve_and_generate(self, query):
        documents = await self.search_handler.load_documents()
        generation_context = " ".join(documents)
        prompt = f"Answer the following question based on the documents: {query}\n{generation_context}"
        # Generate text using the loaded model
        response = self.model(prompt)
        return response

# Example usage
if __name__ == "__main__":
    async def main():
        sh = SearchHandler()
        kb = KnowledgeBase()
        nlp_handler = NLPHandler(search_handler=sh, knowledge_base=kb)
        query = "What is the latest discussion about data analysis?"
        response = await nlp_handler.retrieve_and_generate(query)
        print(response)

    import asyncio
    asyncio.run(main())
