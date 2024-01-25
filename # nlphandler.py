from transformers import pipeline, set_seed
from knowledge_base import KnowledgeBase
from message_fetcher import MessageFetcher
from link_transformer import LinkTransformer

class NLPHandler:
    def __init__(self, knowledge_base, message_fetcher, link_transformer):
        self.knowledge_base = knowledge_base
        self.message_fetcher = message_fetcher
        self.link_transformer = link_transformer
        self.generator = pipeline('text-generation', model='gpt-2', tokenizer='gpt-2')
        set_seed(42)

    def update_knowledge_base_with_messages(self):
        messages = self.message_fetcher.fetch_all_messages()
        transformed_links = [self.link_transformer.transform_twitter_link(message) for message in messages if "twitter.com" in message]
        for message in messages:
            self.knowledge_base.update_with_new_message(message)

    def retrieve_and_generate(self, query):
        relevant_messages = self.knowledge_base.search(query)
        context_for_generation = " ".join([message['content'] for message in relevant_messages])
        prompt = f"Given the following information: {context_for_generation}, answer the question: {query}"
        responses = self.generator(prompt, max_length=150, num_return_sequences=1)
        return responses[0]['generated_text']

if __name__ == "__main__":
    kb = KnowledgeBase()
    mf = MessageFetcher()
    lt = LinkTransformer()
    nlp_handler = NLPHandler(knowledge_base=kb, message_fetcher=mf, link_transformer=lt)
    nlp_handler.update_knowledge_base_with_messages()
    query = "What are the recent discussions about data analysis?"
    generated_answer = nlp_handler.retrieve_and_generate(query)
    print(generated_answer)
