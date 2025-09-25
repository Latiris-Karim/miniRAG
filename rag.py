from openai import OpenAIError, OpenAI
import os
import torch
import time
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import semantic_search
import pickle

# initialize model and client
sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
openai_client = OpenAI(api_key='YOUR_API_KEY', base_url="https://api.deepseek.com")

# load embeddings and texts if they exist
if os.path.exists('embeddings.pt') and os.path.exists('chunks.pkl'):
    embeddings = torch.load('embeddings.pt', map_location='cpu', weights_only=True)
    with open('chunks.pkl', 'rb') as f:
        texts = pickle.load(f)
else:
    embeddings = None
    texts = None

class RAG:
    def __init__(self):
        self.client = openai_client
        self.embeddings = embeddings
        self.model: SentenceTransformer = sentence_model
        self.texts = texts
        
    def get_embedding(self, texts):
        embeddings = self.model.encode(texts)
        return embeddings
    
    def get_context(self, user_input):
        # get embedding for the user input
        output = self.get_embedding([user_input])  
        query_embeddings = torch.FloatTensor(output)
        
        # search for similar chunks
        hits = semantic_search(query_embeddings, self.embeddings, top_k=2)
        
        # retrieve and return the matching context chunks
        context = [self.texts[hits[0][i]['corpus_id']] for i in range(len(hits[0]))]
        return context
   
    def format_query(self, question, context):
        context_str = " ".join(context)
        return f"""
        The following is relevant context:
        {context_str}

        Question: {question}
        """

    def get_response(self, query):
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": query}],
                temperature=0.0
            )
            if response.choices and len(response.choices) > 0:
                return response.choices[0].message.content
            else:
                return "Error: No response from API."
        except OpenAIError as e:  
            return f"OpenAI API Error: {e}"
        except Exception as e:
            return f"Unexpected error: {e}"
    
    def pipeline(self, question):
        t0 = time.time()
        context = self.get_context(question)
        t1 = time.time()
        query = self.format_query(question, context)
        t2 = time.time()
        response = self.get_response(query)
        t3 = time.time()
        
        print(f"Context time: {t1 - t0:.2f}s")
        print(f"Formatting time: {t2 - t1:.2f}s")
        print(f"LLM time: {t3 - t2:.2f}s")
        return response
        
if __name__ == "__main__":
    rag = RAG()
    print("RAG system ready! Type 'quit' to exit.")
    
    while True:
        print("-----------------------------------------------------------------------------------")
        user_question = input("Q: ")
        if user_question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        print("-----------------------------------------------------------------------------------")
        answer = rag.pipeline(user_question)
        print(answer)
