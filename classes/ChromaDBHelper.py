import ollama
import chromadb
# from vllm import LLM, SamplingParams

model_name = "PygmalionAI/mythalion-13b"

class ChromaDBHelper:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.db = chromadb.PersistentClient(path="./faction_db").get_or_create_collection("factions")
            self.model = "nomic-embed-text"
            self._initialized = True
            self.messages = []
            # self.sampling_params = SamplingParams(temperature=0.7, max_tokens=256)
            # self.llm = LLM(model=model_name)

    def get_embedding(self, text):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response['embedding']
    
    def init_context(self, context):
        self.messages.append({"role": "system", "content": context})
    
    def add_embedding(self, id, text, metadata):
        
        embedding = self.get_embedding(text)

        self.db.add(
            ids=[id],
            documents=[text],
            embeddings=embedding,
            metadatas=[metadata]
        )

    def query_docs(self, prompt, filter = None, concat = False):
        embedding = self.get_embedding(prompt)

        res = self.db.query(
            query_embeddings=[embedding],
            n_results=3,
            where=filter,
        )

        docs = [doc for doc in res["documents"][0]]

        if concat:
            return  "\n".join(docs)
        
        return docs
    
    def generate_text(self, prompt):
        new_message = {"role": "user", "content": prompt}
        self.messages.append(new_message)

        res = ollama.chat(model="vthebeast/mythalion-13b", messages=self.messages)

        self.messages.append(res["message"])

        return res["message"]["content"]