import ollama
import chromadb
from models import Metadata

model_name = "nollama/mythomax-l2-13b:Q4_K_M"

class ChromaDBHelper:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.db = chromadb.PersistentClient(path="./faction_db").get_or_create_collection("factions", metadata={"hnsw:space": "cosine"})
            self.model = "nomic-embed-text"
            self._initialized = True
            self.messages = []

    def get_embedding(self, text: str):
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        return response['embedding']
    
    def init_context(self, context: str):
        self.messages.append({"role": "system", "content": context})
    
    def add_embedding(self, id: str, text: str, metadata: Metadata | None):
        
        embedding = self.get_embedding(text)

        kwargs = {
            "ids":[id],
            "documents":[text],
            "embeddings":embedding,
        }

        if metadata is not None:
            kwargs["metadatas"] = [metadata.model_dump(mode="json", exclude_none=True)]

        self.db.add(**kwargs)

    def query_docs(self, prompt: str, filter: dict[str, list[dict[str, str]]] | None = None):
        embedding = self.get_embedding(prompt)

        kwargs = {
            "query_embeddings":[embedding],
            "n_results":5,
        }

        if filter != None:
            kwargs["where"] = filter

        res = self.db.query(**kwargs)

        documents = res.get("documents")
        distances = res.get("distances")

        print(distances)
        print(documents)
        if documents == None or len(documents[0]) == 0:
            return None
        
        return documents
    
    def query_text(self, prompt: str, filter = None) :
        docs = self.query_docs(prompt=prompt, filter=filter)

        if(docs == None):
            return ""
        
        print(docs)

        docs_text = [doc[0] for doc in docs]
        
        return  "\n".join(docs_text)
    
    def generate_text(self, prompt: str) -> str:
        new_message = {"role": "user", "content": prompt}
        self.messages.append(new_message)

        # start = time.time()   
        res = ollama.chat(model=model_name, messages=self.messages)

        # end = time.time()   
        # print("Creating text took: " + str(end - start))
        self.messages.append(res["message"])

        return res["message"]["content"]