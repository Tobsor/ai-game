import chromadb
from models import Metadata
from ai import AISettings, create_chat_provider, create_embedding_provider, get_ai_settings
from logger import get_logger

logger = get_logger(__name__)

class ChromaDBHelper:
    def __init__(self, settings: AISettings | None = None):
        self.settings = settings or get_ai_settings()
        self.db = chromadb.PersistentClient(
            path=self.settings.chroma.path
        ).get_or_create_collection(
            self.settings.chroma.collection,
            metadata={"hnsw:space": self.settings.chroma.distance_space}
        )
        self.embedding_provider = create_embedding_provider(self.settings.embedding_model)
        self.response_provider = create_chat_provider(self.settings.response_llm)
        self.messages = []

    def get_embedding(self, text: str):
        return self.embedding_provider.embed(text)
    
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

        self.db.upsert(**kwargs)

    def query_docs(self, prompt: str, filter: dict[str, list[dict[str, str]]] | None = None):
        embedding = self.get_embedding(prompt)

        kwargs = {
            "query_embeddings": embedding,
            "n_results":5,
        }

        if filter != None:
            kwargs["where"] = filter

        res = self.db.query(**kwargs)

        documents = res.get("documents")
        distances = res.get("distances")

        if documents == None or len(documents[0]) == 0:
            return None
        
        return documents

    def parse_retrieved_docs(self, documents) -> list[str]:
        parsed_docs: list[str] = []

        for doc_group in documents:
            if not isinstance(doc_group, list):
                continue
            for doc in doc_group:
                snippet = str(doc).strip()
                if snippet != "":
                    parsed_docs.append(snippet)

        return parsed_docs
    
    def query_text(self, prompt: str, filter = None, stage_name: str = "RetrievalStage"):
        docs = self.query_docs(prompt=prompt, filter=filter)

        if(docs == None):
            logger.conversation_event(
                stage_name=stage_name,
                event="query_text",
                payload={"prompt": prompt, "filter": filter},
                result={"documents": [], "text": ""},
            )
            return ""

        result = "\n".join(self.parse_retrieved_docs(docs))
        logger.conversation_event(
            stage_name=stage_name,
            event="query_text",
            payload={"prompt": prompt, "filter": filter},
            result={"documents": docs, "text": result},
        )

        return result
    
    def generate_text(self, prompt: str, stage_name: str = "ResponseStage") -> str:
        new_message = {"role": "user", "content": prompt}
        self.messages.append(new_message)

        res = self.response_provider.chat(messages=self.messages)
        message = {"role": "assistant", "content": res.content}
        self.messages.append(message)

        logger.conversation_event(
            stage_name=stage_name,
            event="generate_text",
            payload={"prompt": prompt},
            ai_request={"messages": list(self.messages[:-1])},
            ai_response={"content": res.content, "tool_calls": res.tool_calls},
            result={"reply": res.content},
        )

        return res.content
