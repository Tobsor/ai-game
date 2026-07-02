import chromadb
from models import Metadata
from typing import Any
from ai import AISettings, create_chat_provider, create_embedding_provider, get_ai_settings
from logger import get_logger

logger = get_logger(__name__)

class ChromaDBHelper:
    MAX_QUERY_DISTANCE = 0.4

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
        self.response_context_initialized = False

    def get_embedding(self, text: str):
        return self.embedding_provider.embed(text)
    
    def init_context(self, context: str):
        self.messages.append({"role": "system", "content": context})

    def seed_response_context(self, system_prompt: str, seed_context_prompt: str):
        if self.response_context_initialized:
            return

        seed_messages = []

        if system_prompt.strip() != "":
            seed_messages.append({"role": "system", "content": system_prompt.strip()})

        if seed_context_prompt.strip() != "":
            seed_messages.append({"role": "user", "content": seed_context_prompt.strip()})

        self.messages = seed_messages + self.messages
        self.response_context_initialized = True
    
    def add_embedding(self, id: str, text: str, metadata: Metadata | dict[str, Any] | None):
        embedding = self.get_embedding(text)

        kwargs = {
            "ids":[id],
            "documents":[text],
            "embeddings":embedding,
        }

        if metadata is not None:
            if isinstance(metadata, Metadata):
                kwargs["metadatas"] = [metadata.model_dump(mode="json", exclude_none=True)]
            else:
                kwargs["metadatas"] = [metadata]

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

        if documents is None or distances is None or len(documents[0]) == 0:
            return None

        filtered_documents: list[list[str]] = []

        for doc_group, distance_group in zip(documents, distances):
            if not isinstance(doc_group, list) or not isinstance(distance_group, list):
                continue

            matching_docs = [
                str(doc)
                for doc, distance in zip(doc_group, distance_group)
                if isinstance(distance, (int, float)) and distance <= self.MAX_QUERY_DISTANCE
            ]
            filtered_documents.append(matching_docs)

        if len(filtered_documents) == 0 or len(filtered_documents[0]) == 0:
            return None

        return filtered_documents

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
        request_messages = list(self.messages)
        res = self.response_provider.chat(messages=request_messages)
        message = {"role": "assistant", "content": res.content}
        self.messages.append(message)

        logger.conversation_event(
            stage_name=stage_name,
            event="generate_text",
            payload={"prompt": prompt},
            ai_request={"messages": list(request_messages)},
            ai_response={"content": res.content, "tool_calls": res.tool_calls},
            result={"reply": res.content},
        )

        return res.content
