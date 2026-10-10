import chromadb
from models import Metadata
from typing import Any
from ai import AISettings, create_chat_provider, create_embedding_provider, get_ai_settings
from logger import get_logger
from embedding_text import format_embedding_text

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
        """Accept raw document text; persist and embed the same metadata-prefixed text."""
        if isinstance(metadata, Metadata):
            metadata = metadata.model_dump(mode="json", exclude_none=True)
        text = format_embedding_text(text, metadata)
        embedding = self.get_embedding(text)

        kwargs = {
            "ids":[id],
            "documents":[text],
            "embeddings":embedding,
        }

        if metadata:
            kwargs["metadatas"] = [metadata]

        self.db.upsert(**kwargs)

    def query_records(self, prompt: str, filter: dict | None = None) -> list[dict[str, Any]]:
        """One vector query, retaining evidence identity and provenance for turn-level reuse."""
        embedding = self.get_embedding(prompt)

        kwargs = {
            "query_embeddings": embedding,
            "n_results":5,
            "include": ["documents", "metadatas", "distances"],
        }

        if filter != None:
            kwargs["where"] = filter

        res = self.db.query(**kwargs)

        records = []
        for ids, docs, distances, metadatas in zip(
            res.get("ids") or [], res.get("documents") or [],
            res.get("distances") or [], res.get("metadatas") or [],
        ):
            for identifier, doc, distance, metadata in zip(ids, docs, distances, metadatas):
                if doc and str(doc).strip() and isinstance(distance, (int, float)) and distance <= self.MAX_QUERY_DISTANCE:
                    records.append({"id": identifier, "text": str(doc),
                                    "distance": distance, "metadata": metadata or {}})
        return records

    def query_docs(self, prompt: str, filter: dict | None = None):
        records = self.query_records(prompt, filter)
        return [[record["text"] for record in records]] if records else None

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
    
    def query_text(self, prompt: str, filter = None, stage_name: str = "ContextRetrieval"):
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
