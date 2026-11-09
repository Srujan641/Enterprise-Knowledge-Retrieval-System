import threading
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from utils.auth import can_read
from utils.config import settings


class VectorStoreManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.HUGGINGFACE_EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )
        self.index_path = settings.index_path
        self.vector_store = None
        self._lock = threading.RLock()
        self._load_index()

    def _load_index(self):
        if not self.index_path.is_dir():
            print("No existing FAISS index found. Ready to initialize on first ingestion.")
            return
        try:
            # Safe only because this index is written exclusively by this application.
            self.vector_store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            print("Loaded existing FAISS index.")
        except Exception as e:
            print(f"Failed to load FAISS index: {e}")

    def _chunks_for(self, doc_path: str) -> dict:
        if self.vector_store is None:
            return {}
        return {
            chunk_id: doc
            for chunk_id, doc in self.vector_store.docstore._dict.items()
            if doc.metadata.get("doc_path") == doc_path
        }

    def is_indexed(self, doc_path: str, doc_hash: str) -> bool:
        with self._lock:
            return any(d.metadata.get("doc_hash") == doc_hash for d in self._chunks_for(doc_path).values())

    def replace_document(self, doc_path: str, documents: List[Document]):
        """Index the new chunks, then drop any earlier version of the same file."""
        with self._lock:
            old_ids = list(self._chunks_for(doc_path))
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)
            if old_ids:
                self.vector_store.delete(old_ids)
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.vector_store.save_local(str(self.index_path))

    def search(self, query: str, top_k: int = 5, role: str = "general") -> List[Document]:
        with self._lock:
            if self.vector_store is None or self.vector_store.index.ntotal == 0:
                return []
            # Filter across the whole index so other roles' chunks can't crowd out readable ones.
            results = self.vector_store.similarity_search_with_score(
                query,
                k=top_k,
                fetch_k=self.vector_store.index.ntotal,
                filter=lambda metadata: can_read(role, metadata.get("role", "general")),
            )
        return [doc for doc, _ in results]

    def search_with_context(self, query: str, top_k: int = 5, role: str = "general", max_chunks: int = 12) -> List[Document]:
        """Best matches plus the rest of their sections, in reading order."""
        hits = self.search(query, top_k=top_k, role=role)
        selected = {(d.metadata["doc_path"], d.metadata["chunk_index"]): d for d in hits}
        with self._lock:
            for hit in hits:
                if not hit.metadata.get("section"):
                    continue
                for doc in self._chunks_for(hit.metadata["doc_path"]).values():
                    key = (doc.metadata["doc_path"], doc.metadata["chunk_index"])
                    if len(selected) < max_chunks and doc.metadata.get("section") == hit.metadata["section"]:
                        selected.setdefault(key, doc)
        return [selected[key] for key in sorted(selected)]


vector_store_manager = VectorStoreManager()
