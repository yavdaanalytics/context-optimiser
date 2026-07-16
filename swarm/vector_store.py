"""Vector store module.

Provides VectorMemoryStore class.
"""
import chromadb
import logging

log = logging.getLogger("swarm.vector_store")

class VectorMemoryStore:
    def __init__(self, persist_directory: str = ".agent/data/chromadb"):
        self.persist_directory = persist_directory
        try:
            self.client = chromadb.PersistentClient(path=self.persist_directory)
        except Exception as e:
            log.warning(f"Failed to initialize ChromaDB PersistentClient: {e}")
            self.client = None

    def store(self, doc_id: str, document: str, metadata: dict, collection_name: str = "context_offloads") -> bool:
        if not self.client:
            return False
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            # Chroma expects metadata values to be str, int, float, or bool
            cleaned_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    cleaned_metadata[k] = v
                elif v is None:
                    cleaned_metadata[k] = ""
                else:
                    cleaned_metadata[k] = str(v)
            
            collection.add(
                documents=[document],
                metadatas=[cleaned_metadata],
                ids=[doc_id]
            )
            return True
        except Exception as e:
            log.warning(f"ChromaDB store failed for ID {doc_id}: {e}")
            return False

    def query(self, query_text: str, n_results: int = 3, collection_name: str = "context_offloads") -> list:
        if not self.client:
            return []
        try:
            collection = self.client.get_or_create_collection(name=collection_name)
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results
            )
            output = []
            if results and 'documents' in results and results['documents']:
                for i in range(len(results['documents'][0])):
                    doc = results['documents'][0][i]
                    meta = results['metadatas'][0][i] if 'metadatas' in results and results['metadatas'] else {}
                    doc_id = results['ids'][0][i] if 'ids' in results and results['ids'] else ""
                    output.append({
                        "id": doc_id,
                        "document": doc,
                        "metadata": meta
                    })
            return output
        except Exception as e:
            log.warning(f"ChromaDB query failed: {e}")
            return []
