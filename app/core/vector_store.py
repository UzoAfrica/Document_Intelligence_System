
import chromadb
from sentence_transformers import SentenceTransformer
import hashlib

class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
    def add_document(self, text: str, metadata: dict = None):
        """Add document to vector store"""
        doc_id = hashlib.md5(text[:100].encode()).hexdigest()
        embedding = self.encoder.encode(text).tolist()
        
        self.collection.add(
            embeddings=[embedding],
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        return doc_id
    
    def search_similar(self, query: str, n_results: int = 5):
        """Search similar documents"""
        query_embedding = self.encoder.encode(query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        return results