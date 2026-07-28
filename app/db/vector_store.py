import logging
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from pydantic import BaseModel

# Import the shared service to prevent hardcoded dimension mismatches
from app.config import Setting
from app.ml.embedder import embedding_service

logger = logging.getLogger(__name__)


class SimilarIssue(BaseModel):
    """
    Represents a matched historical issue returned from the vector database.
    Provides type safety and IDE autocomplete for downstream ML pipelines.
    """
    issue_number: int
    title: str
    similarity_score: float

class IssueVectorStore:
    """
    Service class for interacting with ChromaDB.
    Handles the storage and retrieval of issue embeddings to detect duplicates.
    """
    
    def __init__(
        self, 
        persist_directory: str = Setting.CHROMA_PATH,
        collection_name: str = Setting.COLLECTION_NAME
    ):
        """
        Initializes the ChromaDB client.
        Using a PersistentClient ensures vectors are saved to disk and survive server restarts.
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        try:
            logger.info("Initializing ChromaDB client at '%s'...", self.persist_directory)
            
            # Explicit type annotations for better IDE support
            self.client: ClientAPI = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
            
            # hnsw:space = cosine ensures ChromaDB calculates Cosine Distance rather than L2
            self.collection: Collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(
                "Successfully connected to Chroma collection '%s' (Total issues: %d)",
                self.collection_name,
                self.collection.count()
            )
        except Exception:
            logger.exception("Failed to initialize ChromaDB client.")
            raise

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        Converts ChromaDB's cosine distance (0.0 to 2.0) to a similarity score (1.0 to -1.0).
        Clamps the value between 0.0 and 1.0 to prevent floating-point arithmetic quirks.
        """
        return max(0.0, min(1.0, 1.0 - distance))

    @staticmethod
    def _issue_id(repo_name: str, issue_number: int) -> str:
        """Standardizes how unique issue IDs are formatted for ChromaDB."""
        return f"{repo_name}#{issue_number}"

    @staticmethod
    def _validate_metadata(repo_name: str, issue_number: int, title: str) -> None:
        """Ensures all metadata fields contain valid data before insertion."""
        if not repo_name.strip():
            raise ValueError("Repository name cannot be empty.")
        if not title.strip():
            raise ValueError("Title cannot be empty.")
        if issue_number <= 0:
            raise ValueError("Issue number must be strictly positive.")

    @staticmethod
    def _validate_embedding(embedding: list[float]) -> None:
        """Ensures the vector dimension matches the currently loaded ML model."""
        if len(embedding) != embedding_service.dimension:
            raise ValueError(
                f"Embedding dimension must be {embedding_service.dimension}, "
                f"got {len(embedding)}"
            )

    @staticmethod
    def _validate_embeddings(embeddings: list[list[float]]) -> None:
        """Validates dimensions for a batch of vectors."""
        if any(len(emb) != embedding_service.dimension for emb in embeddings):
            raise ValueError(
                f"All embeddings must have dimension {embedding_service.dimension}."
            )

    def store_issue(
        self, 
        repo_name: str, 
        issue_number: int, 
        embedding: list[float], 
        title: str,
        document: str | None = None
    ) -> None:
        """
        Saves a single issue's embedding, text, and metadata to the database.
        Uses upsert to safely overwrite existing data if the issue is re-processed.
        """
        self._validate_metadata(repo_name, issue_number, title)
        self._validate_embedding(embedding)
            
        issue_id = self._issue_id(repo_name, issue_number)
        
        try:
            logger.debug("Upserting vector for Issue %s in ChromaDB", issue_id)
            self.collection.upsert(
                ids=[issue_id],
                embeddings=[embedding],
                documents=[document] if document is not None else None,
                metadatas=[{
                    "repo_name": repo_name,
                    "issue_number": issue_number,
                    "title": title
                }]
            )
        except Exception:
            logger.exception("Failed to upsert issue %s in ChromaDB.", issue_id)
            raise

    def store_issues_batch(
        self, 
        repo_names: list[str], 
        issue_numbers: list[int], 
        embeddings: list[list[float]], 
        titles: list[str],
        documents: list[str] | None = None
    ) -> None:
        """
        Bulk inserts multiple issues at once. Optimized for historical data ingestion.
        """
        if not embeddings:
            return
            
        if not (len(repo_names) == len(issue_numbers) == len(embeddings) == len(titles)):
            raise ValueError("All input lists must have the exact same length.")
            
        self._validate_embeddings(embeddings)
        
        # Validate metadata individually
        for repo, num, title in zip(repo_names, issue_numbers, titles):
            self._validate_metadata(repo, num, title)

        ids = [self._issue_id(repo, num) for repo, num in zip(repo_names, issue_numbers)]
        metadatas = [
            {"repo_name": repo, "issue_number": num, "title": title}
            for repo, num, title in zip(repo_names, issue_numbers, titles)
        ]
        
        if documents is not None and len(documents) != len(ids):
            raise ValueError("Documents list must match the length of other inputs.")
            
        try:
            logger.info("Batch upserting %d issues into ChromaDB...", len(ids))
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings, # type: ignore
                documents=documents if documents is not None else None,
                metadatas=metadatas # type: ignore
            )
        except Exception:
            logger.exception("Failed to execute batch upsert in ChromaDB.")
            raise

    def search_similar_issues(
        self, 
        repo_name: str, 
        embedding: list[float], 
        limit: int = 3
    ) -> list[SimilarIssue]:
        """
        Searches the database for issues with the highest semantic similarity to the input vector.
        """
        if not repo_name.strip():
            raise ValueError("Repository name cannot be empty.")
        if limit <= 0:
            raise ValueError("Search limit must be greater than zero.")
            
        self._validate_embedding(embedding)
            
        try:
            logger.debug("Searching for similar issues in repo '%s'", repo_name)
            
            # Restrict the search to issues from the current repository.
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                where={"repo_name": repo_name},
                include=["metadatas", "distances"]
            )
            
            if not results["ids"] or not results["ids"][0]:
                return []
                
            distances = results["distances"][0] # type: ignore
            metadatas = results["metadatas"][0] # type: ignore
            
            # Use list comprehension for efficient, clean result generation
            return [
                SimilarIssue(
                    issue_number=metadata["issue_number"], # type: ignore
                    title=metadata["title"], # type: ignore
                    similarity_score=self._distance_to_similarity(distance)
                )
                for distance, metadata in zip(distances, metadatas)
            ]
            
        except Exception:
            logger.exception("Failed to search ChromaDB for similar issues.")
            raise

# Instantiate the service globally so the connection pool is reused across requests
vector_store = IssueVectorStore()