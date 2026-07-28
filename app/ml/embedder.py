import logging
from typing import Any, Final

from sentence_transformers import SentenceTransformer
from app.config import Setting
from app.ml.schemas import IssueText

logger = logging.getLogger(__name__)

# Moving the model name to a constant makes it easier to migrate to a config file later.
MODEL_NAME: Final[str] = Setting.EMBEDDING_MODEL

# Explicitly typed shared arguments for consistency across all encoding operations
ENCODE_KWARGS: Final[dict[str, Any]] = {
    "convert_to_numpy": True,
    "normalize_embeddings": True,
    "show_progress_bar": False,
}

class EmbeddingService:
    """
    Service class for managing the NLP embedding model and generating vectors.
    This abstraction allows for easier dependency injection, mocking in tests, 
    and scaling to multiple data types (e.g., Pull Requests, Comments).
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        
        try:
            logger.info("Loading SentenceTransformer model '%s'...", self.model_name)
            
            # The model MUST be loaded before querying its properties
            self.model: SentenceTransformer = SentenceTransformer(self.model_name)
            self._embedding_dimension: int = self.model.get_embedding_dimension()  # type: ignore
            
            logger.info(
                "Loaded '%s' (embedding_dim=%d, device=%s)",
                self.model_name,
                self._embedding_dimension,
                self.model.device,
            )
        except Exception:
            logger.exception("Failed to initialize the SentenceTransformer model.")
            raise

    @property
    def dimension(self) -> int:
        """Exposes the embedding dimension for downstream validation."""
        return self._embedding_dimension

    @staticmethod
    def _validate_text(text: str) -> None:
        """Ensures the text is valid before attempting to embed it."""
        if not text.strip():
            raise ValueError("Cannot embed empty text.")

    @staticmethod
    def _validate_batch(texts: list[str]) -> None:
        """Ensures the batch of texts contains no empty strings."""
        if not texts:
            raise ValueError("Batch cannot be empty.")
        if any(not text.strip() for text in texts):
            raise ValueError("Batch contains empty text.")

    def generate_embedding(self, text: str) -> list[float]:
        """
        Converts a single raw string into a dense numerical vector.
        
        Args:
            text (str): The text to embed.
            
        Returns:
            list[float]: A numerical vector representing the semantic meaning of the text.
        """
        self._validate_text(text)
            
        try:
            # We unpack ENCODE_KWARGS for consistent normalization and types
            embedding_array = self.model.encode(text, **ENCODE_KWARGS)
            return embedding_array.tolist()
        except Exception:
            logger.exception("Failed to generate embedding.")
            raise

    def generate_embeddings(self, texts: list[str], batch_size: int = 64) -> list[list[float]]:
        """
        Converts a list of raw strings into a list of dense numerical vectors.
        Optimized for batch processing historical issues.
        
        Args:
            texts (list[str]): A list of strings to embed.
            batch_size (int): The number of texts to process at once.
            
        Returns:
            list[list[float]]: A list of numerical vectors corresponding to the inputs.
        """
        self._validate_batch(texts)
            
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
            
        try:
            embeddings_array = self.model.encode(
                texts,
                batch_size=batch_size,
                **ENCODE_KWARGS
            )
            return embeddings_array.tolist()
        except Exception:
            logger.exception("Failed to generate batch embeddings.")
            raise

    def embed_issue(self, issue: IssueText) -> list[float]:
        """
        Takes an IssueText schema and returns its vector representation.
        
        Args:
            issue (IssueText): The structured GitHub issue payload.
            
        Returns:
            list[float]: The vector embedding of the issue's combined title and body.
        """
        logger.debug(
            "Generating embedding for Issue #%s in %s",
            issue.issue_number,
            issue.repo_name,
        )
        
        # Use the fused title + body defined by IssueText
        return self.generate_embedding(issue.embedding_text)

# The service is instantiated once at the module level so the heavy 90MB model 
# stays loaded in memory across all incoming FastAPI webhook requests.
embedding_service = EmbeddingService()