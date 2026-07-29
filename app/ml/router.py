import logging
import math
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Final
from app.config import Setting
from app.ml.schemas import PRMetadata, ReviewerRecommendation, ReviewerScore

logger = logging.getLogger(__name__)


class CodeGraph:
    """
    Represents the repository's file structure and contributor history as a graph.
    Now equipped with Exponential Time Decay to phase out inactive contributors.
    """
    def __init__(self):
        # Maps a specific file to its contributors and their ownership scores
        self.file_authors: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Maps a directory to its contributors (aggregated from child files)
        self.dir_authors: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))


    @staticmethod
    def _calculate_decayed_weight(
        commit_date_str: str, 
        base_weight: float = 1.0, 
        half_life_days: int = Setting.HALF_LIFE_DAYS
    ) -> float:
        """
        Calculates the time-decayed weight of a commit using exponential decay.
        """
        try:
            # Handle standard ISO-8601 strings from GitHub, replacing 'Z' with explicit UTC offset
            commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00"))
            
            # Ensure it's timezone aware to prevent subtraction errors
            if commit_date.tzinfo is None:
                commit_date = commit_date.replace(tzinfo=timezone.utc)
                
            now = datetime.now(timezone.utc)
            delta_t = max(0, (now - commit_date).days)
            
            # Lambda = ln(2) / half_life
            decay_constant = math.log(2) / half_life_days
            decayed_weight = base_weight * math.exp(-decay_constant * delta_t)
            
            return decayed_weight
            
        except (ValueError, TypeError):
            logger.warning("Invalid date format: '%s'. Applying no time decay.", commit_date_str)
            return base_weight


    def _get_parent_directories(self, filepath: str) -> list[str]:
        """
        Helper method to extract all parent directories from a filepath.
        E.g., 'src/auth/login.tsx' -> ['src/auth', 'src']
        """
        parents = []
        current_dir = os.path.dirname(filepath)
        
        while current_dir and current_dir != '/':
            parents.append(current_dir)
            current_dir = os.path.dirname(current_dir)
            
        return parents


    def add_commit(
        self, 
        filepath: str, 
        author: str, 
        weight: float = 1.0,
        commit_date_str: str | None = None
    ) -> None:
        """
        Adds a commit to the graph, establishing an edge between the author and the file.
        Dynamically calculates time decay if a commit date is provided.
        """
        if weight <= 0:
            raise ValueError(f"Commit weight must be strictly positive, got {weight}")
            
        if not filepath or not author:
            return

        # Apply exponential time decay if a date is provided
        if commit_date_str:
            weight = self._calculate_decayed_weight(commit_date_str, base_weight=weight)

        # 1. Add direct edge from author to file
        self.file_authors[filepath][author] += weight
        
        # 2. Add edges from author to parent directory nodes
        for directory in self._get_parent_directories(filepath):
            self.dir_authors[directory][author] += weight


    def get_candidate_authors(self, filepath: str) -> dict[str, float]:
        """
        Returns a dictionary of author scores for a given file.
        If the file is new, traverses up the directory tree to find domain experts.
        """
        # Case A: We have direct history for this exact file
        if filepath in self.file_authors:
            return self.file_authors[filepath]
            
        # Case B: It is a brand new file. We traverse up the graph (BFS).
        candidates = {}
        for directory in self._get_parent_directories(filepath):
            if directory in self.dir_authors:
                logger.debug("Traversed to parent dir '%s' for new file '%s'", directory, filepath)
                # Apply penalty because they are a domain expert, not a direct file author
                for author, score in self.dir_authors[directory].items():
                    candidates[author] = score * Setting.DIRECTORY_FALLBACK_WEIGHT
                break # Stop traversing up once we find the closest domain experts
                
        if not candidates:
            logger.debug("No historical data found for '%s' or its parents.", filepath)
            
        return candidates


class PRReviewerRouter:
    """
    Service class responsible for determining the best code reviewers for a Pull Request.
    """
    def __init__(self):
        self.repo_graphs: dict[str, CodeGraph] = defaultdict(CodeGraph)
        logger.info("Initialized PR Reviewer Router.")


    def ingest_repository_history(
        self, 
        repo_name: str, 
        commits: list[dict[str, Any]], 
        overwrite: bool = False
    ) -> None:
        """
        Populates the graph for a specific repository. 
        Expects a list of dictionaries containing 'file', 'author', optional 'weight', and 'date'.
        """
        logger.info("Ingesting history for repository '%s' (Commits: %d)", repo_name, len(commits))
        
        if overwrite:
            logger.debug("Overwriting existing graph for '%s'", repo_name)
            self.repo_graphs[repo_name] = CodeGraph()
            
        graph = self.repo_graphs[repo_name]
        
        for commit in commits:
            filepath = str(commit.get('file', ''))
            author = str(commit.get('author', ''))
            date_str = commit.get('date')  # ISO-8601 string
            
            try:
                weight = float(commit.get('weight', 1.0))
                if weight > 0:
                    graph.add_commit(filepath, author, weight, commit_date_str=date_str)
                else:
                    logger.warning("Ignored commit by %s with non-positive weight: %s", author, weight)
            except (ValueError, TypeError):
                logger.warning("Ignored commit by %s due to invalid weight value", author)


    def recommend_reviewers(
        self, 
        pr_metadata: PRMetadata, 
        top_k: int = 2
    ) -> ReviewerRecommendation:
        """
        Analyzes the files changed in a PR and calculates the best reviewers.
        """
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
            
        # Short-circuit if there are no files to review
        if not pr_metadata.changed_files:
            return ReviewerRecommendation(suggested_reviewers=[])

        repo_name = pr_metadata.repo_name
        pr_author = pr_metadata.author_username
        changed_files = pr_metadata.changed_files
        
        logger.info("Calculating reviewers for PR #%d by %s", pr_metadata.pr_number, pr_author)
        
        graph = self.repo_graphs.get(repo_name)
        if not graph:
            logger.warning("No history found for repository '%s'.", repo_name)
            return ReviewerRecommendation(suggested_reviewers=[])

        candidate_scores: dict[str, float] = defaultdict(float)

        for filepath in changed_files:
            # Delegate traversal and lookup to the CodeGraph directly
            file_candidates = graph.get_candidate_authors(filepath)
            for author, score in file_candidates.items():
                candidate_scores[author] += score

        # Remove the PR author (they cannot review their own code)
        if pr_author in candidate_scores:
            del candidate_scores[pr_author]

        # Convert dict to sorted list of ReviewerScore objects (highest score first)
        sorted_candidates = sorted(
            candidate_scores.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        # Take the top K candidates
        top_candidates = sorted_candidates[:top_k]
        
        results = [
            ReviewerScore(github_username=author, relevance_score=round(score, 2))
            for author, score in top_candidates
        ]
        
        logger.info("Recommended reviewers: %s", [r.github_username for r in results])
        
        return ReviewerRecommendation(suggested_reviewers=results)

# Instantiate as a singleton so the graph remains in memory across API requests
reviewer_router = PRReviewerRouter()