from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.config import Setting

DEFAULT_MODEL = Setting.EMBEDDING_MODEL
DEFAULT_INPUT = Path("data/issue_pairs.csv")
DEFAULT_OUTPUT = Path("data/issue_pairs_with_cosine_similarity.csv")


def normalize_text(value: object) -> str:
    """Keep issue text compact without destroying useful bug details."""
    if pd.isna(value):
        return ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in text.splitlines())


def build_issue_table(pairs: pd.DataFrame) -> pd.DataFrame:
    issue_1 = pairs[["repo", "issue_1_number", "issue_1_text"]].rename(
        columns={"issue_1_number": "issue_number", "issue_1_text": "issue_text"}
    )
    issue_2 = pairs[["repo", "issue_2_number", "issue_2_text"]].rename(
        columns={"issue_2_number": "issue_number", "issue_2_text": "issue_text"}
    )

    issues = pd.concat([issue_1, issue_2], ignore_index=True)
    issues["issue_text"] = issues["issue_text"].map(normalize_text)
    issues["issue_key"] = issues["repo"].astype(str) + "#" + issues["issue_number"].astype(str)
    return issues.drop_duplicates("issue_key").reset_index(drop=True)


def cosine_for_pairs(
    pairs: pd.DataFrame,
    embeddings_by_key: dict[str, np.ndarray],
) -> list[float]:
    scores: list[float] = []

    for row in pairs.itertuples(index=False):
        key_1 = f"{row.repo}#{row.issue_1_number}"
        key_2 = f"{row.repo}#{row.issue_2_number}"
        scores.append(float(np.dot(embeddings_by_key[key_1], embeddings_by_key[key_2])))

    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    pairs = pd.read_csv(args.input)
    issues = build_issue_table(pairs)

    model = SentenceTransformer(args.model)
    embeddings = model.encode(
        issues["issue_text"].tolist(),
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings_by_key = dict(zip(issues["issue_key"], embeddings))
    pairs["cosine_similarity"] = cosine_for_pairs(pairs, embeddings_by_key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)

    print(f"Embedded {len(issues):,} unique issues")
    print(f"Scored {len(pairs):,} issue pairs")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
