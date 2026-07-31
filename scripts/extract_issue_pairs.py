import csv
import itertools
import json
import subprocess
from pathlib import Path

REPOS: list[str] = [
    "OpenLake/OpenLake--Website",
    "OpenLake/Campus-Marketplace",
    "OpenLake/Student_Database_COSA",
    "OpenLake/canonforces",
    "OpenLake/RateMyCourse",
    "OpenLake/iitbh-cgpa",
    "OpenLake/bhilaee-labs",
    "OpenLake/Centre-for-Career-Planning-and-Services-Portal",
    "OpenLake/bhilaee-simulator",
    "OpenLake/Leaderboard-Pro",
    "OpenLake/Smart-Insti-App",
    "OpenLake/Hub",
    "OpenLake/Knowledge-Sharing-Platform",
    "OpenLake/iitbh-calendar",
    "OpenLake/RideShare",
    "OpenLake/Sampoorna"
]

OUTPUT_CSV = Path("issue_pairs.csv")

# Caps issues PULLED per repo (not pairs directly) to keep the pair count
# manageable. n issues -> n*(n-1)/2 pairs, so this grows fast:
#   30 issues  ->   435 pairs
#   60 issues  -> 1,770 pairs
#   100 issues -> 4,950 pairs
# Tune this per repo based on how much manual labeling you're willing to do.
MAX_ISSUES_PER_REPO = 50


def fetch_issues(repo: str) -> list[dict]:
    """Fetches issues (open + closed) for a single repo via the GitHub CLI as JSON."""
    result = subprocess.run(
        [
            "gh", "issue", "list",
            "--repo", repo,
            "--state", "all",
            "--limit", str(MAX_ISSUES_PER_REPO),
            "--json", "number,title,body",
        ],
        capture_output=True,
        encoding="utf-8",
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def combine_text(issue: dict) -> str:
    """
    Fuses title + body into one string.
    Mirrors IssueText.embedding_text in app/ml/schemas.py, so the pairs
    you label here are text-shaped the same way the real embedding
    pipeline will see them later.
    """
    body = (issue.get("body") or "").strip()
    return f"{issue['title']}\n{body}"


def generate_pairs(repo: str, issues: list[dict]) -> list[dict]:
    """Generates every unique unordered pair of issues within a single repo."""
    rows = []
    for issue_a, issue_b in itertools.combinations(issues, 2):
        rows.append({
            "repo": repo,
            "issue_1_number": issue_a["number"],
            "issue_1_text": combine_text(issue_a),
            "issue_2_number": issue_b["number"],
            "issue_2_text": combine_text(issue_b),
            "is_duplicate": "",  # left blank on purpose - fill in manually
        })
    return rows


def main() -> None:
    all_rows: list[dict] = []

    for repo in REPOS:
        print(f"Fetching issues for {repo}...")
        try:
            issues = fetch_issues(repo)
        except subprocess.CalledProcessError as exc:
            print(f"  !! Failed to fetch {repo}: {exc.stderr.strip()}")
            continue

        pair_count = len(issues) * (len(issues) - 1) // 2
        print(f"  -> {len(issues)} issues found, generating {pair_count} pairs")
        all_rows.extend(generate_pairs(repo, issues))

    if not all_rows:
        print("No pairs generated - check your REPOS list and gh authentication.")
        return

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "repo",
                "issue_1_number", "issue_1_text",
                "issue_2_number", "issue_2_text",
                "is_duplicate",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} pairs to {OUTPUT_CSV.resolve()}")
    print("Next: open this CSV and manually fill in is_duplicate as 1 or 0 for each row.")


if __name__ == "__main__":
    main()
