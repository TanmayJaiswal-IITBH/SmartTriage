from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import verify_github_webhook_signature
from app.core.github_app import GithubAppClient
from app.core.utils import clean_text
from app.config import Setting

router = APIRouter()

# Instantiate GitHub App Client
github_client = GithubAppClient(config=Setting)


@router.post("/webhook")
async def github_webhook(
    request: Request, 
    verified_data: dict = Depends(verify_github_webhook_signature)
):
    """
    Handles GitHub Webhook events for Issues and Pull Requests.
    """
    event_type = verified_data["event_type"]
    payload = verified_data["payload"]

    try:
        # 1. Handle Ping Event
        if event_type == "ping":
            return {"status_code": 200, "detail": "Webhook received successfully!"}

        # 2. Handle Issues Event
        if event_type == "issues":
            action = payload.get("action")

            if action in ["opened", "reopened"]:
                installation_id = payload["installation"]["id"]
                repo_name = payload["repository"]["full_name"]
                issue_number = payload["issue"]["number"]
                issue_title = payload["issue"]["title"]
                raw_body = payload["issue"]["body"] or ""

                # Clean raw markdown body
                cleaned_title = clean_text(issue_title)
                cleaned_body = clean_text(raw_body)

                issue_payload = f"TITLE: {cleaned_title} | BODY: {cleaned_body}"

                # =========================================================
                # ML Pipeline Handoff Spot for Issues
                # =========================================================
                # Example:
                # is_dup, score, orig_id = await check_if_duplicate(issue_title, cleaned_body)
                # if is_dup:
                #     msg = f"SmartTriage: Duplicate of #{orig_id} ({score}% match)."
                #     await github_client.post_comment(installation_id, repo_name, issue_number, msg)
                #     await github_client.close_item(installation_id, repo_name, issue_number)

                return {
                    "status_code": 200,
                    "detail": f"Processed Issue #{issue_number} ({action})",
                    "repo": repo_name,
                    "issue_payload": issue_payload
                }

            return {"status_code": 200, "detail": f"Ignored issue action: {action}"}

        # 3. Handle Pull Request Event
        elif event_type == "pull_request":
            action = payload.get("action")

            if action in ["opened", "reopened", "synchronize"]:
                installation_id = payload["installation"]["id"]
                repo_name = payload["repository"]["full_name"]
                pr_number = payload["pull_request"]["number"]
                pr_title = payload["pull_request"]["title"]
                raw_body = payload["pull_request"]["body"] or ""

                # Fetch PR commit messages asynchronously
                raw_commit_messages = await github_client.get_pr_commit_messages(
                    installation_id, repo_name, pr_number
                )

                # Clean texts
                cleaned_body = clean_text(raw_body)
                cleaned_commits = clean_text(raw_commit_messages)

                # Structured Priority Payload for PR ML Pipeline
                pr_text_payload = f"TITLE: {pr_title} | COMMITS: {cleaned_commits} | DESCRIPTION: {cleaned_body}"

                # =========================================================
                # ML Pipeline Handoff Spot for PRs
                # =========================================================
                # Example:
                # is_dup, score, orig_id = await check_if_duplicate_pr(pr_text_payload)
                # if is_dup:
                #     msg = f"SmartTriage: PR appears to duplicate #{orig_id} ({score}% match)."
                #     await github_client.post_comment(installation_id, repo_name, pr_number, msg)
                #     await github_client.close_item(installation_id, repo_name, pr_number)

                return {
                    "status_code": 200,
                    "detail": f"Processed PR #{pr_number} ({action})",
                    "repo": repo_name,
                    "pr_payload": pr_text_payload
                }

            return {"status_code": 200, "detail": f"Ignored PR action: {action}"}

        # 4. Fallback for unhandled events
        return {"status_code": 200, "detail": f"Ignored event type: {event_type}"}

    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Malformed payload: missing key {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")