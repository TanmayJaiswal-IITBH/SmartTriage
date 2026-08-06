from fastapi import APIRouter, Request, HTTPException, Depends
from app.api.deps import verify_github_webhook_signature

# --- Upcoming Imports (We will uncomment these as we build them) ---
# from app.core.utils import clean_text
# from app.ml.embedder import check_if_duplicate 
# from app.core.github_app import GithubAppClient
# from app.config import settings

router = APIRouter()

# Instantiate the client (Usually injected or instantiated at startup)
# github_client = GithubAppClient(settings)

@router.post("/webhook")
async def github_webhook(
    request: Request, 
    verified_data: dict = Depends(verify_github_webhook_signature)
):
    """
    Endpoint to handle GitHub webhook events.
    """
    event = verified_data["event_type"]
    payload = verified_data["payload"]
    
    try:
        # 1. Handle Ping Event
        if event == "ping":
            return {"message": "Webhook received successfully!"}
        
        # 2. Filter for Issues Events
        if event == "issues":
            action = payload.get("action")
            
            # 3. Filter for Action Type
            if action in ["opened", "reopened"]:
                installation_id = payload["installation"]["id"]
                repo_name = payload["repository"]["full_name"]
                issue_number = payload["issue"]["number"]
                issue_title = payload["issue"]["title"]
                issue_body = payload["issue"]["body"] or ""
                
                # ==========================================
                # THE HANDOFF (Phases 7, ML, and 12/13)
                # ==========================================
                
                # STEP A: Clean the text (Phase 7)
                # cleaned_body = clean_text(issue_body)
                
                # STEP B: Ask the ML model for similarity (Friend's ML Pipeline)
                # is_duplicate, match_score, duplicate_of_id = await check_if_duplicate(issue_title, cleaned_body)
                
                # STEP C: Take Action via GitHub API (Phases 12/13)
                # if is_duplicate:
                #     reply = f"Hi! Our ML model detected this is a duplicate of #{duplicate_of_id} with {match_score}% confidence."
                #     await github_client.post_triage_comment(installation_id, repo_name, issue_number, reply)
                #     await github_client.close_issue(installation_id, repo_name, issue_number)
                #     
                #     return {"message": f"Closed duplicate issue #{issue_number}"}
                
                return {"message": f"Processed Issue #{issue_number}: {issue_title}"}
            
            else:
                return {"message": f"Ignored issue action: {action}"}
            
        elif event == "pull_request":
            action = payload.get("action")
            if action in ["opened", "reopened", "synchronize"]:
                installation_id = payload["installation"]["id"]
                repo_name = payload["repository"]["full_name"]
                pr_number = payload["pull_request"]["number"]
                pr_title = payload["pull_request"]["title"]
                pr_body = payload["pull_request"]["body"] or ""
                
                return {"status_code": 200, "detail": f"PR #{pr_number} ({action}): {pr_title}"}
            else:
                return {"status_code": 200, "detail": f"Ignored PR action: {action}"}

        return {"message": f"Ignored event type: {event}"}
        
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Malformed payload: missing key {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")