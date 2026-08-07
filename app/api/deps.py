import hmac
import hashlib
import json
from fastapi import Request, HTTPException
from config import Setting

def verify_github_webhook_signature(request: Request):
    """
    Verifies the GitHub webhook signature.
    """
    signature = request.headers.get("X-Hub-Signature-256")
    if not signature:
        raise HTTPException(status_code=401, detail="Unauthorized: Missing signature header")

    raw_body = request.body()

    secret = Setting.GITHUB_WEBHOOK_SECRET
    computed_signature = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    
    if not hmac.compare_digest(computed_signature, signature):
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid signature")

    try:
        parsed_payload = json.loads(raw_body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Bad Request: Invalid JSON payload")

    event_type = request.headers.get("X-GitHub-Event", "unknown")

    return {
        "event_type": event_type,
        "payload": parsed_payload
    }