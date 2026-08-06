import asyncio
from datetime import datetime, timedelta, timezone
from app.config import Setting, settings as Config
from github import Auth, GithubIntegration, Github

class GithubAppClient:
    def __init__(self, config: Config):
        self.config = config
        self.app_id = config.GITHUB_APP_ID

        auth = Auth.AppAuth(self.app_id, self.config.GITHUB_PRIVATE_KEY)          
        self.integration = GithubIntegration(auth=auth)          
        
        self._token_cache = {}

    async def get_client(self, installation_id: int = None) -> Github:
        """
        Returns an authenticated Github client for the given installation.
        Utilizes caching to prevent unnecessary API calls and rate-limiting.
        Uses asyncio.to_thread to prevent PyGithub from blocking the FastAPI event loop.
        """
        target_id = installation_id or self.config.GITHUB_APP_INSTALLATION_ID
        now_utc = datetime.now(timezone.utc)

        if target_id in self._token_cache:
            cached = self._token_cache[target_id]
            safety_buffer = timedelta(minutes=5)
            
            if cached["expires_at"] > now_utc + safety_buffer:
                return cached["client"]

        installation = await asyncio.to_thread(self.integration.get_installation, target_id)
        token_auth = await asyncio.to_thread(installation.get_installation_auth)
        gh = Github(auth=token_auth)

        self._token_cache[target_id] = {
            "client": gh,
            "token": token_auth.token,
            "expires_at": token_auth.expires_at
        }
        
        return gh