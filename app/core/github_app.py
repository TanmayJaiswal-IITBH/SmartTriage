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

    async def post_comment(self, installation_id: int, repo_name: str, number: int, body: str):
        """
        Posts a comment on an Issue or Pull Request.
        In GitHub's API, PRs are issues, so repo.get_issue works for both.
        """
        gh = await self.get_client(installation_id)
        
        repo = await asyncio.to_thread(gh.get_repo, repo_name)
        issue = await asyncio.to_thread(repo.get_issue, number)
        
        await asyncio.to_thread(issue.create_comment, body)

    async def close_item(self, installation_id: int, repo_name: str, number: int):
        """
        Closes an Issue or Pull Request.
        """
        gh = await self.get_client(installation_id)
        
        repo = await asyncio.to_thread(gh.get_repo, repo_name)
        issue = await asyncio.to_thread(repo.get_issue, number)
        
        await asyncio.to_thread(issue.edit, state="closed")

    async def get_pr_commit_messages(self, installation_id: int, repo_name: str, pr_number: int) -> str:
        """
        Fetches all commits for a specific PR and returns their messages concatenated into a single string.
        Safely isolates PyGithub's PaginatedList iteration to a background thread.
        """
        gh = await self.get_client(installation_id)
        
        repo = await asyncio.to_thread(gh.get_repo, repo_name)
        pr = await asyncio.to_thread(repo.get_pull, pr_number)
        
        # 1. Fetch the PaginatedList reference
        commits = await asyncio.to_thread(pr.get_commits)
        
        # 2. Define an internal helper to handle the blocking list iteration
        def _extract_messages():
            return " ".join([commit.commit.message for commit in commits])
            
        # 3. Execute the iteration in a background thread
        return await asyncio.to_thread(_extract_messages)