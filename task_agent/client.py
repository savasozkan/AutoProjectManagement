from __future__ import annotations

from typing import Any, Dict, Optional

import requests

GRAPHQL_URL = "https://api.github.com/graphql"


class GitHubError(Exception):
    """Raised when the GitHub API returns an error."""


class GitHubClient:
    """Minimal wrapper around GitHub's GraphQL API.

    GitHub Projects (v2) are only available through GraphQL, not the REST API.
    """

    def __init__(self, token: str, timeout: float = 30.0):
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            }
        )
        self._timeout = timeout

    def graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        response = self._session.post(
            GRAPHQL_URL,
            json={"query": query, "variables": variables or {}},
            timeout=self._timeout,
        )
        if response.status_code == 401:
            raise GitHubError("Unauthorized: check that GITHUB_TOKEN is valid.")
        response.raise_for_status()

        payload = response.json()
        if payload.get("errors"):
            messages = "; ".join(e.get("message", str(e)) for e in payload["errors"])
            if "not accessible by personal access token" in messages:
                messages += (
                    ". Use a classic token (ghp_...) with the 'project' scope; fine-grained"
                    " tokens can't access projects owned by a personal account."
                )
            raise GitHubError(messages)
        return payload["data"]
