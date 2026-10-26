from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass(frozen=True)
class Config:
    token: str
    owner: str
    owner_type: str  # "user" or "org"
    project_number: int
    repo: Optional[str] = None  # "owner/name" where `plan` creates issues

    @classmethod
    def from_env(cls) -> "Config":
        """Read settings from environment variables (and a local .env file)."""
        load_dotenv()

        missing = [
            name
            for name in ("GITHUB_TOKEN", "GITHUB_PROJECT_OWNER", "GITHUB_PROJECT_NUMBER")
            if not os.getenv(name)
        ]
        if missing:
            raise RuntimeError(
                "Missing required settings: " + ", ".join(missing)
                + ". Copy .env.example to .env and fill it in."
            )

        owner_type = os.getenv("GITHUB_PROJECT_OWNER_TYPE", "user").lower()
        if owner_type not in ("user", "org"):
            raise RuntimeError("GITHUB_PROJECT_OWNER_TYPE must be 'user' or 'org'.")

        return cls(
            token=os.environ["GITHUB_TOKEN"],
            owner=os.environ["GITHUB_PROJECT_OWNER"],
            owner_type=owner_type,
            project_number=int(os.environ["GITHUB_PROJECT_NUMBER"]),
            repo=os.getenv("GITHUB_REPO") or None,
        )
