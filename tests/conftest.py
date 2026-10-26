"""Shared test helpers: a fake GitHub client that returns canned GraphQL responses."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from task_agent import Config, ProjectService

PROJECT = {
    "id": "PVT_1",
    "title": "Test Project",
    "url": "https://github.com/users/me/projects/1",
    "fields": {
        "nodes": [
            {"id": "F_TITLE", "name": "Title", "dataType": "TITLE"},
            {
                "id": "F_STATUS",
                "name": "Status",
                "dataType": "SINGLE_SELECT",
                "options": [{"id": "o1", "name": "Todo"}, {"id": "o2", "name": "Done"}],
            },
        ]
    },
}


def field_value(field: str, **value: Any) -> Dict[str, Any]:
    """One entry of an item's fieldValues, e.g. field_value("Status", name="Todo")."""
    return {**value, "field": {"name": field}}


def item_node(
    item_id: str,
    item_type: str,
    content: Optional[Dict[str, Any]],
    values: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "id": item_id,
        "type": item_type,
        "content": content,
        "fieldValues": {"nodes": values or []},
    }


class FakeGitHub:
    """Stands in for GitHubClient. `pages` are the item pages returned in order."""

    def __init__(self, pages: List[List[Dict[str, Any]]]):
        self.pages = pages
        self.item_requests: List[Dict[str, Any]] = []

    def graphql(self, query: str, variables: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        variables = variables or {}
        if "projectV2(number" in query:
            return {"user": {"projectV2": PROJECT}}
        if "items(first" in query:
            self.item_requests.append(variables)
            index = int(variables["after"] or 0)
            has_next = index + 1 < len(self.pages)
            return {
                "node": {
                    "items": {
                        "nodes": self.pages[index],
                        "pageInfo": {
                            "hasNextPage": has_next,
                            "endCursor": str(index + 1) if has_next else None,
                        },
                    }
                }
            }
        raise AssertionError(f"Unexpected query in test:\n{query}")


@pytest.fixture
def make_service():
    """Build a ProjectService backed by FakeGitHub. Returns (service, fake)."""

    def build(pages: List[List[Dict[str, Any]]]):
        fake = FakeGitHub(pages)
        return ProjectService(fake, Config("token", "me", "user", 1)), fake

    return build
