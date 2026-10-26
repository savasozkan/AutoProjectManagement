"""Live test: reads the real project configured in .env. Read-only; it never changes anything.

Skipped when .env isn't filled in. Run only these with:  pytest -m live -s
Skip them with:                                          pytest -m "not live"
"""

import pytest

from task_agent import Config, ProjectService

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def service():
    try:
        config = Config.from_env()
    except RuntimeError as exc:
        pytest.skip(f"GitHub settings not configured: {exc}")
    from task_agent import GitHubClient

    return ProjectService(GitHubClient(config.token), config)


def test_connects_to_project(service):
    assert service.project_id
    assert service.title
    field_names = [f.name for f in service.fields()]
    assert "Title" in field_names


def test_retrieves_current_tasks(service):
    items = service.get_items()

    print(f"\n{service.title}: {len(items)} item(s)")
    for item in items:
        ref = f"#{item.number}" if item.number else "draft"
        parent = f"  (sub-issue of #{item.parent['number']})" if item.parent else ""
        print(f"  [{item.status or '-'}] {ref} {item.title}{parent}")

    for item in items:
        assert item.id.startswith("PVTI_")
        assert item.type in {"ISSUE", "PULL_REQUEST", "DRAFT_ISSUE", "REDACTED"}
        if item.type != "REDACTED":
            assert item.title
        if item.type in {"ISSUE", "PULL_REQUEST"}:
            assert item.number and item.url and item.state


def test_sub_issues_point_to_issues_in_project(service):
    items = service.get_items()
    numbers = {i.number for i in items if i.type == "ISSUE"}
    for item in items:
        if item.parent:
            # Masters created by task-agent are added to the project too.
            assert item.parent["number"] in numbers, (
                f"#{item.number}'s master issue #{item.parent['number']} isn't in the project"
            )


def test_status_filter_matches_full_list(service):
    items = service.get_items()
    for status in {i.status for i in items if i.status}:
        filtered = service.get_items(status=status)
        assert len(filtered) == sum(1 for i in items if i.status == status)
