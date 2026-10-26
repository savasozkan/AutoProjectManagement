"""Offline tests for reading items from a project (no network needed)."""

from conftest import field_value, item_node

MASTER = item_node(
    "PVTI_1",
    "ISSUE",
    {"title": "Build blog", "url": "https://github.com/me/repo/issues/1", "number": 1,
     "state": "OPEN", "parent": None},
    [field_value("Title", text="Build blog"), field_value("Status", name="Todo")],
)
SUBTASK = item_node(
    "PVTI_2",
    "ISSUE",
    {"title": "Write RSS feed", "url": "https://github.com/me/repo/issues/2", "number": 2,
     "state": "CLOSED",
     "parent": {"number": 1, "title": "Build blog", "url": "https://github.com/me/repo/issues/1"}},
    [field_value("Status", name="Done"), field_value("Estimate", number=3.0)],
)
DRAFT = item_node(
    "PVTI_3",
    "DRAFT_ISSUE",
    {"title": "Pick a domain name"},
    [field_value("Status", name="Todo"), field_value("Due", date="2026-11-01")],
)
PULL_REQUEST = item_node(
    "PVTI_4",
    "PULL_REQUEST",
    {"title": "Add about page", "url": "https://github.com/me/repo/pull/5", "number": 5,
     "state": "MERGED"},
    # Unknown field-value types come back as {} and must be ignored.
    [{}, field_value("Status", name="Done")],
)
# Items you can't see (e.g. from a private repo) have no content.
REDACTED = item_node("PVTI_5", "REDACTED", None)


def test_reads_issue_fields(make_service):
    service, _ = make_service([[MASTER]])
    [item] = service.get_items()

    assert item.id == "PVTI_1"
    assert item.type == "ISSUE"
    assert item.title == "Build blog"
    assert item.number == 1
    assert item.state == "OPEN"
    assert item.parent is None
    assert item.status == "Todo"
    assert item.values == {"Title": "Build blog", "Status": "Todo"}


def test_reads_sub_issue_parent(make_service):
    service, _ = make_service([[MASTER, SUBTASK]])
    sub = service.get_items()[1]

    assert sub.parent == {"number": 1, "title": "Build blog",
                          "url": "https://github.com/me/repo/issues/1"}
    assert sub.state == "CLOSED"
    assert sub.values["Estimate"] == 3.0


def test_reads_drafts_pull_requests_and_redacted_items(make_service):
    service, _ = make_service([[DRAFT, PULL_REQUEST, REDACTED]])
    draft, pr, redacted = service.get_items()

    assert (draft.type, draft.title, draft.number, draft.url) == ("DRAFT_ISSUE", "Pick a domain name", None, None)
    assert draft.values["Due"] == "2026-11-01"
    assert (pr.type, pr.number, pr.state, pr.status) == ("PULL_REQUEST", 5, "MERGED", "Done")
    assert (redacted.type, redacted.title, redacted.status) == ("REDACTED", None, None)


def test_follows_pagination(make_service):
    service, fake = make_service([[MASTER, SUBTASK], [DRAFT], [PULL_REQUEST]])
    items = service.get_items()

    assert [i.id for i in items] == ["PVTI_1", "PVTI_2", "PVTI_3", "PVTI_4"]
    assert [r["after"] for r in fake.item_requests] == [None, "1", "2"]


def test_empty_project(make_service):
    service, _ = make_service([[]])
    assert service.get_items() == []


def test_filter_by_status_is_case_insensitive(make_service):
    service, _ = make_service([[MASTER, SUBTASK, DRAFT, PULL_REQUEST, REDACTED]])

    assert [i.id for i in service.get_items(status="todo")] == ["PVTI_1", "PVTI_3"]
    assert [i.id for i in service.get_items(status="Done")] == ["PVTI_2", "PVTI_4"]
    assert service.get_items(status="In Progress") == []
