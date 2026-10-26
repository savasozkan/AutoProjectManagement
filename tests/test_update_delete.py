"""Offline tests for [UPDATE] / [DELETE] commands."""

import pytest

from task_agent import workflow
from task_agent.client import GitHubError
from task_agent.operations import parse_instructions, resolve_commands
from task_agent.planner import Plan, Subtask
from task_agent.projects import ProjectItem


def issue(number, title, parent=None, subs=0, item_type="ISSUE"):
    return ProjectItem(
        id=f"PVTI_{number}", type=item_type, title=title, url=f"https://x/{number}",
        values={"Status": "Todo"}, number=number if item_type != "DRAFT_ISSUE" else None,
        state="OPEN", parent={"number": parent} if parent else None,
        body=f"old body {number}", content_id=f"C_{number}", sub_issue_count=subs,
    )


MASTER = issue(31, "Mobile ordering app", subs=2)
UI = issue(32, "Build mobile app UI", parent=31)
DASH = issue(33, "Implement kitchen dashboard", parent=31)
LINK = issue(34, "Connect mobile app to kitchen dashboard", parent=31)
DRAFT = issue(90, "Pick a logo", item_type="DRAFT_ISSUE")
DONE = issue(35, "Set up database", parent=31)
DONE.values = {"Status": "Done"}
CLOSED = issue(36, "Old API spike")
CLOSED.state = "CLOSED"
ITEMS = [MASTER, UI, DASH, LINK, DRAFT, DONE, CLOSED]
ALL_IDS = {i.id for i in ITEMS}


# -- parsing ----------------------------------------------------------------

def test_commands_are_split_from_plain_instructions():
    parsed = parse_instructions(
        "Focus on the backend.\n[UPDATE] Implement kitchen dashboard\n- [delete] #34\n"
        "[DELETE] \"Pick a logo\"  [UPDATE] #32\nKeep it short."
    )
    assert parsed.instructions == "Focus on the backend.\nKeep it short."
    assert [(c.action, c.target) for c in parsed.commands] == [
        ("UPDATE", "Implement kitchen dashboard"),
        ("DELETE", "#34"),
        ("DELETE", "Pick a logo"),
        ("UPDATE", "#32"),
    ]


def test_no_commands():
    parsed = parse_instructions("Just focus on the API.")
    assert parsed.instructions == "Just focus on the API." and parsed.commands == []
    assert parse_instructions(None).commands == []


def test_tasks_found_by_number_title_typo_or_not_at_all():
    commands = parse_instructions(
        "[UPDATE] #33\n[UPDATE] build MOBILE app ui\n[UPDATE] kitchen dashbord\n[DELETE] #99\n[DELETE] Write docs"
    ).commands
    found = resolve_commands(commands, ITEMS)
    assert found[0].item is DASH and found[0].note == ""
    assert found[1].item is UI and found[1].note == ""
    assert found[2].item is DASH and "Closest match" in found[2].note
    assert found[3].item is None and "#99" in found[3].note
    assert found[4].item is None and "No task matching" in found[4].note


def test_ambiguous_title_asks_for_number():
    twins = [issue(1, "Write tests"), issue(2, "Write tests")]
    [r] = resolve_commands(parse_instructions("[DELETE] Write tests").commands, twins)
    assert r.item is None and "Use the number" in r.note


# -- proposing --------------------------------------------------------------

class FakePlanner:
    model = "fake"

    def __init__(self):
        self.plan_calls = []
        self.rewrites = []

    def rewrite_task(self, document, title, description="", instructions=None):
        self.rewrites.append((title, instructions))
        return Subtask(title, f"NEW: {title} ({instructions})")

    def plan(self, description, title, min_tasks, max_tasks, instructions=None, existing=None):
        self.plan_calls.append({"title": title, "instructions": instructions, "existing": existing,
                                "min_tasks": min_tasks})
        return Plan(title or "Master", description, [Subtask("Add payments", "d")])


def test_proposal_contains_updates_deletions_and_new_tasks():
    planner = FakePlanner()
    proposal = workflow.propose(
        "Document text", ITEMS, selected_ids={DASH.id, LINK.id},
        prompt="Use WebSockets.\n[UPDATE] #33\n[DELETE] #34", planner=planner,
    )
    assert [u.item.number for u in proposal.updates] == [33]
    assert proposal.updates[0].new.description == "NEW: Implement kitchen dashboard (Use WebSockets.)"
    assert proposal.deletions == [LINK]
    assert [t.title for t in proposal.plan.tasks] == ["Add payments"]
    # Commands are not passed on as planning instructions...
    assert planner.plan_calls[0]["instructions"] == "Use WebSockets."
    # ...and updated / deleted tasks are not generated again.
    existing = [e.title for e in planner.plan_calls[0]["existing"]]
    assert "Implement kitchen dashboard" in existing
    # The updated task is described to the planner by its new description.
    dash = next(e for e in planner.plan_calls[0]["existing"] if e.number == 33)
    assert dash.description.startswith("NEW:")
    assert proposal.warnings == []
    assert any("Connect mobile app" in t and "being removed" in t for t in existing)


def test_only_commands_without_new_tasks():
    planner = FakePlanner()
    proposal = workflow.propose("Doc", ITEMS, selected_ids={UI.id}, prompt="[DELETE] #32",
                                generate_new=False, planner=planner)
    assert proposal.plan.tasks == [] and planner.plan_calls == []
    assert proposal.deletions == [UI]


def test_conflicts_and_unknown_tasks_become_warnings():
    planner = FakePlanner()
    proposal = workflow.propose(
        "Doc", ITEMS, selected_ids={MASTER.id, DASH.id},
        prompt="[UPDATE] #33\n[DELETE] #33\n[DELETE] #31\n[UPDATE] #404", planner=planner,
    )
    assert proposal.updates == []  # #33 is only deleted
    assert proposal.deletions == [DASH, MASTER]
    assert proposal.master is None  # the ticked master is being deleted
    text = " ".join(proposal.warnings)
    assert "both update and delete" in text
    assert "a new master will be created" in text
    assert "lose their master" in text
    assert "#404" in text


# -- applying ---------------------------------------------------------------

class FakeService:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def update_task(self, item, title, body):
        if item.number in self.fail_on:
            raise GitHubError("boom")
        self.calls.append(("update", item.number, title, body))

    def remove_task(self, item, mode):
        self.calls.append(("remove", item.number, mode))

    def add_to_master(self, master_id, tasks, repo, fields):
        self.calls.append(("add_to_master", master_id, [t.title for t in tasks]))
        return []


def test_apply_runs_every_change_and_keeps_going_after_errors():
    svc = FakeService(fail_on={32})
    result = workflow.apply(
        svc, "me/repo",
        new_tasks=[Subtask("Add payments", "d")], master=MASTER,
        updates=[workflow.Update(UI, Subtask("Build mobile app UI", "x")),
                 workflow.Update(DASH, Subtask("Build live dashboard", "y"))],
        deletions=[(LINK, "close")],
    )
    assert svc.calls == [
        ("add_to_master", "C_31", ["Add payments"]),
        ("update", 33, "Build live dashboard", "y"),  # title changed -> sent
        ("remove", 34, "close"),
    ]
    assert result.reused_master and [i.number for i in result.updated] == [33]
    assert len(result.errors) == 1 and "#32" in result.errors[0]


def test_unchanged_title_is_not_sent():
    svc = FakeService()
    workflow.apply(svc, None, updates=[workflow.Update(DASH, Subtask(DASH.title, "new"))])
    assert svc.calls == [("update", 33, None, "new")]


# -- GitHub calls ----------------------------------------------------------

class Recorder:
    def __init__(self):
        self.calls = []

    def graphql(self, query, variables=None):
        if "projectV2(number" in query:
            return {"user": {"projectV2": {"id": "P", "title": "T", "url": "u", "fields": {"nodes": []}}}}
        name = query.split("{")[1].split("(")[0].strip()
        self.calls.append((name, variables))
        return {}


@pytest.fixture
def service():
    from task_agent import Config, ProjectService
    rec = Recorder()
    return ProjectService(rec, Config("t", "me", "user", 1)), rec


def test_update_uses_the_right_mutation_per_type(service):
    svc, rec = service
    svc.update_task(DASH, "New title", "New body")
    svc.update_task(DRAFT, None, "Draft body")
    assert rec.calls[0] == ("updateIssue", {"id": "C_33", "title": "New title", "body": "New body"})
    assert rec.calls[1][0] == "updateProjectV2DraftIssue"


def test_remove_modes(service):
    svc, rec = service
    svc.remove_task(DASH, "delete")
    svc.remove_task(DASH, "close")
    svc.remove_task(DASH, "remove")
    svc.remove_task(DRAFT, "delete")  # drafts can only be removed from the project
    assert [c[0] for c in rec.calls] == [
        "deleteIssue", "closeIssue", "deleteProjectV2Item", "deleteProjectV2Item",
    ]
    assert rec.calls[0][1] == {"id": "C_33"}
    assert rec.calls[2][1] == {"projectId": "P", "itemId": "PVTI_33"}


def test_nothing_new_is_only_allowed_when_tasks_are_ticked_as_covered():
    planner = FakePlanner()
    workflow.propose("Doc", ITEMS, prompt="[UPDATE] #33", planner=planner)
    assert planner.plan_calls[-1]["min_tasks"] == 3  # commands alone don't cover the document
    workflow.propose("Doc", ITEMS, selected_ids={MASTER.id}, prompt="[UPDATE] #33", planner=planner)
    assert planner.plan_calls[-1]["min_tasks"] == 0  # ticked master: its sub-issues may cover it all


def test_unticked_tasks_are_not_changed():
    planner = FakePlanner()
    proposal = workflow.propose(
        "Doc", ITEMS, selected_ids={UI.id}, prompt="[UPDATE] #33\n[DELETE] #34\n[DELETE] #32",
        planner=planner,
    )
    assert proposal.updates == [] and planner.rewrites == []
    assert proposal.deletions == [UI]  # only the ticked one
    assert sum("isn't ticked" in w for w in proposal.warnings) == 2


def test_done_tasks_are_not_changed_even_when_ticked():
    planner = FakePlanner()
    proposal = workflow.propose(
        "Doc", ITEMS, selected_ids=ALL_IDS,
        prompt="[UPDATE] Set up database\n[DELETE] #36\n[UPDATE] #33", planner=planner,
    )
    assert [u.item.number for u in proposal.updates] == [33]
    assert proposal.deletions == []
    done_warnings = [w for w in proposal.warnings if "already done" in w]
    assert len(done_warnings) == 2
    assert any("#35" in w for w in done_warnings) and any("#36" in w for w in done_warnings)


def test_done_check_is_repeated_when_applying():
    svc = FakeService()
    result = workflow.apply(
        svc, None,
        updates=[workflow.Update(DONE, Subtask("x", "y"))], deletions=[(CLOSED, "delete")],
    )
    assert svc.calls == []
    assert len(result.errors) == 2 and all("already done" in e for e in result.errors)
