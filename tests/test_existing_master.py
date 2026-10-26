"""Offline tests: a ticked existing master gets the new subtasks instead of a new master."""

from conftest import field_value, item_node

from task_agent.planner import Subtask
from task_agent.projects import ProjectItem, split_selection


def issue(number, title, parent=None, subs=0):
    return ProjectItem(
        id=f"PVTI_{number}", type="ISSUE", title=title, url=f"https://x/{number}",
        values={"Status": "Todo"}, number=number, state="OPEN",
        parent={"number": parent, "title": "", "url": ""} if parent else None,
        body=f"body {number}", content_id=f"I_{number}", sub_issue_count=subs,
    )


MASTER = issue(31, "Mobile ordering app", subs=2)
UI = issue(32, "Build mobile app UI", parent=31)
DASH = issue(33, "Implement kitchen dashboard", parent=31)
OTHER_MASTER = issue(40, "Blog", subs=1)
BLOG_POST = issue(41, "Write first post", parent=40)
LOOSE = issue(50, "Set up CI")
ALL = [MASTER, UI, DASH, OTHER_MASTER, BLOG_POST, LOOSE]


def test_reads_master_details_from_github(make_service):
    node = item_node(
        "PVTI_31", "ISSUE",
        {"id": "I_31", "title": "Mobile ordering app", "url": "u", "number": 31, "state": "OPEN",
         "parent": None, "body": "b", "subIssuesSummary": {"total": 3, "completed": 1}},
        [field_value("Status", name="Todo")],
    )
    draft = item_node("PVTI_9", "DRAFT_ISSUE", {"title": "Draft"})
    service, _ = make_service([[node, draft]])
    master, draft_item = service.get_items()

    assert (master.content_id, master.sub_issue_count, master.is_master) == ("I_31", 3, True)
    assert (draft_item.sub_issue_count, draft_item.is_master) == (0, False)


def test_no_master_ticked():
    master, covered = split_selection([LOOSE, UI], ALL)
    assert master is None
    assert [i.number for i in covered] == [50, 32]


def test_ticked_master_is_used_and_its_sub_issues_count_as_covered():
    master, covered = split_selection([MASTER], ALL)
    assert master is MASTER
    # The master itself is not "covered work": its description is the whole job.
    assert [i.number for i in covered] == [32, 33]


def test_master_plus_its_own_sub_issue_is_not_duplicated():
    master, covered = split_selection([UI, MASTER, LOOSE], ALL)
    assert master is MASTER
    assert [i.number for i in covered] == [32, 33, 50]


def test_first_ticked_master_wins():
    master, covered = split_selection([OTHER_MASTER, MASTER], ALL)
    assert master is OTHER_MASTER
    assert sorted(i.number for i in covered) == [32, 33, 41]


class CreateRecorder:
    def __init__(self):
        self.calls = []
        self.n = 100

    def graphql(self, query, variables=None):
        if "projectV2(number" in query:
            return {"user": {"projectV2": {"id": "P", "title": "T", "url": "u", "fields": {"nodes": []}}}}
        if "repository(owner" in query:
            return {"repository": {"id": "R"}}
        if "createIssue" in query:
            self.n += 1
            self.calls.append(("create", variables["title"], variables["parentIssueId"]))
            return {"createIssue": {"issue": {"id": f"I_{self.n}", "number": self.n, "url": f"x/{self.n}"}}}
        if "addProjectV2ItemById" in query:
            self.calls.append(("add", variables["contentId"]))
            return {"addProjectV2ItemById": {"item": {"id": "PVTI_" + variables["contentId"]}}}
        raise AssertionError(query)


def test_add_to_master_creates_only_sub_issues():
    from task_agent import Config, ProjectService

    fake = CreateRecorder()
    service = ProjectService(fake, Config("t", "me", "user", 1))
    created = service.add_to_master(
        "I_31", [Subtask("Add payments", "d"), Subtask("Add receipts", "d")], "me/repo"
    )

    assert [c.number for c in created] == [101, 102]
    assert fake.calls == [
        ("create", "Add payments", "I_31"), ("add", "I_101"),
        ("create", "Add receipts", "I_31"), ("add", "I_102"),
    ]  # no new master issue
