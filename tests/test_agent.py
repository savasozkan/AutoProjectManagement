"""Offline tests for the conversational agent and its special tokens."""

from task_agent.agent import Agent, Session, parse_message
from task_agent.documents import Document
from task_agent.planner import Plan, Subtask
from task_agent.projects import CreatedIssue, ProjectItem
from task_agent.similarity import Match


def issue(number, title, parent=None, subs=0, status="Todo"):
    return ProjectItem(
        id=f"PVTI_{number}", type="ISSUE", title=title, url=f"https://x/{number}",
        values={"Status": status}, number=number, state="OPEN",
        parent={"number": parent} if parent else None, body=f"body {number}",
        content_id=f"C_{number}", sub_issue_count=subs,
    )


MASTER = issue(1, "Food ordering app", subs=2)
API = issue(2, "Implement REST API", parent=1)
DASH = issue(3, "Build kitchen dashboard", parent=1)
BLOG = issue(9, "Write blog post")
ITEMS = [MASTER, API, DASH, BLOG]


class FakeConfig:
    repo = "me/repo"


class FakeService:
    config = FakeConfig()

    def __init__(self, items=ITEMS):
        self.items = list(items)
        self.created = []

    def get_items(self):
        return list(self.items)

    def add_to_master(self, master_id, tasks, repo, fields):
        self.created.append(("under", master_id, [t.title for t in tasks]))
        return [CreatedIssue(f"I{n}", 100 + n, f"https://x/{100 + n}", f"PVTI_{100 + n}") for n, _ in enumerate(tasks)]

    def add_plan(self, plan, repo, fields):
        self.created.append(("new", plan.master_title, [t.title for t in plan.tasks]))
        master = CreatedIssue("IM", 99, "https://x/99", "PVTI_99")
        return master, [CreatedIssue(f"I{n}", 100 + n, f"https://x/{100 + n}", f"PVTI_{100 + n}")
                        for n, _ in enumerate(plan.tasks)]


class FakeFinder:
    method = "keywords"
    queries = []

    def __init__(self, log=None):
        pass

    def find(self, text, items, top_k=15, min_score=0.0):
        FakeFinder.queries.append(text)
        scores = {"PVTI_1": 0.8, "PVTI_2": 0.5, "PVTI_3": 0.3, "PVTI_9": 0.01}
        ranked = sorted((Match(i, scores[i.id]) for i in items), key=lambda m: -m.score)
        return ranked[:top_k]


class FakePlanner:
    model = "fake"
    calls = []

    def plan(self, description, title, min_tasks, max_tasks, instructions=None, existing=None):
        FakePlanner.calls.append({"instructions": instructions, "existing": [e.title for e in existing or []],
                                  "title": title})
        return Plan(title or "Master", description, [Subtask("Add payments", "d1"), Subtask("Add receipts", "d2")])

    def rewrite_task(self, document, title, description="", instructions=None):
        return Subtask(title, "rewritten")


def make_agent(service=None):
    FakeFinder.queries.clear()
    FakePlanner.calls.clear()
    return Agent(service or FakeService(), planner_factory=FakePlanner, finder_factory=FakeFinder)


def loaded_session(agent):
    session = Session()
    agent.document_loaded(session, "spec.pdf", Document("Food ordering app with payments.", "Food app"))
    return session


# -- parsing ----------------------------------------------------------------

def test_parse_message_splits_tokens_in_order():
    assert parse_message("focus on API [search] payments [TASK] keep it short [ADD]") == [
        (None, "focus on API"), ("SEARCH", "payments"), ("TASK", "keep it short"), ("ADD", ""),
    ]
    assert parse_message("") == []
    # [UPDATE]/[DELETE] are not agent tokens: they stay inside the [TASK] guidance.
    assert parse_message("[TASK] [UPDATE] #3") == [("TASK", "[UPDATE] #3")]


# -- commands ---------------------------------------------------------------

def test_commands_need_a_document_or_a_proposal():
    agent = make_agent()
    session = Session()
    [r] = agent.handle(session, "[TASK]")
    assert r["type"] == "error" and "Load a document" in r["text"]
    [r] = agent.handle(session, "[SEARCH]")
    assert r["type"] == "error"
    [r] = agent.handle(session, "[ADD]")
    assert r["type"] == "error" and "[TASK]" in r["text"]


def test_search_lists_all_tasks_sorted_by_similarity():
    agent = make_agent()
    session = loaded_session(agent)
    [r] = agent.handle(session, "[SEARCH]")
    assert r["type"] == "search"
    assert FakeFinder.queries[-1].startswith("Food ordering app")  # the document
    # Every task is listed, most similar first, including the weak #9.
    assert [x["number"] for x in r["results"]] == [1, 2, 3, 9]
    assert [x["strength"] for x in r["results"]] == ["strong", "strong", "strong", "weak"]
    assert "3 of the 4 task(s)" in r["text"]
    assert r["results"][0]["sub_issue_count"] == 2

    agent.handle(session, "[SEARCH] kitchen dashboard")
    assert FakeFinder.queries[-1] == "kitchen dashboard"


def test_search_marks_already_selected_tasks():
    agent = make_agent()
    session = loaded_session(agent)
    [r] = agent.handle(session, "[SEARCH]", selected_ids=["PVTI_2"])
    assert [x["selected"] for x in r["results"]] == [False, True, False, False]


def test_tasks_skip_selected_work_and_use_notes_and_guidance():
    agent = make_agent()
    session = loaded_session(agent)
    [note] = agent.handle(session, "Customers pay by card")
    assert note["type"] == "text" and "Noted" in note["text"]

    [r] = agent.handle(session, "[TASK] keep it short", selected_ids=["PVTI_2"])
    assert r["type"] == "proposal"
    assert [t["title"] for t in r["tasks"]] == ["Add payments", "Add receipts"]
    call = FakePlanner.calls[-1]
    assert call["instructions"] == "Customers pay by card\nkeep it short"
    assert call["existing"] == ["Implement REST API"]
    assert r["master"] is None and session.proposal is not None


def test_selecting_a_master_puts_new_tasks_under_it():
    agent = make_agent()
    session = loaded_session(agent)
    [r] = agent.handle(session, "[TASK]", selected_ids=["PVTI_1"])
    assert r["master"]["number"] == 1
    # The master's sub-issues count as covered.
    assert FakePlanner.calls[-1]["existing"] == ["Implement REST API", "Build kitchen dashboard"]


def test_add_uses_edits_and_can_only_run_once():
    service = FakeService()
    agent = make_agent(service)
    session = loaded_session(agent)
    [p] = agent.handle(session, "[TASK]", selected_ids=["PVTI_1"])
    edits = {
        "proposal_id": p["proposal_id"], "use_existing_master": True,
        "tasks": [{"title": "Add card payments", "description": "edited"}],  # one unticked, one renamed
    }
    [r] = agent.handle(session, "[ADD]", edits=edits)
    assert r["type"] == "added"
    assert service.created == [("under", "C_1", ["Add card payments"])]
    assert [s["title"] for s in r["subtasks"]] == ["Add card payments"]
    [again] = agent.handle(session, "[ADD]", edits=edits)
    assert again["type"] == "error"


def test_add_with_a_new_master_and_without_edits():
    service = FakeService()
    agent = make_agent(service)
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[ADD]")
    assert service.created == [("new", "Food app", ["Add payments", "Add receipts"])]
    assert r["master"]["number"] == 99


def test_edits_for_an_old_proposal_are_rejected():
    agent = make_agent()
    session = loaded_session(agent)
    [old] = agent.handle(session, "[TASK]")
    agent.handle(session, "[TASK] again")
    [r] = agent.handle(session, "[ADD]", edits={"proposal_id": old["proposal_id"], "tasks": []})
    assert r["type"] == "error" and "older" in r["text"]


def test_several_commands_in_one_message_run_in_order():
    service = FakeService()
    agent = make_agent(service)
    session = loaded_session(agent)
    replies = agent.handle(session, "[SEARCH] [TASK] [ADD]", selected_ids=["PVTI_1"])
    assert [r["type"] for r in replies] == ["search", "proposal", "added"]
    assert service.created[0][0] == "under"


def test_new_document_keeps_earlier_documents_tasks_and_notes():
    agent = make_agent()
    session = loaded_session(agent)
    agent.handle(session, "focus on payments")
    agent.handle(session, "[TASK]")
    [r] = agent.document_loaded(session, "update.pdf", Document("Kitchen screen is no longer needed."))
    assert session.proposal is not None and session.notes == ["focus on payments"]
    assert [d["name"] for d in session.documents] == ["spec.pdf", "update.pdf"]
    assert "update to spec.pdf" in r["text"] and "[TASK]" in r["text"]
    assert "Document 2 (update, received later): update.pdf" in session.document_text


def test_help():
    [r] = make_agent().handle(Session(), "[help]")
    assert all(f"[{c}]" in r["text"] for c in ("SEARCH", "TASK", "ADD"))


# -- refining with a second [TASK] -------------------------------------------

from task_agent.similarity import SimilarityFinder


class CountingPlanner(FakePlanner):
    def plan(self, description, title, min_tasks, max_tasks, instructions=None, existing=None):
        super().plan(description, title, min_tasks, max_tasks, instructions, existing)
        return Plan(title or "Master", description, [
            Subtask("Implement REST API endpoints", "Create endpoints for orders."),
            Subtask("Add payments", "Let customers pay by card."),
            Subtask("Write onboarding guide", "Explain the app to new users."),
        ])


def refine_agent(service=None):
    FakePlanner.calls.clear()
    return Agent(service or FakeService(), planner_factory=CountingPlanner,
                 finder_factory=lambda log=None: SimilarityFinder(use_embeddings=False))


def test_second_tasks_only_removes_overlaps_without_generating():
    agent = refine_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    assert len(FakePlanner.calls) == 1

    [r] = agent.handle(session, "[TASK]", selected_ids=["PVTI_2"])  # tick "Implement REST API"
    assert len(FakePlanner.calls) == 1  # nothing regenerated
    assert [t["title"] for t in r["tasks"]] == ["Add payments", "Write onboarding guide"]
    assert [(x["title"], x["overlaps"]["number"]) for x in r["removed"]] == [("Implement REST API endpoints", 2)]
    assert "no new tasks were generated" in r["text"]


def test_refine_keeps_everything_when_nothing_overlaps():
    agent = refine_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[TASK]", selected_ids=["PVTI_9"])  # "Write blog post"
    assert len(r["tasks"]) == 3 and r["removed"] == []
    assert len(FakePlanner.calls) == 1


def test_refine_works_on_the_users_edits():
    agent = refine_agent()
    session = loaded_session(agent)
    [p] = agent.handle(session, "[TASK]")
    edits = {"proposal_id": p["proposal_id"], "tasks": [
        {"title": "Add payments", "description": "Card and Apple Pay."},  # edited; others removed by the user
        {"title": "Implement REST API endpoints", "description": "Create endpoints for orders."},
    ]}
    [r] = agent.handle(session, "[TASK]", selected_ids=["PVTI_2"], edits=edits)
    assert r["tasks"] == [{"title": "Add payments", "description": "Card and Apple Pay."}]


def test_ticking_a_master_moves_tasks_under_it_and_checks_its_sub_issues():
    agent = refine_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[TASK]", selected_ids=["PVTI_1"])
    assert r["master"]["number"] == 1
    assert [x["overlaps"]["number"] for x in r["removed"]] == [2]  # #2 is a sub-issue of #1


def test_regenerate_starts_over_with_guidance():
    agent = refine_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    agent.handle(session, "[TASK] regenerate: focus on payments", selected_ids=["PVTI_2"])
    assert len(FakePlanner.calls) == 2
    assert FakePlanner.calls[-1]["instructions"] == "focus on payments"
    assert FakePlanner.calls[-1]["existing"] == ["Implement REST API"]


def test_refine_then_add_in_one_message_adds_the_refined_tasks():
    service = FakeService()
    agent = refine_agent(service)
    session = loaded_session(agent)
    [p] = agent.handle(session, "[TASK]")
    edits = {"proposal_id": p["proposal_id"], "tasks": [{"title": t["title"], "description": t["description"]} for t in p["tasks"]]}
    replies = agent.handle(session, "[TASK] [ADD]", selected_ids=["PVTI_2"], edits=edits)
    assert [r["type"] for r in replies] == ["proposal", "added"]
    assert service.created == [("new", "Food app", ["Add payments", "Write onboarding guide"])]


# -- [MERGE] ---------------------------------------------------------------------

from task_agent.agent import parse_merge


class MergePlanner(CountingPlanner):
    merges = []
    picks = []

    def merge_tasks(self, tasks, guidance=None):
        MergePlanner.merges.append(([t.title for t in tasks], guidance))
        return Subtask("Merged: " + " + ".join(t.title for t in tasks), "combined")

    def pick_tasks(self, tasks, request):
        MergePlanner.picks.append(request)
        return [0, 2]


def merge_agent(service=None):
    FakePlanner.calls.clear()
    MergePlanner.merges.clear()
    MergePlanner.picks.clear()
    return Agent(service or FakeService(), planner_factory=MergePlanner,
                 finder_factory=lambda log=None: SimilarityFinder(use_embeddings=False))


def test_parse_merge():
    assert parse_merge("2, 3") == ([2, 3], "")
    assert parse_merge("#1 and #3: one API task") == ([1, 3], "one API task")
    assert parse_merge("the API tasks") == ([], "the API tasks")


def test_merge_by_number_replaces_the_tasks_in_place():
    agent = merge_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")  # 1 REST API, 2 payments, 3 onboarding
    [r] = agent.handle(session, "[MERGE] 1, 3: keep it short")
    assert r["type"] == "proposal"
    assert [t["title"] for t in r["tasks"]] == [
        "Merged: Implement REST API endpoints + Write onboarding guide", "Add payments"]
    assert MergePlanner.merges == [(["Implement REST API endpoints", "Write onboarding guide"], "keep it short")]
    assert r["merged"]["position"] == 1 and len(r["merged"]["from"]) == 2
    assert len(FakePlanner.calls) == 1  # nothing regenerated


def test_merge_by_description_asks_the_model_which_tasks():
    agent = merge_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[MERGE] the API and the guide")
    assert MergePlanner.picks == ["the API and the guide"]
    assert len(r["tasks"]) == 2


def test_merge_uses_the_users_edits():
    agent = merge_agent()
    session = loaded_session(agent)
    [p] = agent.handle(session, "[TASK]")
    edits = {"proposal_id": p["proposal_id"], "tasks": [
        {"title": "A", "description": ""}, {"title": "B", "description": ""}]}
    [r] = agent.handle(session, "[MERGE] 1 2", edits=edits)
    assert [t["title"] for t in r["tasks"]] == ["Merged: A + B"]


def test_merge_errors():
    agent = merge_agent()
    session = loaded_session(agent)
    [r] = agent.handle(session, "[MERGE] 1, 2")
    assert r["type"] == "error" and "[TASK]" in r["text"]  # nothing generated yet
    agent.handle(session, "[TASK]")
    for text, expect in [("[MERGE] 2", "at least two"), ("[MERGE] 1, 9", "no task 9"), ("[MERGE]", "at least two")]:
        [r] = agent.handle(session, text)
        assert r["type"] == "error" and expect in r["text"], (text, r["text"])
    assert len(session.proposal.plan.tasks) == 3  # unchanged after errors


def test_merge_then_add_in_one_message():
    service = FakeService()
    agent = merge_agent(service)
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    replies = agent.handle(session, "[MERGE] 2, 3 [ADD]")
    assert [r["type"] for r in replies] == ["proposal", "added"]
    assert service.created[0][2] == ["Implement REST API endpoints", "Merged: Add payments + Write onboarding guide"]


# -- [SPLIT] ---------------------------------------------------------------------

class SplitPlanner(CountingPlanner):
    """Generated list: 1 REST API endpoints, 2 payments, 3 onboarding guide."""
    calls = []
    parts = {
        "Add payments": [Subtask("Add card payments", "Cards via Stripe."),
                         Subtask("Add wallet payments", "Apple Pay and Google Pay.")],
        "Write onboarding guide": [Subtask("Write onboarding screens", "First-run screens."),
                                   Subtask("Implement REST API endpoints", "Create endpoints for orders.")],  # repeats task 1
        "Implement REST API endpoints": [Subtask("Implement REST API endpoints", "Create endpoints for orders."),
                                         Subtask("Write onboarding guide", "Explain the app to new users.")],
    }

    def split_task(self, task, guidance=None, others=(), document=None):
        SplitPlanner.calls.append({"task": task.title, "guidance": guidance, "others": list(others)})
        return list(SplitPlanner.parts[task.title])

    def pick_tasks(self, tasks, request, action="merge"):
        return [1]


def split_agent(service=None):
    FakePlanner.calls.clear()
    SplitPlanner.calls.clear()
    return Agent(service or FakeService(), planner_factory=SplitPlanner,
                 finder_factory=lambda log=None: SimilarityFinder(use_embeddings=False))


def titles(reply):
    return [t["title"] for t in reply["tasks"]]


def test_split_by_number_replaces_the_task_in_place():
    agent = split_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[SPLIT] 2: by payment method")
    assert titles(r) == ["Implement REST API endpoints", "Add card payments", "Add wallet payments", "Write onboarding guide"]
    call = SplitPlanner.calls[0]
    assert call["guidance"] == "by payment method"
    assert len(call["others"]) == 2 and not any("payments" in o for o in call["others"])  # the other tasks
    assert r["split"] == [{"from": "Add payments", "position": 2, "parts": ["Add card payments", "Add wallet payments"],
                           "dropped": [], "split": True}]
    assert len(FakePlanner.calls) == 1  # nothing regenerated


def test_split_by_description():
    agent = split_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[SPLIT] the payments task")
    assert SplitPlanner.calls[0]["guidance"] == "the payments task"
    assert len(r["tasks"]) == 4


def test_parts_that_repeat_another_task_are_left_out_and_the_task_is_kept_if_too_few_remain():
    agent = split_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    # Task 3's second part repeats task 1: only one part would remain, so task 3 is kept as it was.
    [r] = agent.handle(session, "[SPLIT] 2, 3")
    assert titles(r) == ["Implement REST API endpoints", "Add card payments", "Add wallet payments", "Write onboarding guide"]
    kept = [x for x in r["split"] if not x["split"]][0]
    assert kept["from"] == "Write onboarding guide"
    assert kept["dropped"][0]["overlaps"]["label"] == "task 1"
    assert "couldn't be split" in r["text"]


def test_split_that_only_overlaps_is_an_error_and_changes_nothing():
    agent = split_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    [r] = agent.handle(session, "[SPLIT] 1")  # both parts repeat tasks 1 (itself excluded) and 3
    assert r["type"] == "error"
    assert len(session.proposal.plan.tasks) == 3


def test_split_checks_ticked_project_tasks_too():
    SplitPlanner.parts["Add payments"].append(Subtask("Build kitchen dashboard", "Show orders to staff."))
    try:
        agent = split_agent()
        session = loaded_session(agent)
        agent.handle(session, "[TASK]")
        [r] = agent.handle(session, "[SPLIT] 2", selected_ids=["PVTI_3"])  # #3 Build kitchen dashboard
        assert "Build kitchen dashboard" not in titles(r)
        assert r["split"][0]["dropped"][0]["overlaps"]["label"] == "#3"
    finally:
        SplitPlanner.parts["Add payments"].pop()


def test_split_errors():
    agent = split_agent()
    session = loaded_session(agent)
    [r] = agent.handle(session, "[SPLIT] 1")
    assert r["type"] == "error" and "[TASK]" in r["text"]
    agent.handle(session, "[TASK]")
    for text in ["[SPLIT]", "[SPLIT] 7"]:
        [r] = agent.handle(session, text)
        assert r["type"] == "error", text



# -- several documents: scope first, updates later --------------------------------

from task_agent.planner import Revision


class RevisePlanner(CountingPlanner):
    revisions = []

    def revise_tasks(self, tasks, earlier, newer, guidance=None):
        RevisePlanner.revisions.append({"tasks": [t.title for t in tasks], "earlier": earlier,
                                        "newer": newer, "guidance": guidance})
        # Update: drop the onboarding guide, change payments, add a courier dashboard.
        return Revision(
            [tasks[0], Subtask("Add card and wallet payments", "changed"), Subtask("Build courier dashboard", "new")],
            ["kept", "changed", "new"],
            [(tasks[2], "The update says onboarding is out of scope.")],
        )


def revise_agent(service=None):
    FakePlanner.calls.clear()
    RevisePlanner.revisions.clear()
    return Agent(service or FakeService(), planner_factory=RevisePlanner,
                 finder_factory=lambda log=None: SimilarityFinder(use_embeddings=False))


def test_tasks_after_a_new_document_revise_the_existing_tasks():
    agent = revise_agent()
    session = loaded_session(agent)
    [p] = agent.handle(session, "[TASK]")
    edits = {"proposal_id": p["proposal_id"], "tasks": [
        {"title": t["title"], "description": t["description"]} for t in p["tasks"]]}
    edits["tasks"][0]["title"] = "Implement REST API endpoints (v2)"  # a user edit must survive
    agent.document_loaded(session, "update.pdf", Document("Couriers need a dashboard. No onboarding."))

    [r] = agent.handle(session, "[TASK] keep it short", edits=edits)
    assert len(FakePlanner.calls) == 1  # not regenerated from scratch
    call = RevisePlanner.revisions[0]
    assert call["tasks"][0] == "Implement REST API endpoints (v2)"  # revised the user's version
    assert "Food ordering app" in call["earlier"] and "Couriers" in call["newer"]
    assert call["guidance"] == "keep it short"
    assert [t["title"] for t in r["tasks"]] == [
        "Implement REST API endpoints (v2)", "Add card and wallet payments", "Build courier dashboard"]
    assert r["revision"]["changes"] == ["kept", "changed", "new"]
    assert r["revision"]["removed"] == [{"title": "Write onboarding guide", "description": "Explain the app to new users.",
                                         "reason": "The update says onboarding is out of scope."}]
    assert "1 kept, 1 changed, 1 new, 1 removed" in r["text"]
    assert "### Update: update.pdf" in session.proposal.plan.master_description


def test_tasks_again_after_revising_only_refines():
    agent = revise_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    agent.document_loaded(session, "update.pdf", Document("update"))
    agent.handle(session, "[TASK]")
    agent.handle(session, "[TASK]")  # no new document since: refine, no model calls
    assert len(RevisePlanner.revisions) == 1 and len(FakePlanner.calls) == 1


def test_regenerate_uses_all_documents():
    agent = revise_agent()
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    agent.document_loaded(session, "update.pdf", Document("Couriers need a dashboard."))
    agent.handle(session, "[TASK] regenerate")
    assert len(FakePlanner.calls) == 2 and RevisePlanner.revisions == []


def test_tasks_added_earlier_are_not_generated_again_after_an_update():
    service = FakeService()
    agent = revise_agent(service)
    session = loaded_session(agent)
    agent.handle(session, "[TASK]")
    agent.handle(session, "[ADD]")  # creates master #99 and sub-issues #100.. (item ids PVTI_99, PVTI_100..)
    service.items += [issue(99, "Food app", subs=3)] + [
        issue(100 + n, t, parent=99) for n, t in enumerate(
            ["Implement REST API endpoints", "Add payments", "Write onboarding guide"])]
    agent.document_loaded(session, "update.pdf", Document("Couriers need a dashboard."))

    agent.handle(session, "[TASK]")
    call = FakePlanner.calls[-1]
    # The added tasks count as covered, and the new tasks go under the master we created.
    assert set(call["existing"]) == {"Implement REST API endpoints", "Add payments", "Write onboarding guide"}
    assert session.proposal.master.number == 99


def test_revision_never_forgets_a_task_the_model_skips():
    import json
    from task_agent.planner import TaskPlanner

    class Skipping:
        def chat(self, *a, **k):
            return json.dumps({"tasks": [{"number": 2, "change": "kept", "title": "B", "description": ""}],
                               "removed": []})

    planner = TaskPlanner()
    planner._ollama = Skipping()
    r = planner.revise_tasks([Subtask("A", ""), Subtask("B", ""), Subtask("C", "")], "old", "new")
    assert [t.title for t in r.tasks] == ["A", "B", "C"] and r.changes == ["kept"] * 3



def test_revision_checks_each_task_and_trims_partly_outdated_ones():
    import json
    from task_agent.planner import TaskPlanner

    class Scripted:
        """First call: the revision. Then one check per task; a 'partly' check is followed by its rewrite."""
        def __init__(self):
            self.replies = [
                {"update_changes": ["Kitchen staff don't want a kitchen screen.", "Add a courier dashboard."],
                 "removed": [],
                 "tasks": [{"number": 1, "change": "kept", "title": "Define API endpoints", "description": "x"},
                           {"number": 2, "change": "kept", "title": "Kitchen dashboard", "description": "x"},
                           {"number": 3, "change": "kept", "title": "Mobile app", "description": "x"},
                           {"number": 0, "change": "new", "title": "Build courier dashboard: show couriers on a map in real time",
                            "description": ""}]},
                {"reason": "Orders endpoints are still needed, kitchen ones are not.", "decision": "partly"},
                {"new_details": [], "title": "Define API endpoints", "description": "Endpoints for orders only."},
                {"reason": "Kitchen staff don't want a screen.", "decision": "not_needed"},
                {"reason": "Unaffected.", "decision": "needed"},
            ]
            self.prompts = []

        def chat(self, model, messages, **kwargs):
            self.prompts.append(messages[-1]["content"])
            return json.dumps(self.replies.pop(0))

    planner = TaskPlanner()
    planner._ollama = fake = Scripted()
    tasks = [Subtask("Define API endpoints", "Endpoints for orders and the kitchen dashboard."),
             Subtask("Kitchen dashboard", "Screen for kitchen staff."), Subtask("Mobile app", "Customer screens.")]
    r = planner.revise_tasks(tasks, "scope", "update")
    assert [t.title for t in r.tasks] == ["Define API endpoints", "Mobile app", "Build courier dashboard"]
    assert r.changes == ["changed", "kept", "new"]
    assert r.tasks[0].description == "Endpoints for orders only."
    assert r.tasks[2].description == "show couriers on a map in real time"  # glued title split up
    assert [(t.title, why) for t, why in r.removed] == [("Kitchen dashboard", "Kitchen staff don't want a screen.")]
    assert "Kitchen staff don't want a kitchen screen." in fake.prompts[1]  # checks see the listed changes


def test_old_tasks_token_still_works():
    assert parse_message("[tasks] focus [TASK]") == [("TASK", "focus"), ("TASK", "")]
