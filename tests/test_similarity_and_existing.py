"""Offline tests for finding similar tasks and planning around existing ones."""

import json

from task_agent.ollama import OllamaError
from task_agent.planner import ExistingTask, TaskPlanner
from task_agent.projects import ProjectItem
from task_agent.similarity import SimilarityFinder, chunk_text, strength

DOCUMENT = """# Food ordering system

Customers order food from a mobile app. Orders are sent to a REST API server.

The API stores orders in a PostgreSQL database.

Kitchen staff see incoming orders on a live kitchen dashboard."""


def item(number, title, body="", item_type="ISSUE", status="Todo"):
    return ProjectItem(
        id=f"PVTI_{number}", type=item_type, title=title, url=f"https://x/{number}",
        values={"Status": status}, number=number, state="OPEN", body=body,
    )


ITEMS = [
    item(1, "Set up PostgreSQL database", "Create the orders table in PostgreSQL."),
    item(2, "Build kitchen dashboard", "Show incoming orders to kitchen staff."),
    item(3, "Write the company newsletter", "Monthly newsletter about the office party."),
    ProjectItem("PVTI_4", "REDACTED", None, None, {}),
]


class NoEmbeddings:
    def embed(self, model, texts):
        raise OllamaError("Model 'x' not found.")


class FakeEmbeddings:
    """Vectors from a tiny fixed vocabulary, so results are predictable."""

    VOCAB = ["postgresql", "database", "kitchen", "dashboard", "newsletter", "order"]

    def embed(self, model, texts):
        return [[float(w in t.lower()) for w in self.VOCAB] + [0.01] for t in texts]


def test_chunking_splits_long_documents():
    chunks = chunk_text("para one\n\n" + "x" * 1500 + "\n\npara three", size=600)
    assert chunks[0].startswith("para one")
    assert all(len(c) <= 610 for c in chunks)
    assert chunks[-1].endswith("para three")


def test_keyword_matching_ranks_related_tasks_first_and_skips_redacted():
    logs = []
    finder = SimilarityFinder(ollama=NoEmbeddings(), log=logs.append)
    matches = finder.find(DOCUMENT, ITEMS)

    assert finder.method == "keywords"
    assert "keyword matching" in logs[0]
    numbers = [m.item.number for m in matches]
    assert set(numbers[:2]) == {1, 2}
    assert 4 not in [m.item.id for m in matches]  # redacted item ignored
    newsletter = [m for m in matches if m.item.number == 3]
    assert not newsletter or strength(newsletter[0].score, "keywords") == "weak"


def test_embedding_matching_is_used_when_available():
    finder = SimilarityFinder(ollama=FakeEmbeddings())
    matches = finder.find(DOCUMENT, ITEMS)

    assert finder.method == "embeddings"
    assert [m.item.number for m in matches][:2] in ([1, 2], [2, 1])
    assert matches[-1].item.number == 3


def test_no_items_or_empty_document():
    assert SimilarityFinder(ollama=NoEmbeddings()).find(DOCUMENT, []) == []
    assert SimilarityFinder(ollama=NoEmbeddings()).find("", ITEMS) == []


class RecordingOllama:
    def __init__(self):
        self.messages = None

    def chat(self, model, messages, **kwargs):
        self.messages = messages
        return json.dumps({
            "master_title": "Food ordering",
            "tasks": [
                {"title": "Build mobile ordering app", "description": "App screens."},
                {"title": "Set up PostgreSQL database", "description": "Duplicate of existing."},
            ],
        })


def test_planner_is_told_about_existing_tasks_and_links_them():
    planner = TaskPlanner()
    fake = RecordingOllama()
    planner._ollama = fake
    existing = [ExistingTask("Set up PostgreSQL database", "Orders table", "Done", 1, "https://x/1")]

    plan = planner.plan(DOCUMENT, existing=existing, instructions="Keep it short")

    system, user = fake.messages[0]["content"], fake.messages[1]["content"]
    assert "Return ONLY NEW subtasks" in system
    assert "#1 Set up PostgreSQL database [Done]" in user
    assert user.rstrip().endswith("Keep it short")  # instructions stay last
    # A generated task with the same title as an existing one is dropped.
    assert [t.title for t in plan.tasks] == ["Build mobile ordering app"]
    assert "[#1 Set up PostgreSQL database](https://x/1)" in plan.master_description


def test_copied_or_reworded_existing_tasks_are_dropped_even_if_nothing_is_left():
    class CopyingOllama:
        def chat(self, model, messages, **kwargs):
            return json.dumps({"master_title": "x", "tasks": [
                {"title": "Build the mobile app UI", "description": ""},
                {"title": "Implement kitchen dashboards", "description": ""},
            ]})

    planner = TaskPlanner()
    planner._ollama = CopyingOllama()
    existing = [ExistingTask("Build mobile app UI"), ExistingTask("Implement kitchen dashboard")]
    plan = planner.plan(DOCUMENT, existing=existing)
    assert plan.tasks == []  # nothing new, and the copies are not put back


def test_broken_model_reply_is_retried_once_with_memory_freed():
    class FlakyOllama:
        def __init__(self):
            self.replies = ["{", json.dumps({"title": "Set up DB", "new_details": [], "description": "Use Prisma."})]
            self.unloaded = 0

        def chat(self, model, messages, **kwargs):
            return self.replies.pop(0)

        def unload_all(self):
            self.unloaded += 1
            return ["qwen3-embedding:0.6b"]

    planner = TaskPlanner()
    planner._ollama = flaky = FlakyOllama()
    new = planner.rewrite_task("doc", "Set up DB", "old")
    assert new.description == "Use Prisma." and flaky.unloaded == 1
