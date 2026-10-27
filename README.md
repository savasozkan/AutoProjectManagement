# task-agent

Add and manage items in a GitHub Project (v2) from Python or the command line.

## Setup

```bash
/usr/bin/python3 -m venv .venv        # needs Python 3.9+
source .venv/bin/activate
pip install -e .
cp .env.example .env                  # then fill in your token and project
```

Token permissions:
- **Classic token (recommended):** `project` scope (add `repo` if you'll add issues from private repos).
- **Fine-grained token:** only works for **org-owned** projects, with "Projects: Read and write"
  under the org's permissions. It cannot access projects owned by a personal account.

The project number is in its URL: `github.com/users/<owner>/projects/<number>`
(or `github.com/orgs/<org>/projects/<number>` — set `GITHUB_PROJECT_OWNER_TYPE=org`).

## Agent (chat UI)

```bash
task-agent ui            # then open http://127.0.0.1:8000
```

A chat with a project agent that remembers the conversation: the loaded document, the tasks you
selected and the latest generated tasks. Attach a document with 📎 (or drag it in): it waits above
the input box until you press **Send**, so you can add a description or a command first. On Send
the document is read first, then your message runs. Guide the agent with special tokens:

| Token | What the agent does |
|---|---|
| `[SEARCH]` | Finds tasks already in the project that relate to the document. `[SEARCH] payments` searches for that text instead. Results can be selected. |
| `[TASK]` | Generates tasks for the document that don't overlap the selected tasks. Text after it is guidance. Selecting a master task puts the new tasks under it. `[UPDATE] #12` / `[DELETE] #12` work here too. **Asked again**, it doesn't generate anything new: it keeps the current tasks (with your edits) and only removes the ones that overlap the selected project tasks. `[TASK] regenerate` starts over. |
| `[MERGE]` | Merges two or more generated tasks into one. Name them by number (`[MERGE] 2, 3`, text after the numbers guides the merged task: `[MERGE] 2, 3: one task for the API`) or describe them (`[MERGE] the testing and dashboard tasks`). The merged task takes the place of the first one; nothing else changes. |
| `[SPLIT]` | Splits a generated task into smaller, meaningful tasks: `[SPLIT] 3`, `[SPLIT] 3: by platform` / `into 3 tasks`, or `[SPLIT] the mobile app task`. Parts never repeat other tasks in the list or the selected project tasks: overlapping parts are left out, and if fewer than two parts remain the task is kept as it was. |
| `[ADD]` | Adds the generated tasks (with your edits and ticks) to GitHub, after a confirmation. |
| `[HELP]` | Lists the commands. |

You can load **several documents** in one conversation, e.g. the overall scope first and later
updates. The agent keeps all of them and the tasks generated so far. A `[TASK]` after a new
document *revises* the existing tasks instead of starting over: unaffected tasks are kept as they
are (including your edits), affected ones are changed, tasks the update makes unnecessary are
removed (with the reason), and new work becomes new tasks. Tasks the model skips without a reason
are kept, so nothing is forgotten silently. Tasks already added with `[ADD]` in the conversation
count as covered for later rounds.

Text without a token is remembered as guidance for the next `[TASK]`. Several tokens can go in
one message and run in order, e.g. `[SEARCH] [TASK] focus on the API`.

The previous step-by-step page is still available at `/classic`.

## Classic web UI (`/classic`)

```bash
task-agent ui            # then open http://127.0.0.1:8000
```

1. **Load** a PDF, .txt or .md file. Its text and drawings are read.
2. **Tasks already in the project** that are similar to the document are listed, labelled
   *strong* or *possible* match. Tick the ones that already cover part of the work; no new
   subtasks are generated for that work. ("Show all project tasks" lists the rest too.)
   If you tick a **master** task (one with sub-issues), the new subtasks are added under that
   master instead of a new one, and its existing sub-issues count as already covered.
3. **Instructions** (optional): what to emphasize, how detailed to be, what to skip. To change
   existing tasks, add one line per task (by title or `#number`):
   - `[UPDATE] Implement kitchen dashboard` rewrites that task's description (and its title, if it
     no longer fits) from the document and the rest of the instructions.
   - `[DELETE] #34` removes that task.

   Commands only apply to tasks you **ticked in step 2** that **aren't done** (Status "Done",
   or a closed issue / merged PR). Others are skipped with a warning in the review.

   Then **Prepare changes**. Untick "Also generate new subtasks" to only update/delete.
4. **Review**: proposed updates are shown old vs new (editable), deletions with a choice of
   *delete permanently*, *close the issue* or *remove from the project only*, plus any warnings
   (e.g. a task name that matched nothing). Tick or untick subtasks, edit titles and descriptions, add or remove rows. To try
   again, change the ticks in step 2 or the instructions in step 3 and generate again.
5. **Apply to GitHub**: after a confirmation listing every change, applies the ticked updates and
   deletions and creates the selected subtasks as sub-issues, either under the ticked existing
   master or under a new master issue (which links the existing tasks you ticked), adds them to
   the project, and shows links to them.

The UI only listens on your own computer by default (`--host`, `--port` to change).

### How similar tasks are found

The document is split into chunks and each project task (title + description) is scored by its
best-matching chunk, so a task covering one small part of a long document still scores well.
Scoring uses the `qwen3-embedding:0.6b` embedding model (`ollama pull qwen3-embedding:0.6b`),
which matches by meaning. Without it, keyword matching is used instead.

## Command line

```bash
task-agent info                                   # project title + fields and their options
task-agent list                                   # all items, with sub-issue links
task-agent list --status Todo --json              # filter by Status, output as JSON
task-agent add --title "Write docs" --body "..." --field Status=Todo
task-agent add --url https://github.com/me/repo/issues/12 --field Status="In Progress"
task-agent set <ITEM_ID> --field Priority=High --field "Due date=2026-11-01"
task-agent delete <ITEM_ID>
```

## AI task breakdown

Describe a master task in a paragraph and a local model (via [Ollama](https://ollama.com))
splits it into subtasks. You see the list first and choose to add, regenerate, or cancel.

```bash
ollama serve                 # if it isn't already running
ollama pull qwen3:4b         # one-time download
ollama pull qwen3-embedding:0.6b   # optional: better matching with existing tasks

task-agent plan tasks/example.txt                    # description from a text file
task-agent plan tasks/example.txt --field Status=Todo
task-agent plan "Build a personal blog with ..."     # or pass the text directly
task-agent plan spec.pdf -p "Focus on the database part"   # extra instructions
task-agent plan spec.pdf --covered 12,15                  # these issues already cover part of it
task-agent plan spec.pdf --covered 31                     # #31 is a master: add the new subtasks under it
task-agent plan spec.pdf -p "[UPDATE] #33
[DELETE] #34" --no-new                                      # only update / delete (asks before applying)
task-agent plan spec.pdf -p "[DELETE] #34" --delete-mode close   # close instead of deleting
task-agent plan --dry-run "..."          # preview only, adds nothing
task-agent plan --model qwen3:8b "..."   # use a different model
```

It creates the master task as an issue in `GITHUB_REPO` (or `--repo owner/name`), then creates
each subtask as a **sub-issue** linked to it. GitHub shows the sub-issues under the master issue with
a progress bar. All of the issues are added to the project.

Use `--drafts` to add draft items instead (no repository needed, but they can't be linked as
sub-issues). Set `OLLAMA_MODEL` or `OLLAMA_HOST` in `.env` to change the model defaults.

A description file is plain text. If its first line is `# Some title`, that becomes the
master task's title; otherwise the model suggests one (or pass `--title`):

```text
# Launch personal blog

Build a personal blog website with a home page listing posts, Markdown post pages,
an about page, and an RSS feed. Deploy it to GitHub Pages with a custom domain.
```

### Extra instructions

`-p / --prompt` (or the Instructions box in the UI) tells the planner what to emphasize, how
detailed to be, or what to skip, without editing the document. Emphasized parts get several
detailed subtasks; other parts still get at least one unless you say to skip them. For PDFs the
instructions also tell the vision model which details in drawings matter most. They are saved in
the master issue under "Planning focus".

### PDF documents

`task-agent plan` also accepts a PDF. The text of every page is extracted, and pages that contain
drawings, diagrams, images or scanned content are rendered and described by a vision model, so
sketches and architecture diagrams are part of the plan too.

```bash
ollama pull qwen3-vl:4b                           # one-time: vision model for drawings
task-agent plan spec.pdf --dry-run --show-text    # see what was read from the PDF
task-agent plan spec.pdf --no-vision              # text only, skip drawings
```

If the vision model isn't installed, it warns and continues with the text only. The PDF's title
metadata is used as the master-task title when there's no `--title`. Set `OLLAMA_VISION_MODEL`
in `.env` to use a different vision model.

## Python

```python
from task_agent import ProjectService

project = ProjectService.from_env()
item_id = project.add_draft("Write docs", body="Cover setup", fields={"Status": "Todo"})
project.set_field(item_id, "Status", "Done")

for item in project.iter_items():
    print(item.title, item.values)
```

## Tests

```bash
pip install -e ".[dev]"
pytest                  # all tests
pytest -m "not live"    # offline tests only (fake GitHub data)
pytest -m live -s       # read your real project from .env and print its tasks
```

The live tests only read from GitHub; they never create or change anything. They're skipped
when `.env` isn't filled in.

## Layout

- `task_agent/config.py` – reads settings from `.env`
- `task_agent/client.py` – small GraphQL client (Projects v2 has no REST API)
- `task_agent/projects.py` – `ProjectService`: add, list, update, delete items
- `task_agent/planner.py` – `TaskPlanner`: turns a description into subtasks with Ollama
- `task_agent/documents.py` – reads .txt/.md/.pdf files; describes PDF drawings with a vision model
- `task_agent/ollama.py` – small client for the local Ollama server
- `task_agent/agent.py` – the conversational agent and its `[SEARCH]` / `[TASK]` / `[ADD]` commands
- `task_agent/web.py` + `task_agent/static/` – web server, chat UI (`index.html`) and classic UI (`classic.html`)
- `task_agent/similarity.py` – finds existing project tasks similar to a document
- `task_agent/operations.py` – reads `[UPDATE]` / `[DELETE]` commands and finds their tasks
- `task_agent/workflow.py` – proposes all changes for review, then applies them
- `task_agent/cli.py` – the `task-agent` command
