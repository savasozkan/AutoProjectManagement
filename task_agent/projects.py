from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

from .client import GitHubClient, GitHubError
from .config import Config

if TYPE_CHECKING:
    from .planner import Plan, Subtask

# ---------------------------------------------------------------------------
# GraphQL documents
# ---------------------------------------------------------------------------

_FIELDS_FRAGMENT = """
fields(first: 50) {
  nodes {
    ... on ProjectV2FieldCommon { id name dataType }
    ... on ProjectV2SingleSelectField { options { id name } }
    ... on ProjectV2IterationField {
      configuration { iterations { id title startDate } }
    }
  }
}
"""

_PROJECT_QUERY = """
query($owner: String!, $number: Int!) {
  %(owner_field)s(login: $owner) {
    projectV2(number: $number) {
      id
      title
      url
      %(fields)s
    }
  }
}
"""

_ITEMS_QUERY = """
query($projectId: ID!, $after: String) {
  node(id: $projectId) {
    ... on ProjectV2 {
      items(first: 50, after: $after) {
        pageInfo { hasNextPage endCursor }
        nodes {
          id
          type
          content {
            ... on DraftIssue { id title body }
            ... on Issue {
              id title body url number state
              parent { number title url }
              subIssuesSummary { total completed }
            }
            ... on PullRequest { id title body url number state }
          }
          fieldValues(first: 20) {
            nodes {
              ... on ProjectV2ItemFieldTextValue { text field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldNumberValue { number field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldDateValue { date field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldSingleSelectValue { name field { ... on ProjectV2FieldCommon { name } } }
              ... on ProjectV2ItemFieldIterationValue { title field { ... on ProjectV2FieldCommon { name } } }
            }
          }
        }
      }
    }
  }
}
"""

_RESOLVE_URL_QUERY = """
query($url: URI!) {
  resource(url: $url) {
    ... on Issue { id }
    ... on PullRequest { id }
  }
}
"""

_ADD_DRAFT_MUTATION = """
mutation($projectId: ID!, $title: String!, $body: String) {
  addProjectV2DraftIssue(input: {projectId: $projectId, title: $title, body: $body}) {
    projectItem { id }
  }
}
"""

_ADD_EXISTING_MUTATION = """
mutation($projectId: ID!, $contentId: ID!) {
  addProjectV2ItemById(input: {projectId: $projectId, contentId: $contentId}) {
    item { id }
  }
}
"""

_UPDATE_FIELD_MUTATION = """
mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $value: ProjectV2FieldValue!) {
  updateProjectV2ItemFieldValue(
    input: {projectId: $projectId, itemId: $itemId, fieldId: $fieldId, value: $value}
  ) {
    projectV2Item { id }
  }
}
"""

_REPO_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) { id }
}
"""

_CREATE_ISSUE_MUTATION = """
mutation($repositoryId: ID!, $title: String!, $body: String, $parentIssueId: ID) {
  createIssue(input: {
    repositoryId: $repositoryId, title: $title, body: $body, parentIssueId: $parentIssueId
  }) {
    issue { id number url }
  }
}
"""

_UPDATE_ISSUE_MUTATION = """
mutation($id: ID!, $title: String, $body: String) {
  updateIssue(input: {id: $id, title: $title, body: $body}) { issue { id } }
}
"""

_UPDATE_DRAFT_MUTATION = """
mutation($id: ID!, $title: String, $body: String) {
  updateProjectV2DraftIssue(input: {draftIssueId: $id, title: $title, body: $body}) {
    draftIssue { id }
  }
}
"""

_UPDATE_PR_MUTATION = """
mutation($id: ID!, $title: String, $body: String) {
  updatePullRequest(input: {pullRequestId: $id, title: $title, body: $body}) {
    pullRequest { id }
  }
}
"""

_DELETE_ISSUE_MUTATION = """
mutation($id: ID!) {
  deleteIssue(input: {issueId: $id}) { clientMutationId }
}
"""

_CLOSE_ISSUE_MUTATION = """
mutation($id: ID!) {
  closeIssue(input: {issueId: $id, stateReason: NOT_PLANNED}) { issue { id } }
}
"""

_DELETE_ITEM_MUTATION = """
mutation($projectId: ID!, $itemId: ID!) {
  deleteProjectV2Item(input: {projectId: $projectId, itemId: $itemId}) {
    deletedItemId
  }
}
"""

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

FieldValue = Union[str, int, float]


@dataclass
class ProjectField:
    id: str
    name: str
    data_type: str  # TEXT, NUMBER, DATE, SINGLE_SELECT, ITERATION, TITLE, ...
    options: Dict[str, str] = field(default_factory=dict)  # option/iteration name -> id


@dataclass
class ProjectItem:
    id: str
    type: str  # DRAFT_ISSUE, ISSUE, PULL_REQUEST, REDACTED
    title: Optional[str]
    url: Optional[str]
    values: Dict[str, Any]  # project field name -> value, e.g. {"Status": "Todo"}
    number: Optional[int] = None  # issue / PR number (None for drafts)
    state: Optional[str] = None  # OPEN, CLOSED or MERGED (None for drafts)
    parent: Optional[Dict[str, Any]] = None  # {"number", "title", "url"} of the master issue
    body: Optional[str] = None  # issue / PR / draft description
    content_id: Optional[str] = None  # node ID of the issue / PR / draft behind this item
    sub_issue_count: int = 0  # > 0 means this issue is a master task

    @property
    def is_master(self) -> bool:
        return self.sub_issue_count > 0

    @property
    def is_done(self) -> bool:
        """Finished work: Status is "Done", or the issue is closed / the PR merged."""
        return (self.status or "").strip().lower() == "done" or self.state in ("CLOSED", "MERGED")

    @property
    def status(self) -> Optional[str]:
        return self.values.get("Status")


@dataclass
class CreatedIssue:
    id: str  # issue node ID
    number: int
    url: str
    item_id: str  # project item ID


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class ProjectService:
    """High-level operations on one GitHub Project (v2)."""

    def __init__(self, client: GitHubClient, config: Config):
        self._client = client
        self._config = config
        self._project: Optional[Dict[str, Any]] = None
        self._fields: Dict[str, ProjectField] = {}
        self._repo_ids: Dict[str, str] = {}

    @classmethod
    def from_env(cls) -> "ProjectService":
        config = Config.from_env()
        return cls(GitHubClient(config.token), config)

    # -- Project metadata ---------------------------------------------------

    def _load(self) -> Dict[str, Any]:
        if self._project is None:
            owner_field = "organization" if self._config.owner_type == "org" else "user"
            query = _PROJECT_QUERY % {"owner_field": owner_field, "fields": _FIELDS_FRAGMENT}
            data = self._client.graphql(
                query,
                {"owner": self._config.owner, "number": self._config.project_number},
            )
            project = (data.get(owner_field) or {}).get("projectV2")
            if not project:
                raise GitHubError(
                    f"Project #{self._config.project_number} not found for "
                    f"{self._config.owner_type} '{self._config.owner}'."
                )
            self._project = project
            self._fields = {f.name.lower(): f for f in self._parse_fields(project)}
        return self._project

    @staticmethod
    def _parse_fields(project: Dict[str, Any]) -> List[ProjectField]:
        result = []
        for node in project["fields"]["nodes"]:
            if not node:
                continue
            options = {o["name"]: o["id"] for o in node.get("options") or []}
            iterations = (node.get("configuration") or {}).get("iterations") or []
            options.update({it["title"]: it["id"] for it in iterations})
            result.append(ProjectField(node["id"], node["name"], node["dataType"], options))
        return result

    @property
    def config(self) -> Config:
        return self._config

    @property
    def project_id(self) -> str:
        return self._load()["id"]

    @property
    def title(self) -> str:
        return self._load()["title"]

    @property
    def url(self) -> str:
        return self._load()["url"]

    def fields(self) -> List[ProjectField]:
        self._load()
        return list(self._fields.values())

    def get_field(self, name: str) -> ProjectField:
        self._load()
        try:
            return self._fields[name.lower()]
        except KeyError:
            known = ", ".join(f.name for f in self._fields.values())
            raise GitHubError(f"Unknown field '{name}'. Available fields: {known}") from None

    # -- Reading items ------------------------------------------------------

    def get_items(self, status: Optional[str] = None) -> List[ProjectItem]:
        """All items in the project, optionally only those with a given Status (case-insensitive)."""
        items = list(self.iter_items())
        if status is not None:
            items = [i for i in items if (i.status or "").lower() == status.lower()]
        return items

    def iter_items(self) -> Iterator[ProjectItem]:
        cursor = None
        while True:
            data = self._client.graphql(
                _ITEMS_QUERY, {"projectId": self.project_id, "after": cursor}
            )
            items = data["node"]["items"]
            for node in items["nodes"]:
                yield self._parse_item(node)
            if not items["pageInfo"]["hasNextPage"]:
                return
            cursor = items["pageInfo"]["endCursor"]

    @staticmethod
    def _parse_item(node: Dict[str, Any]) -> ProjectItem:
        content = node.get("content") or {}
        values: Dict[str, Any] = {}
        for fv in node["fieldValues"]["nodes"]:
            if not fv or "field" not in fv:
                continue
            name = fv["field"].get("name")
            for key in ("text", "number", "date", "name", "title"):
                if key in fv:
                    values[name] = fv[key]
                    break
        return ProjectItem(
            id=node["id"],
            type=node["type"],
            title=content.get("title"),
            url=content.get("url"),
            values=values,
            number=content.get("number"),
            state=content.get("state"),
            parent=content.get("parent"),
            body=content.get("body"),
            content_id=content.get("id"),
            sub_issue_count=(content.get("subIssuesSummary") or {}).get("total", 0),
        )

    # -- Adding / changing items -------------------------------------------

    def add_draft(
        self,
        title: str,
        body: Optional[str] = None,
        fields: Optional[Dict[str, FieldValue]] = None,
    ) -> str:
        """Add a draft issue to the project. Returns the new item ID."""
        data = self._client.graphql(
            _ADD_DRAFT_MUTATION,
            {"projectId": self.project_id, "title": title, "body": body},
        )
        item_id = data["addProjectV2DraftIssue"]["projectItem"]["id"]
        self.set_fields(item_id, fields or {})
        return item_id

    def add_existing(self, url: str, fields: Optional[Dict[str, FieldValue]] = None) -> str:
        """Add an existing issue or pull request (by its GitHub URL). Returns the item ID."""
        data = self._client.graphql(_RESOLVE_URL_QUERY, {"url": url})
        resource = data.get("resource") or {}
        if "id" not in resource:
            raise GitHubError(f"No issue or pull request found at {url}")
        return self._add_content(resource["id"], fields)

    def _add_content(self, content_id: str, fields: Optional[Dict[str, FieldValue]] = None) -> str:
        data = self._client.graphql(
            _ADD_EXISTING_MUTATION,
            {"projectId": self.project_id, "contentId": content_id},
        )
        item_id = data["addProjectV2ItemById"]["item"]["id"]
        self.set_fields(item_id, fields or {})
        return item_id

    def _repo_id(self, repo: str) -> str:
        if repo not in self._repo_ids:
            owner, sep, name = repo.partition("/")
            if not sep or not owner or not name:
                raise GitHubError(f"Repository must look like 'owner/name', got '{repo}'.")
            data = self._client.graphql(_REPO_QUERY, {"owner": owner, "name": name})
            if not data.get("repository"):
                raise GitHubError(f"Repository '{repo}' not found or not accessible.")
            self._repo_ids[repo] = data["repository"]["id"]
        return self._repo_ids[repo]

    def create_issue(
        self,
        repo: str,
        title: str,
        body: Optional[str] = None,
        parent_issue_id: Optional[str] = None,
        fields: Optional[Dict[str, FieldValue]] = None,
    ) -> CreatedIssue:
        """Create an issue in `repo`, optionally as a sub-issue, and add it to the project."""
        data = self._client.graphql(
            _CREATE_ISSUE_MUTATION,
            {
                "repositoryId": self._repo_id(repo),
                "title": title,
                "body": body,
                "parentIssueId": parent_issue_id,
            },
        )
        issue = data["createIssue"]["issue"]
        item_id = self._add_content(issue["id"], fields)
        return CreatedIssue(issue["id"], issue["number"], issue["url"], item_id)

    def add_plan(
        self,
        plan: "Plan",
        repo: str,
        fields: Optional[Dict[str, FieldValue]] = None,
    ) -> Tuple[CreatedIssue, List[CreatedIssue]]:
        """Create the master task as an issue and each subtask as a linked sub-issue.

        All issues are added to the project; `fields` are applied to every item.
        """
        master = self.create_issue(repo, plan.master_title, plan.master_description, fields=fields)
        subtasks = [
            self.create_issue(repo, task.title, task.description, master.id, fields)
            for task in plan.tasks
        ]
        return master, subtasks

    def add_to_master(
        self,
        master_issue_id: str,
        tasks: List["Subtask"],
        repo: str,
        fields: Optional[Dict[str, FieldValue]] = None,
    ) -> List[CreatedIssue]:
        """Create each task as a sub-issue of an existing master issue, and add it to the project."""
        return [
            self.create_issue(repo, task.title, task.description, master_issue_id, fields)
            for task in tasks
        ]

    def add_plan_as_drafts(
        self, plan: "Plan", fields: Optional[Dict[str, FieldValue]] = None
    ) -> List[str]:
        """Add the plan as draft items (no repository needed, but no real parent link).

        Returns the item IDs, master first.
        """
        checklist = "\n".join(f"- [ ] {t.title}" for t in plan.tasks)
        master_body = f"{plan.master_description}\n\n### Subtasks\n{checklist}"
        ids = [self.add_draft(plan.master_title, master_body, fields)]
        for task in plan.tasks:
            body = f"{task.description}\n\n_Part of: **{plan.master_title}**_"
            ids.append(self.add_draft(task.title, body, fields))
        return ids

    def set_fields(self, item_id: str, fields: Dict[str, FieldValue]) -> None:
        for name, value in fields.items():
            self.set_field(item_id, name, value)

    def set_field(self, item_id: str, field_name: str, value: FieldValue) -> None:
        """Set one field on an item, e.g. set_field(item, "Status", "In Progress")."""
        project_field = self.get_field(field_name)
        self._client.graphql(
            _UPDATE_FIELD_MUTATION,
            {
                "projectId": self.project_id,
                "itemId": item_id,
                "fieldId": project_field.id,
                "value": self._build_value(project_field, value),
            },
        )

    @staticmethod
    def _build_value(project_field: ProjectField, value: FieldValue) -> Dict[str, Any]:
        kind = project_field.data_type
        if kind == "TEXT":
            return {"text": str(value)}
        if kind == "NUMBER":
            return {"number": float(value)}
        if kind == "DATE":
            return {"date": str(value)}  # YYYY-MM-DD
        if kind in ("SINGLE_SELECT", "ITERATION"):
            lookup = {k.lower(): v for k, v in project_field.options.items()}
            option_id = lookup.get(str(value).lower())
            if option_id is None:
                choices = ", ".join(project_field.options)
                raise GitHubError(
                    f"'{value}' is not an option for '{project_field.name}'. Choices: {choices}"
                )
            key = "singleSelectOptionId" if kind == "SINGLE_SELECT" else "iterationId"
            return {key: option_id}
        raise GitHubError(
            f"Field '{project_field.name}' has type {kind}, which can't be set through this API."
        )

    def update_task(self, item: ProjectItem, title: Optional[str], body: Optional[str]) -> None:
        """Change the title and/or description of an issue, pull request or draft item."""
        mutation = {
            "ISSUE": _UPDATE_ISSUE_MUTATION,
            "DRAFT_ISSUE": _UPDATE_DRAFT_MUTATION,
            "PULL_REQUEST": _UPDATE_PR_MUTATION,
        }.get(item.type)
        if not mutation or not item.content_id:
            raise GitHubError(f"'{item.title}' can't be updated (type {item.type}).")
        self._client.graphql(mutation, {"id": item.content_id, "title": title, "body": body})

    def remove_task(self, item: ProjectItem, mode: str = "delete") -> None:
        """Remove a task.

        mode "delete": delete the issue from GitHub for good (drafts are removed from the project).
        mode "close":  close the issue as not planned; it stays in the project.
        mode "remove": take it off the project board only; the issue itself is kept.
        """
        if mode == "remove" or item.type == "DRAFT_ISSUE":
            self.delete_item(item.id)
        elif item.type != "ISSUE" or not item.content_id:
            raise GitHubError(
                f"'{item.title}' is a {item.type.lower().replace('_', ' ')}; only issues can be "
                "deleted or closed. Use 'remove from project' instead."
            )
        elif mode == "delete":
            self._client.graphql(_DELETE_ISSUE_MUTATION, {"id": item.content_id})
        elif mode == "close":
            self._client.graphql(_CLOSE_ISSUE_MUTATION, {"id": item.content_id})
        else:
            raise GitHubError(f"Unknown removal mode '{mode}'.")

    def delete_item(self, item_id: str) -> None:
        self._client.graphql(
            _DELETE_ITEM_MUTATION, {"projectId": self.project_id, "itemId": item_id}
        )


def split_selection(
    selected: List[ProjectItem], all_items: List[ProjectItem]
) -> Tuple[Optional[ProjectItem], List[ProjectItem]]:
    """Work out what the user's ticked tasks mean for planning.

    Returns (master, covered):
    - master: the first ticked master issue (one with sub-issues). New subtasks go under it
      instead of under a new master. None if no master was ticked.
    - covered: tasks whose work should not be generated again. A ticked master is replaced by
      its sub-issues, since its own description is the whole job.
    """
    master = next((i for i in selected if i.is_master and i.content_id), None)
    covered: List[ProjectItem] = []
    seen = set()

    def add(item: ProjectItem) -> None:
        if item.id not in seen:
            seen.add(item.id)
            covered.append(item)

    for item in selected:
        if item.is_master:
            for child in all_items:
                if child.parent and child.parent.get("number") == item.number:
                    add(child)
        else:
            add(item)
    return master, covered
