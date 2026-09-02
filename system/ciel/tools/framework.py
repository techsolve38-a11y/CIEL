"""
CIEL Tool Framework
---------------------
Per Development Plan Phase I §7: "Tools must be permission-controlled.
CIEL should know what it can do, what it cannot do, and what requires
user authorization."
"""

from __future__ import annotations
from ciel.tools.documents_handler import create_document

import dataclasses
from enum import Enum
from typing import Callable, Optional


class Permission(str, Enum):
    ALLOWED = "allowed"                  # CIEL may use freely
    REQUIRES_AUTHORIZATION = "requires_authorization"  # must ask user first, each time or per-session
    FORBIDDEN = "forbidden"              # not permitted regardless of request


@dataclasses.dataclass
class Tool:
    name: str
    description: str
    permission: Permission
    handler: Optional[Callable] = None   # callable(**kwargs) -> result; None for not-yet-implemented tools

    def invoke(self, authorized: bool = False, **kwargs):
        if self.permission == Permission.FORBIDDEN:
            raise PermissionError(f"Tool '{self.name}' is forbidden and cannot be invoked.")
        if self.permission == Permission.REQUIRES_AUTHORIZATION and not authorized:
            raise PermissionError(f"Tool '{self.name}' requires explicit user authorization before use.")
        if self.handler is None:
            raise NotImplementedError(f"Tool '{self.name}' has no handler implemented yet.")
        return self.handler(**kwargs)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def list_all(self) -> list[Tool]:
        return list(self._tools.values())

    def list_summaries(self) -> list[dict]:
        return [{"name": t.name, "description": t.description, "permission": t.permission.value}
                 for t in self._tools.values()]


def default_tool_registry() -> ToolRegistry:
    """Seeds the categories named in Development Plan Phase I §7. Handlers
    are wired in incrementally — an unwired tool is visible (so CIEL knows
    it exists) but raises NotImplementedError if invoked."""
    reg = ToolRegistry()
    reg.register(Tool("web_research", "Search and read web content", Permission.REQUIRES_AUTHORIZATION))
    reg.register(Tool("file_system", "Read/write local files", Permission.REQUIRES_AUTHORIZATION))
    reg.register(Tool("database", "Query/update structured data stores", Permission.ALLOWED))
    reg.register(Tool("code_execution", "Execute code in a sandbox", Permission.REQUIRES_AUTHORIZATION))
    reg.register(Tool("external_api", "Call external/third-party APIs", Permission.REQUIRES_AUTHORIZATION))
    reg.register(Tool("documents", "Create/edit documents", Permission.ALLOWED, handler=create_document))
    reg.register(Tool("local_applications", "Control local applications", Permission.REQUIRES_AUTHORIZATION))
    reg.register(Tool("business_systems", "Interact with business/financial systems", Permission.REQUIRES_AUTHORIZATION))
    reg.register(Tool("external_ai_models", "Call other AI models as cognitive resources", Permission.ALLOWED))
    return reg
