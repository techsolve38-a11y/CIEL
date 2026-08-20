"""
CIEL Constitution Loader
-------------------------
Per Foundation Spec §4 (Immutable Core) and Development Plan Phase I §1:
"The constitution must be stored separately from ordinary memories and
user data. It should not be treated as ordinary prompt content."

This module loads the constitution as structured DATA (not a prompt
string assembled ad hoc), verifies its integrity via hash, and exposes
it read-only to the rest of the system. Nothing outside this module may
mutate constitutional content at runtime.
"""

from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field

import yaml

_CONSTITUTION_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "constitution.yaml"


@dataclass(frozen=True)
class CoreLaw:
    id: str
    name: str
    text: str


@dataclass(frozen=True)
class Constitution:
    version: str
    identity: dict
    core_relationship: dict
    core_laws: tuple  # tuple[CoreLaw, ...] — immutable
    immutable_core: dict
    primary_objectives: tuple
    freedom_definition: dict
    sovereignty_model_chain: tuple
    wealth_philosophy_chain: tuple
    sha256: str = field(compare=False)

    def as_system_context(self) -> str:
        """Render the constitution into text for injection into the
        orchestrator's reasoning calls. This is the ONE sanctioned path
        from constitution -> prompt; nothing else should hand-assemble
        constitutional text."""
        lines = [
            f"You are {self.identity['name']} ({self.identity['full_name']}).",
            self.identity["purpose"].strip(),
            "",
            "CORE RELATIONSHIP WITH USER:",
            self.core_relationship["summary"].strip(),
        ]
        for p in self.core_relationship["principles"]:
            lines.append(f"- {p}")
        lines.append("")
        lines.append("CORE LAWS (in priority order where they conflict, Law I is paramount):")
        for law in self.core_laws:
            lines.append(f"Law {law.id} — {law.name}: {law.text.strip()}")
        lines.append("")
        lines.append("IMMUTABLE CORE (cannot be self-modified, ever): " +
                      ", ".join(self.immutable_core["immutable"]))
        lines.append(self.immutable_core["motto"])
        return "\n".join(lines)


def _sha256_of_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_constitution(path: pathlib.Path = _CONSTITUTION_PATH) -> Constitution:
    raw = yaml.safe_load(path.read_text())
    laws = tuple(CoreLaw(**law) for law in raw["core_laws"])
    return Constitution(
        version=raw["version"],
        identity=raw["identity"],
        core_relationship=raw["core_relationship"],
        core_laws=laws,
        immutable_core=raw["immutable_core"],
        primary_objectives=tuple(raw["primary_objectives"]),
        freedom_definition=raw["freedom_definition"],
        sovereignty_model_chain=tuple(raw["sovereignty_model_chain"]),
        wealth_philosophy_chain=tuple(raw["wealth_philosophy_chain"]),
        sha256=_sha256_of_file(path),
    )


def verify_integrity(constitution: Constitution, path: pathlib.Path = _CONSTITUTION_PATH) -> bool:
    """Detect unauthorized modification of the constitution file. The
    orchestrator should call this at startup (and periodically) and
    refuse to operate — or flag loudly to the user — if it fails."""
    return constitution.sha256 == _sha256_of_file(path)
