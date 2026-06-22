"""
Verification, ledger, and escalation primitives for RewindSession.

Three-state model
-----------------
  PASS    — verifier ran and confirmed the change is good
  FAIL    — verifier ran and confirmed the change is broken
  UNKNOWN — verifier could not produce a trustworthy signal (crash, timeout,
            unparseable output); never silently coerced to PASS or FAIL

Escalation resolutions (human-only decision point)
---------------------------------------------------
  CONTINUE  — proceed without a trustworthy result; log UNKNOWN to ledger
  ROLLBACK  — revert to the last good checkpoint
  STOP      — halt execution; leave the sandbox alive for manual inspection
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ---------------------------------------------------------------------------
# Status & result
# ---------------------------------------------------------------------------

class VerificationStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


@dataclass
class VerificationResult:
    status: VerificationStatus
    raw_output: dict
    notes: str | None = None


# ---------------------------------------------------------------------------
# Verifier configuration
# ---------------------------------------------------------------------------

@dataclass
class Verifier:
    """Mirrors the AutoCheckpointConfig / AutoRollbackConfig dataclass style."""
    command: str | list[str]
    retries: int = 3
    retry_delay: float = 2.0
    timeout: float | None = 30.0


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

@dataclass
class LedgerEntry:
    timestamp: str
    event_type: str
    status: str | None
    checkpoint: str | None
    raw_output: dict
    notes: str | None
    resolution: str | None


class VerificationLedger:
    """
    Append-only record of verification events and escalation resolutions.

    Lives on RewindSession.ledger and is intentionally outside the rollback
    scope — neither MemoryStore.rollback() nor engine.rollback_to_checkpoint()
    ever touches it.
    """

    def __init__(self) -> None:
        self._entries: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> None:
        self._entries.append(entry)

    def history(self) -> list[LedgerEntry]:
        return list(self._entries)

    def by_checkpoint(self, label: str) -> list[LedgerEntry]:
        return [e for e in self._entries if e.checkpoint == label]

    @staticmethod
    def _now() -> str:
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def record_verification(
        self,
        *,
        status: VerificationStatus,
        checkpoint: str | None,
        raw_output: dict,
        notes: str | None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            timestamp=self._now(),
            event_type="verification",
            status=status.value,
            checkpoint=checkpoint,
            raw_output=raw_output,
            notes=notes,
            resolution=None,
        )
        self.append(entry)
        return entry

    def record_escalation(
        self,
        *,
        status: VerificationStatus,
        checkpoint: str | None,
        raw_output: dict,
        notes: str | None,
        resolution: EscalationResolution,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            timestamp=self._now(),
            event_type="escalation",
            status=status.value,
            checkpoint=checkpoint,
            raw_output=raw_output,
            notes=notes,
            resolution=resolution.value,
        )
        self.append(entry)
        return entry

    def record_rollback(
        self,
        *,
        checkpoint: str | None,
        notes: str | None,
    ) -> LedgerEntry:
        entry = LedgerEntry(
            timestamp=self._now(),
            event_type="rollback",
            status=None,
            checkpoint=checkpoint,
            raw_output={},
            notes=notes,
            resolution="rollback",
        )
        self.append(entry)
        return entry


# ---------------------------------------------------------------------------
# Escalation protocol
# ---------------------------------------------------------------------------

class EscalationResolution(str, Enum):
    CONTINUE = "continue"
    ROLLBACK = "rollback"
    STOP = "stop"


@dataclass
class EscalationContext:
    checkpoint: str | None
    verifier_command: str | list[str]
    last_result: VerificationResult
    attempts: int
    session_name: str


EscalationHandler = Callable[[EscalationContext], EscalationResolution]


# ---------------------------------------------------------------------------
# Halt signal
# ---------------------------------------------------------------------------

class VerificationHaltError(Exception):
    """
    Raised when a verifier returns UNKNOWN after all retries are exhausted and
    the escalation handler resolves STOP.

    The sandbox container is left alive so the developer can inspect state and
    manually re-invoke the verifier.
    """

    def __init__(
        self,
        message: str,
        *,
        checkpoint: str | None,
        verifier_command: str | list[str],
        last_result: VerificationResult,
    ) -> None:
        super().__init__(message)
        self.checkpoint = checkpoint
        self.verifier_command = verifier_command
        self.last_result = last_result


# ---------------------------------------------------------------------------
# Verifier execution
# ---------------------------------------------------------------------------

def parse_verifier_output(stdout: str, stderr: str) -> VerificationResult:
    """
    Parse structured JSON verifier output from captured stdout/stderr.

    Contract: the verifier prints a JSON object to stdout that contains at
    least a ``status`` field ("pass", "fail", or "unknown").  Any other fields
    are passed through as-is in ``raw_output`` without schema validation.

    Returns UNKNOWN when stdout cannot be parsed as a JSON object or the
    ``status`` field is missing or unrecognised.
    """
    try:
        data = json.loads(stdout)
        if not isinstance(data, dict):
            raise ValueError("stdout is not a JSON object")
        status_str = data.get("status", "")
        try:
            status = VerificationStatus(status_str)
        except ValueError:
            raise ValueError(f"Unrecognised status value: {status_str!r}")
        return VerificationResult(status=status, raw_output=data)
    except (json.JSONDecodeError, ValueError) as exc:
        return VerificationResult(
            status=VerificationStatus.UNKNOWN,
            raw_output={"raw_stdout": stdout, "raw_stderr": stderr},
            notes=f"Could not parse verifier output: {exc}",
        )


# ---------------------------------------------------------------------------
# Built-in escalation handlers
# ---------------------------------------------------------------------------

def stdin_escalation_handler(ctx: EscalationContext) -> EscalationResolution:
    """
    Default SDK handler: blocks on stdin so a developer at a keyboard can
    decide how to proceed.  Suitable for direct SDK use and the CLI.
    """
    cmd_display = ctx.verifier_command if isinstance(ctx.verifier_command, str) else " ".join(ctx.verifier_command)
    print(
        f"\n[rewind] Verifier returned UNKNOWN after {ctx.attempts} attempt(s).\n"
        f"  Session   : {ctx.session_name}\n"
        f"  Checkpoint: {ctx.checkpoint or 'none'}\n"
        f"  Command   : {cmd_display}"
    )
    if ctx.last_result.notes:
        print(f"  Details   : {ctx.last_result.notes}")
    print(
        "\nChoose how to proceed:\n"
        "  [c] Continue  — proceed without trustworthy verification (no filesystem change)\n"
        "  [r] Rollback  — revert to last good checkpoint\n"
        "  [s] Stop      — halt and leave the sandbox alive for manual inspection\n"
    )
    while True:
        choice = input("Your choice [c/r/s]: ").strip().lower()
        if choice == "c":
            return EscalationResolution.CONTINUE
        if choice == "r":
            return EscalationResolution.ROLLBACK
        if choice == "s":
            return EscalationResolution.STOP
        print("Please enter c, r, or s.")


def stop_escalation_handler(ctx: EscalationContext) -> EscalationResolution:
    """
    Conservative handler for non-interactive consumers (MCP server, etc.).
    Always resolves STOP so execution is halted rather than silently continuing
    or rolling back without a human decision.
    """
    return EscalationResolution.STOP
