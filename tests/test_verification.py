"""
Tests for the verification/ledger/escalation system.

All tests use FakeEngine (no Docker) so they run offline in CI.
The six scenarios cover:
  1. Verifier recovers on retry (UNKNOWN → UNKNOWN → PASS)
  2. Retries exhausted → CONTINUE resolution
  3. Retries exhausted → ROLLBACK resolution
  4. Retries exhausted → STOP resolution (VerificationHaltError)
  5. Ledger entries survive a rollback() call
  6. Backward-compat: no verifier configured → old pass/fail binary behaviour
"""

import sys

import pytest

# Import the package first so __init__.py runs and loads all submodules.
import rewind_sdk

# Reach the actual session *module* via sys.modules — rewind_sdk.session (the
# attribute) is the factory function shadowing the submodule after __init__.py
# runs "from .session import session".
_session_module = sys.modules["rewind_sdk.session"]

from rewind_sdk.verification import (
    EscalationContext,
    EscalationResolution,
    VerificationHaltError,
    VerificationResult,
    VerificationStatus,
    VerifierConfig,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

class FakeEngine:
    """Minimal engine stub; avoids any Docker calls."""

    def __init__(self):
        self.checkpoint_history = []
        self.rolled_back_to = None

    def load_metadata(self):
        return True

    def run_cmd(self, cmd):
        raise RuntimeError(f"Command failed: {cmd}")

    def create_checkpoint(self, label):
        self.checkpoint_history.append(label)

    def rollback_to_checkpoint(self, label):
        self.rolled_back_to = label


def _make_session(escalation_handler=None):
    """Return a session wired to FakeEngine with no escalation delay (retry_delay=0)."""
    engine = FakeEngine()
    session = rewind_sdk.RewindSession(
        engine=engine,
        destroy_on_exit=False,
        escalation_handler=escalation_handler,
    )
    return session, engine


def _verifier_config(command="fake_verifier", retries=2, retry_delay=0.0):
    return VerifierConfig(command=command, retries=retries, retry_delay=retry_delay, timeout=5.0)


def _sequence_verifier(*statuses):
    """
    Return a function that monkeypatches run_verifier to return results from
    *statuses* in order, cycling the last value once exhausted.
    """
    results = list(statuses)
    calls = {"n": 0}

    def fake_run_verifier(config):
        idx = min(calls["n"], len(results) - 1)
        calls["n"] += 1
        status = results[idx]
        return VerificationResult(status=status, raw_output={"status": status.value})

    return fake_run_verifier


def _always_unknown_handler(_ctx: EscalationContext) -> EscalationResolution:
    """Escalation handler that always escalates UNKNOWN to CONTINUE."""
    return EscalationResolution.CONTINUE


def _rollback_handler(_ctx: EscalationContext) -> EscalationResolution:
    return EscalationResolution.ROLLBACK


def _stop_handler(_ctx: EscalationContext) -> EscalationResolution:
    return EscalationResolution.STOP


# ---------------------------------------------------------------------------
# 1. Verifier recovers on retry
# ---------------------------------------------------------------------------

def test_verifier_recovers_on_retry(monkeypatch):
    """
    Verifier returns UNKNOWN twice then PASS on the third attempt.
    No rollback should happen and the ledger should record a PASS entry.
    """
    session, engine = _make_session()

    # Give the session a checkpoint so rollback has somewhere to go if triggered.
    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")

    session.auto_rollback("test_failure", to="good")

    # Attach a verifier that recovers on the third attempt.
    session._auto_rollback.verifier = _verifier_config(retries=3, retry_delay=0.0)

    monkeypatch.setattr(
        _session_module,
        "run_verifier",
        _sequence_verifier(
            VerificationStatus.UNKNOWN,
            VerificationStatus.UNKNOWN,
            VerificationStatus.PASS,
        ),
    )

    session._maybe_auto_rollback("test_failure", patch_notes="run failed")

    # No rollback should have occurred.
    assert engine.rolled_back_to is None
    assert session.last_auto_rollback is None

    # Ledger should have one PASS verification entry.
    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "verification"
    assert entries[0].status == "pass"


# ---------------------------------------------------------------------------
# 2. Retries exhausted → CONTINUE
# ---------------------------------------------------------------------------

def test_exhausted_retries_continue_resolution(monkeypatch):
    """
    Verifier always returns UNKNOWN; escalation handler returns CONTINUE.
    No rollback, ledger has one escalation entry with resolution=continue.
    """
    session, engine = _make_session(escalation_handler=_always_unknown_handler)

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback("test_failure", to="good")
    session._auto_rollback.verifier = _verifier_config(retries=2, retry_delay=0.0)

    monkeypatch.setattr(
        _session_module,
        "run_verifier",
        _sequence_verifier(VerificationStatus.UNKNOWN),
    )

    result = session._maybe_auto_rollback("test_failure")

    assert result is None
    assert engine.rolled_back_to is None

    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "escalation"
    assert entries[0].status == "unknown"
    assert entries[0].resolution == "continue"


# ---------------------------------------------------------------------------
# 3. Retries exhausted → ROLLBACK
# ---------------------------------------------------------------------------

def test_exhausted_retries_rollback_resolution(monkeypatch):
    """
    Verifier always returns UNKNOWN; escalation handler returns ROLLBACK.
    Rollback is executed and ledger records escalation(resolution=rollback)
    followed by a rollback entry.
    """
    session, engine = _make_session(escalation_handler=_rollback_handler)

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback("test_failure", to="good")
    session._auto_rollback.verifier = _verifier_config(retries=1, retry_delay=0.0)

    monkeypatch.setattr(
        _session_module,
        "run_verifier",
        _sequence_verifier(VerificationStatus.UNKNOWN),
    )

    session._maybe_auto_rollback("test_failure")

    assert engine.rolled_back_to == "good"
    assert session.last_auto_rollback is not None
    assert session.last_auto_rollback["event"] == "test_failure"

    entries = session.ledger.history()
    event_types = [e.event_type for e in entries]
    assert "escalation" in event_types
    assert "rollback" in event_types

    escalation_entry = next(e for e in entries if e.event_type == "escalation")
    assert escalation_entry.resolution == "rollback"


# ---------------------------------------------------------------------------
# 4. Retries exhausted → STOP
# ---------------------------------------------------------------------------

def test_exhausted_retries_stop_resolution(monkeypatch):
    """
    Verifier always returns UNKNOWN; escalation handler returns STOP.
    VerificationHaltError is raised; ledger records escalation(resolution=stop).
    """
    session, engine = _make_session(escalation_handler=_stop_handler)

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback("test_failure", to="good")
    session._auto_rollback.verifier = _verifier_config(retries=1, retry_delay=0.0)

    monkeypatch.setattr(
        _session_module,
        "run_verifier",
        _sequence_verifier(VerificationStatus.UNKNOWN),
    )

    with pytest.raises(VerificationHaltError) as exc_info:
        session._maybe_auto_rollback("test_failure")

    halt = exc_info.value
    assert halt.checkpoint == "good"
    assert halt.last_result.status == VerificationStatus.UNKNOWN

    # No rollback — sandbox left alive.
    assert engine.rolled_back_to is None

    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "escalation"
    assert entries[0].resolution == "stop"


# ---------------------------------------------------------------------------
# 5. Ledger survives rollback()
# ---------------------------------------------------------------------------

def test_ledger_survives_rollback(monkeypatch):
    """
    Ledger entries written before a rollback() call must still be present
    afterwards — the ledger is outside the rollback scope.
    """
    session, engine = _make_session(escalation_handler=_always_unknown_handler)

    session.memory.snapshot("stable")
    engine.checkpoint_history.append("stable")
    session.auto_rollback("test_failure", to="stable")
    session._auto_rollback.verifier = _verifier_config(retries=0, retry_delay=0.0)

    monkeypatch.setattr(
        _session_module,
        "run_verifier",
        _sequence_verifier(VerificationStatus.UNKNOWN),
    )

    # Trigger an escalation → CONTINUE; this writes a ledger entry.
    session._maybe_auto_rollback("test_failure")
    assert len(session.ledger.history()) == 1

    # Now do an explicit rollback — ledger must not be touched.
    session.rollback("stable")

    entries = session.ledger.history()
    assert len(entries) == 1, "Ledger was truncated by rollback() — should be immutable"
    assert entries[0].event_type == "escalation"


# ---------------------------------------------------------------------------
# 6. Backward-compat: no verifier configured
# ---------------------------------------------------------------------------

def test_backward_compat_no_verifier():
    """
    When no verifier is configured on AutoRollbackConfig the old behaviour is
    preserved: an exception immediately triggers rollback, last_auto_rollback
    is populated, and no UNKNOWN path is entered.
    """
    engine = FakeEngine()
    session = rewind_sdk.RewindSession(engine=engine, destroy_on_exit=False)

    messages = [
        {"role": "user", "content": "Refactor auth."},
        {"role": "assistant", "content": "Starting."},
    ]

    session.auto_checkpoint(trigger="before_tool_call", keep_last=2)
    session.auto_rollback("test_failure", "exception", to="latest", test_command="pytest")

    # No verifier set.
    assert session._auto_rollback.verifier is None

    session.on_tool_call(messages=messages, tool_name="write_file")

    with pytest.raises(RuntimeError):
        session.run_tests("pytest")

    # Rollback should have happened.
    assert engine.rolled_back_to is not None
    assert session.last_auto_rollback is not None
    assert session.last_auto_rollback["event"] == "test_failure"

    # Ledger should have a rollback entry (even without a verifier, _execute_rollback
    # records the event).
    entries = session.ledger.history()
    assert any(e.event_type == "rollback" for e in entries)
