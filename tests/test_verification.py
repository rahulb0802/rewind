"""
Tests for the verification/ledger/escalation system.

All tests use FakeEngine (no Docker) so they run offline in CI.
The twelve scenarios cover:
  1. Verifier recovers on retry (UNKNOWN → UNKNOWN → PASS)
  2. Retries exhausted → CONTINUE resolution
  3. Retries exhausted → ROLLBACK resolution
  4. Retries exhausted → STOP resolution (VerificationHaltError)
  5. Ledger entries survive a rollback() call
  6–8. parse_verifier_output: pass, fail, unknown (bad JSON)
  9–11. run_tests in-container JSON path: pass, fail → rollback, unknown → escalation
"""

import json

import pytest

import rewind_sdk

from rewind_sdk.verification import (
    EscalationContext,
    EscalationResolution,
    VerificationHaltError,
    VerificationStatus,
    Verifier,
    parse_verifier_output,
)


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

class FakeEngine:
    """Minimal engine stub; avoids any Docker calls."""

    def __init__(self):
        self.checkpoint_history = []
        self.rolled_back_to = None
        self._stdout = ""
        self._stderr = ""
        self._returncode = 1
        self._stdout_sequence: list[str] = []

    def load_metadata(self):
        return True

    def run_cmd(self, cmd):
        if self._returncode != 0:
            raise RuntimeError(f"Command failed: {cmd}")
        return self._stdout.strip()

    def run_cmd_capturing(self, cmd, timeout=None):
        if self._stdout_sequence:
            stdout = self._stdout_sequence.pop(0)
            return stdout, self._stderr, self._returncode
        return self._stdout, self._stderr, self._returncode

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
    return Verifier(command=command, retries=retries, retry_delay=retry_delay, timeout=5.0)


def _stdout_for_status(status):
    if status == VerificationStatus.UNKNOWN:
        return "not valid json"
    return json.dumps({"status": status.value})


def _stdout_sequence_for_statuses(*statuses):
    return [_stdout_for_status(status) for status in statuses]


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

def test_verifier_recovers_on_retry():
    """
    Verifier returns UNKNOWN twice then PASS on the third attempt.
    No rollback should happen and the ledger should record a PASS entry.
    """
    session, engine = _make_session()
    session._started = True

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")

    session.auto_rollback("test_failure", to="good", verifier=_verifier_config(retries=3, retry_delay=0.0))

    engine._stdout_sequence = _stdout_sequence_for_statuses(
        VerificationStatus.UNKNOWN,
        VerificationStatus.UNKNOWN,
        VerificationStatus.PASS,
    )

    output = session.run_tests()

    assert output == "Verification passed."
    assert engine.rolled_back_to is None
    assert session.last_auto_rollback is None

    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "verification"
    assert entries[0].status == "pass"


# ---------------------------------------------------------------------------
# 2. Retries exhausted → CONTINUE
# ---------------------------------------------------------------------------

def test_exhausted_retries_continue_resolution():
    """
    Verifier always returns UNKNOWN; escalation handler returns CONTINUE.
    No rollback, ledger has one escalation entry with resolution=continue.
    """
    session, engine = _make_session(escalation_handler=_always_unknown_handler)
    session._started = True

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback("test_failure", to="good", verifier=_verifier_config(retries=2, retry_delay=0.0))

    engine._stdout_sequence = _stdout_sequence_for_statuses(VerificationStatus.UNKNOWN) * 4

    with pytest.raises(RuntimeError):
        session.run_tests()

    assert engine.rolled_back_to is None

    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "escalation"
    assert entries[0].status == "unknown"
    assert entries[0].resolution == "continue"


# ---------------------------------------------------------------------------
# 3. Retries exhausted → ROLLBACK
# ---------------------------------------------------------------------------

def test_exhausted_retries_rollback_resolution():
    """
    Verifier always returns UNKNOWN; escalation handler returns ROLLBACK.
    Rollback is executed and ledger records escalation(resolution=rollback)
    followed by a rollback entry.
    """
    session, engine = _make_session(escalation_handler=_rollback_handler)
    session._started = True

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback("test_failure", to="good", verifier=_verifier_config(retries=1, retry_delay=0.0))

    engine._stdout_sequence = _stdout_sequence_for_statuses(VerificationStatus.UNKNOWN) * 3

    with pytest.raises(RuntimeError):
        session.run_tests()

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

def test_exhausted_retries_stop_resolution():
    """
    Verifier always returns UNKNOWN; escalation handler returns STOP.
    VerificationHaltError is raised; ledger records escalation(resolution=stop).
    """
    session, engine = _make_session(escalation_handler=_stop_handler)
    session._started = True

    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback("test_failure", to="good", verifier=_verifier_config(retries=1, retry_delay=0.0))

    engine._stdout_sequence = _stdout_sequence_for_statuses(VerificationStatus.UNKNOWN) * 3

    with pytest.raises(VerificationHaltError) as exc_info:
        session.run_tests()

    halt = exc_info.value
    assert halt.checkpoint == "good"
    assert halt.last_result.status == VerificationStatus.UNKNOWN

    assert engine.rolled_back_to is None

    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "escalation"
    assert entries[0].resolution == "stop"


# ---------------------------------------------------------------------------
# 5. Ledger survives rollback()
# ---------------------------------------------------------------------------

def test_ledger_survives_rollback():
    """
    Ledger entries written before a rollback() call must still be present
    afterwards — the ledger is outside the rollback scope.
    """
    session, engine = _make_session(escalation_handler=_always_unknown_handler)
    session._started = True

    session.memory.snapshot("stable")
    engine.checkpoint_history.append("stable")
    session.auto_rollback("test_failure", to="stable", verifier=_verifier_config(retries=0, retry_delay=0.0))

    engine._stdout_sequence = _stdout_sequence_for_statuses(VerificationStatus.UNKNOWN)

    with pytest.raises(RuntimeError):
        session.run_tests()
    assert len(session.ledger.history()) == 1

    session.rollback("stable")

    entries = session.ledger.history()
    assert len(entries) == 1, "Ledger was truncated by rollback() — should be immutable"
    assert entries[0].event_type == "escalation"


def test_auto_rollback_verifier_kwarg():
    """verifier= on auto_rollback() is stored on the public config object."""
    session, _engine = _make_session()
    config = _verifier_config(command="python3 verify.py", retries=1)
    session.auto_rollback("test_failure", to="good", verifier=config)
    assert session._auto_rollback.verifier is config
    assert session._auto_rollback.verifier.command == "python3 verify.py"


# ---------------------------------------------------------------------------
# parse_verifier_output unit tests
# ---------------------------------------------------------------------------

def test_parse_verifier_output_pass():
    result = parse_verifier_output('{"status": "pass"}', "")
    assert result.status == VerificationStatus.PASS
    assert result.raw_output == {"status": "pass"}


def test_parse_verifier_output_fail():
    result = parse_verifier_output('{"status": "fail", "errors": ["boom"]}', "")
    assert result.status == VerificationStatus.FAIL
    assert result.raw_output["errors"] == ["boom"]


def test_parse_verifier_output_unknown_bad_json():
    result = parse_verifier_output("not json at all", "stderr noise")
    assert result.status == VerificationStatus.UNKNOWN
    assert result.raw_output["raw_stdout"] == "not json at all"
    assert result.raw_output["raw_stderr"] == "stderr noise"
    assert result.notes is not None


# ---------------------------------------------------------------------------
# In-container JSON verifier path (run_tests)
# ---------------------------------------------------------------------------

def _make_verified_session(engine, escalation_handler=None):
    """Session with auto-rollback + verifier configured and a known-good checkpoint."""
    session = rewind_sdk.RewindSession(
        engine=engine,
        destroy_on_exit=False,
        escalation_handler=escalation_handler,
    )
    session._started = True
    session.memory.snapshot("good")
    engine.checkpoint_history.append("good")
    session.auto_rollback(
        "test_failure",
        to="good",
        verifier=_verifier_config(command="pytest", retries=0, retry_delay=0.0),
    )
    return session


def test_run_tests_json_only_pass():
    """
    Verifier configured: in-container stdout is valid JSON pass.
    No rollback.
    """
    engine = FakeEngine()
    engine._stdout = '{"status": "pass"}'
    engine._returncode = 0
    session = _make_verified_session(engine)

    output = session.run_tests()

    assert output == "Verification passed."
    assert engine.rolled_back_to is None
    assert session.last_auto_rollback is None


def test_run_tests_json_only_fail():
    """
    Verifier configured: in-container stdout is valid JSON fail.
    Rollback is triggered without re-running the verifier on the host.
    """
    engine = FakeEngine()
    engine._stdout = '{"status": "fail", "errors": ["assertion failed"]}'
    engine._returncode = 1
    session = _make_verified_session(engine)

    with pytest.raises(RuntimeError):
        session.run_tests()

    assert engine.rolled_back_to == "good"
    assert session.last_auto_rollback is not None
    assert session.last_auto_rollback["event"] == "test_failure"

    entries = session.ledger.history()
    assert len(entries) == 2
    verification = next(e for e in entries if e.event_type == "verification")
    assert verification.status == "fail"
    assert "rollback" in [e.event_type for e in entries]


def test_run_tests_json_only_unknown_escalates():
    """
    Verifier configured: in-container stdout is not valid JSON.
    UNKNOWN is escalated via the handler; no rollback when handler returns CONTINUE.
    """
    engine = FakeEngine()
    engine._stdout = "pytest: 3 failed"
    engine._stderr = "traceback..."
    engine._returncode = 1
    session = _make_verified_session(engine, escalation_handler=_always_unknown_handler)

    with pytest.raises(RuntimeError):
        session.run_tests()

    assert engine.rolled_back_to is None

    entries = session.ledger.history()
    assert len(entries) == 1
    assert entries[0].event_type == "escalation"
    assert entries[0].status == "unknown"
    assert entries[0].resolution == "continue"
