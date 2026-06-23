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
    VerificationResult,
    VerificationStatus,
    Verifier,
    format_verification_result,
    parse_verifier_output,
    stdin_escalation_handler,
    stop_escalation_handler,
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
        self.destroyed = False
        self.committed = False

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

    def destroy_sandbox(self):
        self.destroyed = True

    def commit(self, workspace):
        self.committed = True


def _make_session(escalation_handler=None, mode="interactive", **kwargs):
    """Return a session wired to FakeEngine with no escalation delay (retry_delay=0)."""
    engine = FakeEngine()
    session = rewind_sdk.RewindSession(
        engine=engine,
        destroy_on_exit=False,
        escalation_handler=escalation_handler,
        mode=mode,
        **kwargs,
    )
    return session, engine


def _make_halt_error():
    return VerificationHaltError(
        "halted",
        checkpoint="good",
        verifier_command="fake",
        last_result=VerificationResult(
            status=VerificationStatus.UNKNOWN,
            raw_output={},
            notes="unknown",
        ),
    )


def _invoke_tool(tool):
    if hasattr(tool, "invoke"):
        return tool.invoke({})
    return tool()


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


# ---------------------------------------------------------------------------
# mode defaults
# ---------------------------------------------------------------------------

def test_mode_agent_sets_stop_handler():
    session, _engine = _make_session(mode="agent")
    assert session._escalation_handler is stop_escalation_handler


def test_mode_interactive_uses_stdin_handler():
    session, _engine = _make_session()
    assert session._escalation_handler is stdin_escalation_handler


def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be one of"):
        rewind_sdk.RewindSession(mode="bad", destroy_on_exit=False)


# ---------------------------------------------------------------------------
# halt-aware __exit__
# ---------------------------------------------------------------------------

def test_exit_on_halt_preserves_sandbox_agent_mode():
    engine = FakeEngine()
    session = rewind_sdk.RewindSession(
        engine=engine,
        destroy_on_exit=True,
        mode="agent",
    )
    session._started = True
    halt = _make_halt_error()

    session.__exit__(VerificationHaltError, halt, None)

    assert engine.destroyed is False


def test_exit_on_halt_destroys_in_interactive_mode():
    engine = FakeEngine()
    session = rewind_sdk.RewindSession(
        engine=engine,
        destroy_on_exit=True,
        mode="interactive",
    )
    session._started = True
    halt = _make_halt_error()

    session.__exit__(VerificationHaltError, halt, None)

    assert engine.destroyed is True


def test_exit_on_halt_skips_commit():
    engine = FakeEngine()
    session = rewind_sdk.RewindSession(
        engine=engine,
        destroy_on_exit=False,
        auto_commit=True,
        mode="agent",
    )
    session._started = True
    halt = _make_halt_error()

    session.__exit__(VerificationHaltError, halt, None)

    assert engine.committed is False


# ---------------------------------------------------------------------------
# @sandbox.tool decorator
# ---------------------------------------------------------------------------

def test_tool_decorator_injects_on_tool_call():
    session, engine = _make_session()
    session._started = True
    session.auto_checkpoint(trigger="before_tool_call")

    @session.tool
    def my_tool():
        return "ok"

    _invoke_tool(my_tool)

    assert len(engine.checkpoint_history) == 1


def test_tool_decorator_converts_runtime_error():
    session, _engine = _make_session()
    session._started = True

    @session.tool
    def failing_tool():
        raise RuntimeError("something broke")

    result = _invoke_tool(failing_tool)

    assert result == "ERROR: something broke"


def test_tool_decorator_propagates_halt_error():
    session, _engine = _make_session()
    session._started = True
    halt = _make_halt_error()

    @session.tool
    def halt_tool():
        raise halt

    with pytest.raises(VerificationHaltError):
        _invoke_tool(halt_tool)


def _session_with_exception_rollback(label="good"):
    """Session with auto_rollback on exception targeting a known checkpoint."""
    session, engine = _make_session()
    session._started = True
    session.memory.snapshot(label)
    engine.checkpoint_history.append(label)
    session.auto_rollback("exception", to=label)
    return session, engine


def test_tool_rollback_on_error_false_no_rollback():
    """
    RuntimeError inside a rollback_on_error=False tool must not trigger
    auto-rollback even when auto_rollback("exception") is configured.
    """
    session, engine = _session_with_exception_rollback()

    @session.tool(rollback_on_error=False)
    def failing_sql():
        """Run a read-only SQL query (rollback suppressed on failure)."""
        session.run("SELECT bad")

    _invoke_tool(failing_sql)

    assert engine.rolled_back_to is None
    assert session.last_auto_rollback is None


def test_tool_rollback_on_error_true_triggers_rollback():
    """
    RuntimeError inside a rollback_on_error=True tool triggers auto-rollback
    when auto_rollback("exception") is configured.
    """
    session, engine = _session_with_exception_rollback()

    @session.tool(rollback_on_error=True)
    def failing_script():
        """Run a state-changing script (rollback enabled on failure)."""
        session.run("bad-script")

    _invoke_tool(failing_script)

    assert engine.rolled_back_to == "good"
    assert session.last_auto_rollback is not None
    assert session.last_auto_rollback["event"] == "exception"


def test_rollback_notice_in_error_string():
    """When rollback fires inside a tool, the returned error string includes [REWIND]."""
    session, engine = _session_with_exception_rollback(label="pre_migration")

    @session.tool
    def failing_script():
        """Run a script that fails and triggers rollback."""
        session.run("bad-script")

    result = _invoke_tool(failing_script)

    assert result.startswith("ERROR:")
    assert "[REWIND]" in result
    assert "pre_migration" in result
    assert engine.rolled_back_to == "pre_migration"


def test_noop_guard_skips_second_rollback():
    """Second rollback to the same checkpoint must not call engine.rollback again."""
    session, engine = _session_with_exception_rollback()

    rollback_calls = []

    def tracking_rollback(label):
        rollback_calls.append(label)
        engine.rolled_back_to = label

    engine.rollback_to_checkpoint = tracking_rollback

    session._maybe_auto_rollback("exception", patch_notes="first")
    assert len(rollback_calls) == 1

    result = session._maybe_auto_rollback("exception", patch_notes="second")
    assert result is None
    assert len(rollback_calls) == 1


def test_noop_guard_ledger_entry():
    """Skipped duplicate rollback is recorded in the ledger as skipped_noop."""
    session, _engine = _session_with_exception_rollback()

    session._maybe_auto_rollback("exception", patch_notes="first")
    session._maybe_auto_rollback("exception", patch_notes="second")

    entries = session.ledger.history()
    noop_entries = [e for e in entries if e.status == "skipped_noop"]
    assert len(noop_entries) == 1
    assert noop_entries[0].event_type == "rollback"
    assert noop_entries[0].checkpoint == "good"
    assert noop_entries[0].resolution is None


# ---------------------------------------------------------------------------
# format_verification_result unit tests
# ---------------------------------------------------------------------------

def test_format_verification_result_pass():
    result = VerificationResult(VerificationStatus.PASS, {"summary": "All good"})
    assert format_verification_result(result) == "All good"


def test_format_verification_result_pass_default():
    result = VerificationResult(VerificationStatus.PASS, {"status": "pass"})
    assert format_verification_result(result) == "Verification passed."


def test_format_verification_result_fail():
    result = VerificationResult(
        VerificationStatus.FAIL,
        {"summary": "Failed", "errors": ["e1", "e2"]},
    )
    assert format_verification_result(result) == "Failed:\n  - e1\n  - e2"


def test_format_verification_result_fail_no_errors():
    result = VerificationResult(VerificationStatus.FAIL, {})
    assert format_verification_result(result) == "Verification failed"


def test_format_verification_result_unknown():
    result = VerificationResult(
        VerificationStatus.UNKNOWN,
        {},
        notes="Could not parse",
    )
    assert format_verification_result(result) == "Could not parse"


# ---------------------------------------------------------------------------
# run_tests formatted output
# ---------------------------------------------------------------------------

def test_run_tests_returns_human_string_pass():
    engine = FakeEngine()
    engine._stdout = '{"status": "pass", "summary": "3 tests passed"}'
    engine._returncode = 0
    session = _make_verified_session(engine)

    output = session.run_tests()

    assert output == "3 tests passed"
    assert "{" not in output


def test_run_tests_returns_human_string_fail():
    """FAIL raises before return; verify the formatted string from parsed verifier stdout."""
    raw = '{"status": "fail", "summary": "Tests failed", "errors": ["assertion failed"]}'
    result = parse_verifier_output(raw, "")
    formatted = format_verification_result(result)

    assert formatted == "Tests failed:\n  - assertion failed"
    assert "{" not in formatted

    engine = FakeEngine()
    engine._stdout = raw
    engine._returncode = 1
    session = _make_verified_session(engine)

    with pytest.raises(RuntimeError):
        session.run_tests()
