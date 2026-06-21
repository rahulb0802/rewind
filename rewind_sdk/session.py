import os
import time
import warnings
from dataclasses import dataclass

from .adapters.langgraph import dicts_to_messages, infer_message_format, messages_to_dicts
from .engine import SandboxEngine
from .memory import MemoryStore
from .verification import (
    EscalationContext,
    EscalationHandler,
    EscalationResolution,
    VerificationHaltError,
    VerificationLedger,
    VerificationStatus,
    VerifierConfig,
    run_verifier,
    stdin_escalation_handler,
)


@dataclass
class AutoCheckpointConfig:
    trigger: str
    keep_last: int | None = None


@dataclass
class AutoRollbackConfig:
    events: set[str]
    to: str = "latest"
    test_command: str | None = None
    verifier: VerifierConfig | None = None


class RewindSession:
    """Unified developer-facing API for filesystem and memory time travel."""

    def __init__(
        self,
        name="rewind_sandbox",
        workspace=".",
        *,
        container_name=None,
        engine=None,
        memory=None,
        destroy_on_exit=True,
        auto_commit=False,
        escalation_handler: EscalationHandler | None = None,
    ):
        self.name = container_name or name
        self.workspace = workspace
        self.engine = engine or SandboxEngine(container_name=self.name)
        self.memory = memory or MemoryStore()
        self.destroy_on_exit = destroy_on_exit
        self._auto_checkpoint = None
        self._auto_rollback = None
        self._auto_counter = 0
        self._auto_labels = []
        self._message_format = "dict"
        self.last_auto_rollback = None
        self._started = False
        self.auto_commit = auto_commit
        self._escalation_handler: EscalationHandler = escalation_handler or stdin_escalation_handler
        # Durable ledger: lives outside both engine and memory rollback scopes.
        # rollback() never touches this attribute.
        self.ledger = VerificationLedger()

    def __enter__(self):
        self.start(force=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        # Only auto-commit if the agent loop finished successfully
        if exc_type is None and self.auto_commit:
            try:
                self.commit()
            except Exception as e:
                print(f"Warning: Failed to auto-commit: {e}")
        elif exc_type is None and not self.auto_commit:
            warnings.warn(
                f"Session '{self.name}' exited without auto_commit; no changes persisted to '{self.workspace}'.",
                stacklevel=2,
            )
                
        if self.destroy_on_exit:
            self.destroy()
        return False


    def start(self, workspace=None, *, force=False):
        if workspace is not None:
            self.workspace = workspace

        if not os.path.exists(self.workspace):
            os.makedirs(self.workspace, exist_ok=True)

        if self.engine.container_exists() and not force:
            if self.engine.load_metadata():
                self._started = True
                return self

        self.engine.init_sandbox(self.workspace)
        self._started = True
        return self

    def attach(self):
        if not self.engine.load_metadata():
            raise RuntimeError(f"No active sandbox found: {self.name}")
        self._started = True
        return self

    def destroy(self):
        self.engine.destroy_sandbox()
        self._started = False

    def run(self, cmd):
        self._ensure_ready()
        try:
            return self.engine.run_cmd(cmd)
        except Exception as exc:
            event = "test_failure" if self._looks_like_test_command(cmd) else "exception"
            self._maybe_auto_rollback(event, patch_notes=str(exc))
            raise

    def run_tests(self, cmd=None):
        command = cmd or self._default_test_command()
        try:
            self._ensure_ready()
            return self.engine.run_cmd(command)
        except Exception as exc:
            self._maybe_auto_rollback("test_failure", patch_notes=str(exc))
            raise

    def write_file(self, path, content):
        self._ensure_ready()
        return self.engine.write_file(path, content)

    def read_file(self, path):
        self._ensure_ready()
        return self.engine.read_file(path)

    def sync_memory(self, messages, message_format="auto"):
        """Track framework-native messages without exposing index bookkeeping."""
        resolved_format = (
            infer_message_format(messages) if message_format == "auto" else message_format
        )
        self._message_format = resolved_format
        self.memory.update(messages_to_dicts(messages))
        return self.get_messages(message_format=resolved_format)

    def get_messages(self, message_format="auto"):
        resolved_format = self._message_format if message_format == "auto" else message_format
        return dicts_to_messages(self.memory.get_messages(), message_format=resolved_format)

    def checkpoint(self, label, messages=None):
        self._ensure_ready()
        if messages is not None:
            self.sync_memory(messages)
        self.engine.create_checkpoint(label)
        self.memory.snapshot(label)
        return label

    def rollback(self, label="latest", patch_notes=None, message_format="auto"):
        self._ensure_ready()
        resolved_label = self._resolve_label(label)
        self.engine.rollback_to_checkpoint(resolved_label)
        messages = self.memory.rollback(resolved_label, patch_notes=patch_notes)
        resolved_format = self._message_format if message_format == "auto" else message_format
        return dicts_to_messages(messages, message_format=resolved_format)

    def status(self):
        self._ensure_ready()
        status = self.engine.get_status()
        status["memory_checkpoints"] = self.memory.checkpoints()
        return status

    def auto_checkpoint(self, trigger="before_tool_call", keep_last=None):
        self._auto_checkpoint = AutoCheckpointConfig(trigger=trigger, keep_last=keep_last)
        return self

    def auto_rollback(self, *events, on=None, to=None, test_command=None):
        """Configure automatic rollback behavior on specified events.
        IMPORTANT: 
        `to` should almost always be an explicit checkpoint label you
        created with `session.checkpoint(...)` BEFORE the operation began
        (for example to="pre_migration"), not the default "latest" auto-checkpoint.

        Auto-checkpoints are taken immediately BEFORE each tool call, which means
        the most recent auto-checkpoint can already contain the very change that
        caused the failure you're rolling back from. Passing to="latest" rolls
        back to that checkpoint.

        Use to="latest" only if you understand this and specifically want "undo
        just the last tool call.
        """
        if to is None:
            to = "latest"
            warnings.warn(
                'auto_rollback() called without an explicit `to=` checkpoint label. '
                'Defaulting to to="latest", which rolls back to the most recent '
                'auto-checkpoint; this may already contain the change that caused '
                'the failure. Pass an explicit checkpoint label (for example, '
                'to="my_known_good_label") created before the risky operation for '
                'correct recovery behavior.',
                stacklevel=2,
            )
        if on is not None:
            warnings.warn(
                "auto_rollback(on=...) is deprecated; pass event names positionally, "
                'e.g. auto_rollback("exception", "test_failure").',
                DeprecationWarning,
                stacklevel=2,
            )
        selected_events = self._normalize_rollback_events(events, on)
        self._auto_rollback = AutoRollbackConfig(
            events=selected_events,
            to=to,
            test_command=test_command,
        )
        return self

    def on_tool_call(self, messages=None, tool_name=None):
        if messages is not None:
            self.sync_memory(messages)

        if self._auto_checkpoint and self._auto_checkpoint.trigger == "before_tool_call":
            label = self._next_auto_label(tool_name)
            self.checkpoint(label)
            self._remember_auto_label(label)
            return label
        return None

    def on_tool_result(self, messages=None, error=None):
        if messages is not None:
            self.sync_memory(messages)

        if (
            error is not None
            and self._auto_rollback
            and "exception" in self._auto_rollback.events
        ):
            return self._maybe_auto_rollback("exception", patch_notes=str(error))
        return None

    def _ensure_ready(self):
        if not self._started and not self.engine.load_metadata():
            raise RuntimeError(
                f"Session '{self.name}' is not started. Use 'with rewind.session(...)' "
                "or call session.start()."
            )
        self._started = True

    def _resolve_label(self, label):
        if label != "latest":
            return label
        if self._auto_labels:
            return self._auto_labels[-1]
        memory_label = self.memory.latest_label()
        if memory_label:
            return memory_label
        if self.engine.checkpoint_history:
            return self.engine.checkpoint_history[-1]
        raise ValueError("No checkpoint is available.")

    def _next_auto_label(self, tool_name=None):
        self._auto_counter += 1
        clean_tool = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in (tool_name or "tool"))
        return f"auto_{self._auto_counter:04d}_{clean_tool}_{int(time.time() * 1000)}"

    def _remember_auto_label(self, label):
        self._auto_labels.append(label)
        # OverlayFS layers remain physically present for rollback correctness; this
        # retention window controls only the session's convenience label history.
        if self._auto_checkpoint and self._auto_checkpoint.keep_last:
            self._auto_labels = self._auto_labels[-self._auto_checkpoint.keep_last :]

    def _normalize_rollback_events(self, events, on):
        selected = []
        if events:
            selected.extend(events)
        if on is not None:
            if isinstance(on, str):
                selected.append(on)
            else:
                selected.extend(on)
        if not selected:
            selected.append("exception")
        return {str(event) for event in selected}

    def _maybe_auto_rollback(self, event, patch_notes=None):
        if not self._auto_rollback or event not in self._auto_rollback.events:
            return None

        verifier = self._auto_rollback.verifier
        if verifier is None:
            # Backward-compatible path: no verifier configured, rollback immediately.
            return self._execute_rollback(event, patch_notes)

        result, attempts = self._run_verifier_with_retries(verifier)

        try:
            checkpoint_label = self._resolve_label(self._auto_rollback.to)
        except ValueError:
            checkpoint_label = None

        if result.status == VerificationStatus.PASS:
            self.ledger.record_verification(
                status=result.status,
                checkpoint=checkpoint_label,
                raw_output=result.raw_output,
                notes=result.notes,
            )
            return None

        if result.status == VerificationStatus.FAIL:
            self.ledger.record_verification(
                status=result.status,
                checkpoint=checkpoint_label,
                raw_output=result.raw_output,
                notes=result.notes,
            )
            return self._execute_rollback(event, patch_notes)

        # UNKNOWN after all retries exhausted.
        return self._handle_unknown_escalation(event, patch_notes, result, attempts, checkpoint_label)

    def _run_verifier_with_retries(self, config: VerifierConfig):
        """Run the verifier, retrying on UNKNOWN with backoff.

        Returns (VerificationResult, total_attempts).
        """
        total = config.retries + 1
        for attempt in range(1, total + 1):
            result = run_verifier(config)
            if result.status != VerificationStatus.UNKNOWN:
                return result, attempt
            if attempt < total:
                print(
                    f"[rewind] Verifier returned unknown "
                    f"(attempt {attempt}/{total}), retrying in {config.retry_delay}s..."
                )
                time.sleep(config.retry_delay)
            else:
                print(
                    f"[rewind] Verifier exhausted {config.retries} retries, "
                    "still unknown. Escalating to human decision point..."
                )
        return result, total

    def _execute_rollback(self, event, patch_notes, result=None):
        """Perform the actual filesystem + memory rollback and update session state."""
        try:
            checkpoint_label = self._resolve_label(self._auto_rollback.to)
        except ValueError:
            checkpoint_label = None

        try:
            messages = self.rollback(
                self._auto_rollback.to,
                patch_notes=patch_notes or f"Automatic rollback triggered by {event}.",
            )
        except ValueError:
            warnings.warn(
                f"Auto-rollback was triggered by '{event}' but no checkpoint exists yet "
                "to roll back to. The error was not recovered; it will propagate normally. "
                "Call session.checkpoint(...) before this point if you want auto-rollback "
                "to be able to act here.",
                stacklevel=2,
            )
            return None

        self.ledger.record_rollback(
            checkpoint=checkpoint_label,
            notes=patch_notes or f"Automatic rollback triggered by {event}.",
        )
        self.last_auto_rollback = {
            "event": event,
            "to": self._auto_rollback.to,
            "messages": messages,
        }
        return messages

    def _handle_unknown_escalation(self, event, patch_notes, result, attempts, checkpoint_label):
        """Dispatch to the escalation handler and act on its resolution."""
        cmd = self._auto_rollback.verifier.command
        ctx = EscalationContext(
            checkpoint=checkpoint_label,
            verifier_command=cmd,
            last_result=result,
            attempts=attempts,
            session_name=self.name,
        )
        resolution = self._escalation_handler(ctx)

        self.ledger.record_escalation(
            status=result.status,
            checkpoint=checkpoint_label,
            raw_output=result.raw_output,
            notes=result.notes,
            resolution=resolution,
        )

        if resolution == EscalationResolution.CONTINUE:
            return None

        if resolution == EscalationResolution.ROLLBACK:
            return self._execute_rollback(event, patch_notes)

        # STOP: surface what the developer needs to manually recover.
        cmd_display = cmd if isinstance(cmd, str) else " ".join(cmd)
        raise VerificationHaltError(
            f"[rewind] Execution halted — verifier returned UNKNOWN after {attempts} attempt(s).\n"
            f"  Checkpoint : {checkpoint_label or 'none'}\n"
            f"  Command    : {cmd_display}\n"
            f"  Details    : {result.notes or 'none'}\n"
            "Sandbox container is still alive. Fix the verifier and re-run your script to resume.",
            checkpoint=checkpoint_label,
            verifier_command=cmd,
            last_result=result,
        )

    def _default_test_command(self):
        if self._auto_rollback and self._auto_rollback.test_command:
            return self._auto_rollback.test_command
        return "pytest"

    def _looks_like_test_command(self, cmd):
        configured = self._default_test_command()
        if isinstance(cmd, str):
            return configured in cmd or "pytest" in cmd or "unittest" in cmd
        joined = " ".join(str(part) for part in cmd)
        return configured in joined or "pytest" in joined or "unittest" in joined
    
    def get_ledger(self) -> VerificationLedger:
        """Return the durable ledger; survives rollback() calls."""
        return self.ledger

    def commit(self):
        """Commits the sandbox workspace state back to the host machine."""
        self._ensure_ready()
        self.engine.commit(self.workspace) # delegate to engine


def session(name="rewind_sandbox", workspace=".", **kwargs):
    return RewindSession(name=name, workspace=workspace, **kwargs)
