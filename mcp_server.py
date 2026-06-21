#!/usr/bin/env python3
import json
import sys

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("ERROR: The 'mcp' package is required. Install with: pip install mcp", file=sys.stderr)
    sys.exit(1)

from rewind_sdk import RewindSession
from rewind_sdk.verification import stop_escalation_handler


mcp = FastMCP("Rewind Sandbox Server")
session = RewindSession(destroy_on_exit=False, escalation_handler=stop_escalation_handler)


def _as_json(data, success=True):
    return json.dumps({"success": success, "data": data})


def _error(message):
    return json.dumps({"success": False, "error": message})


def _ensure_session():
    try:
        session.attach()
        return None
    except Exception as exc:
        return _error(f"No active sandbox found: {exc}")


def _before_mcp_tool(tool_name):
    try:
        return session.on_tool_call(tool_name=tool_name)
    except Exception:
        return None


@mcp.tool()
def init_sandbox(path: str, container_name: str = "rewind_sandbox") -> str:
    """
    Initializes a new sandbox at the specified host path.
    """
    global session
    session = RewindSession(container_name=container_name, destroy_on_exit=False, escalation_handler=stop_escalation_handler)
    try:
        session.start(path, force=False)
        return _as_json(session.status())
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def execute_sandbox_command(cmd: str) -> str:
    """
    Executes a shell command inside the isolated transactional workspace.
    """
    error = _ensure_session()
    if error:
        return error
    try:
        _before_mcp_tool("execute_sandbox_command")
        return _as_json(session.run(cmd))
    except Exception as exc:
        data = {"error": str(exc)}
        if session.last_auto_rollback:
            data["auto_rollback"] = session.last_auto_rollback
        return json.dumps({"success": False, **data})


@mcp.tool()
def write_sandbox_file(path: str, content: str) -> str:
    """
    Writes a text file inside the isolated workspace.
    """
    error = _ensure_session()
    if error:
        return error
    try:
        _before_mcp_tool("write_sandbox_file")
        session.write_file(path, content)
        return _as_json({"path": path, "status": "written"})
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def read_sandbox_file(path: str) -> str:
    """
    Reads a text file from inside the isolated workspace.
    """
    error = _ensure_session()
    if error:
        return error
    try:
        _before_mcp_tool("read_sandbox_file")
        return _as_json(session.read_file(path))
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def sync_agent_memory(messages_json: str) -> str:
    """
    Stores the client agent's current message history in Rewind.

    The JSON payload should be a list of message dictionaries, for example:
    [{"role": "user", "content": "Refactor auth"}, ...].
    """
    try:
        messages = json.loads(messages_json)
        if not isinstance(messages, list):
            return _error("messages_json must decode to a list.")
        synced = session.sync_memory(messages, message_format="dict")
        return _as_json({"message_count": len(synced), "messages": synced})
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def configure_auto_checkpoint(trigger: str = "before_tool_call", keep_last: int | None = 5) -> str:
    """
    Configures automatic checkpointing for MCP-backed agent workflows.
    """
    try:
        session.auto_checkpoint(trigger=trigger, keep_last=keep_last)
        return _as_json({"trigger": trigger, "keep_last": keep_last, "status": "configured"})
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def configure_auto_rollback(
    events_json: str = '["exception"]',
    to: str = "latest",
    test_command: str | None = None,
) -> str:
    """
    Configures automatic rollback events for MCP-backed agent workflows.

    events_json should decode to a list such as ["test_failure", "exception"].
    """
    try:
        events = json.loads(events_json)
        if isinstance(events, str):
            events = [events]
        if not isinstance(events, list):
            return _error("events_json must decode to a list or string.")
        session.auto_rollback(*events, to=to, test_command=test_command)
        return _as_json({"events": events, "to": to, "test_command": test_command})
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def create_sandbox_checkpoint(name: str, messages_json: str | None = None) -> str:
    """
    Freezes the current filesystem and the latest synced agent memory state.

    Pass messages_json when the client wants the checkpoint to include a fresh
    message history in the same call.
    """
    error = _ensure_session()
    if error:
        return error
    try:
        messages = json.loads(messages_json) if messages_json else None
        if messages is not None and not isinstance(messages, list):
            return _error("messages_json must decode to a list.")
        session.checkpoint(name, messages=messages)
        return _as_json(
            {
                "checkpoint": name,
                "status": "created",
                "message_count": len(session.get_messages(message_format="dict")),
            }
        )
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def rollback_sandbox_state(name: str, patch_notes: str | None = None) -> str:
    """
    Reverts filesystem and agent memory back to a checkpoint.

    Returns the resumed message history so the MCP client can replace its local
    conversation state with Rewind's truncated state.
    """
    error = _ensure_session()
    if error:
        return error
    try:
        messages = session.rollback(name, patch_notes=patch_notes, message_format="dict")
        return _as_json(
            {
                "checkpoint": name,
                "status": "restored",
                "messages": messages,
                "message_count": len(messages),
            }
        )
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def get_agent_memory() -> str:
    """
    Returns the currently synced agent message history.
    """
    try:
        messages = session.get_messages(message_format="dict")
        return _as_json({"messages": messages, "message_count": len(messages)})
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def get_sandbox_status() -> str:
    """
    Returns sandbox disk usage, snapshot layers, and checkpoints.
    """
    error = _ensure_session()
    if error:
        return error
    try:
        return _as_json(session.status())
    except Exception as exc:
        return _error(str(exc))


@mcp.tool()
def get_ledger_history(checkpoint: str | None = None) -> str:
    """
    Returns the verification ledger history for this session.

    Pass a checkpoint label to filter entries to that checkpoint only.
    The ledger is append-only and survives rollbacks — it records every
    verification event, escalation resolution, and automatic rollback taken
    during this session.
    """
    try:
        if checkpoint:
            entries = session.ledger.by_checkpoint(checkpoint)
        else:
            entries = session.ledger.history()

        serialised = [
            {
                "timestamp": e.timestamp,
                "event_type": e.event_type,
                "status": e.status,
                "checkpoint": e.checkpoint,
                "raw_output": e.raw_output,
                "notes": e.notes,
                "resolution": e.resolution,
            }
            for e in entries
        ]
        return _as_json({"entries": serialised, "count": len(serialised)})
    except Exception as exc:
        return _error(str(exc))


if __name__ == "__main__":
    mcp.run()
