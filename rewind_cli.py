#!/usr/bin/env python3
import argparse
import json
import os
import sys

from rewind_sdk import RewindSession
from rewind_sdk.verification import VerificationHaltError


EXIT_SUCCESS = 0
EXIT_ERROR_GENERIC = 1
EXIT_ERROR_CONTAINER_NOT_FOUND = 2
EXIT_ERROR_INVALID_CHECKPOINT = 3
EXIT_ERROR_DOCKER_FAILURE = 4
EXIT_ERROR_INVALID_PATH = 5


class AgentNativeCLI:
    def __init__(self, container_name, use_json=False, quiet=False):
        self.session = RewindSession(container_name=container_name, destroy_on_exit=False)
        self.use_json = use_json
        self.quiet = quiet

    def log_info(self, msg):
        if not self.quiet:
            print(f"[*] {msg}", file=sys.stderr)

    def log_result(self, data, success=True):
        if self.use_json:
            print(json.dumps({"success": success, "data": data}))
        elif isinstance(data, dict):
            for key, value in data.items():
                print(f"{key}: {value}")
        else:
            print(data)

    def log_error(self, msg, code=EXIT_ERROR_GENERIC):
        if self.use_json:
            print(json.dumps({"success": False, "error": msg, "code": code}))
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(code)

    def cmd_init(self, path, force=False):
        if not os.path.exists(path):
            self.log_error(f"Path does not exist: {path}", EXIT_ERROR_INVALID_PATH)

        path = os.path.abspath(path)
        try:
            if self.session.engine.container_exists() and not force:
                self.log_info(f"Sandbox '{self.session.name}' already exists. Attaching...")
                if self.session.engine.load_metadata():
                    self.session._started = True
                    self.log_result(self.session.status())
                    return
                self.log_info("Existing container metadata is missing. Re-initializing...")

            self.log_info(f"Initializing sandbox at {path}...")
            self.session.start(path, force=True)
            self.log_result({"path": path, "status": "initialized"})
        except Exception as exc:
            self.log_error(f"Docker failure during init: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def _ensure_attached(self):
        try:
            self.session.attach()
        except Exception:
            self.log_error(f"No active sandbox found: {self.session.name}", EXIT_ERROR_CONTAINER_NOT_FOUND)

    def cmd_exec(self, cmd):
        self._ensure_attached()
        try:
            self.log_result(self.session.run(cmd))
        except VerificationHaltError as exc:
            msg = (
                f"Execution halted by verifier.\n"
                f"  Checkpoint : {exc.checkpoint or 'none'}\n"
                f"  Command    : {exc.verifier_command if isinstance(exc.verifier_command, str) else ' '.join(exc.verifier_command)}\n"
                f"  Details    : {exc.last_result.notes or 'none'}\n"
                "The sandbox container is still alive. Inspect it, fix the verifier, "
                "then re-run your script to resume."
            )
            self.log_error(msg, EXIT_ERROR_GENERIC)
        except Exception as exc:
            self.log_error(f"Execution failed: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def cmd_read(self, path):
        self._ensure_attached()
        try:
            self.log_result(self.session.read_file(path))
        except Exception as exc:
            self.log_error(f"Read failed: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def cmd_write(self, path, content):
        self._ensure_attached()
        try:
            target = self.session.write_file(path, content)
            self.log_result({"path": path, "target": target, "status": "written"})
        except Exception as exc:
            self.log_error(f"Write failed: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def cmd_checkpoint(self, name):
        self._ensure_attached()
        if name in self.session.engine.checkpoint_history:
            self.log_error(f"Checkpoint '{name}' already exists.", EXIT_ERROR_INVALID_CHECKPOINT)
        try:
            self.session.checkpoint(name)
            self.log_result({"checkpoint": name, "status": "created"})
        except ValueError as exc:
            self.log_error(str(exc), EXIT_ERROR_INVALID_CHECKPOINT)
        except Exception as exc:
            self.log_error(f"Checkpoint failed: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def cmd_rollback(self, name):
        self._ensure_attached()
        if name not in self.session.engine.checkpoint_history:
            self.log_error(f"Checkpoint '{name}' not found.", EXIT_ERROR_INVALID_CHECKPOINT)
        try:
            self.session.engine.rollback_to_checkpoint(name)
            self.log_result({"checkpoint": name, "status": "restored"})
        except Exception as exc:
            self.log_error(f"Rollback failed: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def cmd_status(self):
        self._ensure_attached()
        try:
            self.log_result(self.session.status())
        except Exception as exc:
            self.log_error(f"Status check failed: {exc}", EXIT_ERROR_DOCKER_FAILURE)

    def cmd_destroy(self):
        self.log_info(f"Destroying sandbox '{self.session.name}'...")
        self.session.destroy()
        self.log_result("SUCCESS: Sandbox destroyed")

    def cmd_ledger(self, checkpoint=None):
        self._ensure_attached()
        if checkpoint:
            entries = self.session.ledger.by_checkpoint(checkpoint)
        else:
            entries = self.session.ledger.history()

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

        if self.use_json:
            print(json.dumps({"success": True, "data": serialised}))
            return

        if not serialised:
            print("No ledger entries recorded in this session.")
            return

        for i, entry in enumerate(serialised, 1):
            print(f"--- Entry {i} ---")
            print(f"  timestamp  : {entry['timestamp']}")
            print(f"  event_type : {entry['event_type']}")
            print(f"  status     : {entry['status'] or '-'}")
            print(f"  checkpoint : {entry['checkpoint'] or '-'}")
            print(f"  resolution : {entry['resolution'] or '-'}")
            if entry["notes"]:
                print(f"  notes      : {entry['notes']}")
            if entry["raw_output"]:
                print(f"  raw_output : {json.dumps(entry['raw_output'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Rewind Time-Travel Sandbox CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AGENT INTEGRATION GUIDE:
  - Use --json for programmatic state management.
  - Use --quiet to suppress logs; stdout will contain only the JSON result.
  - Prefer the Python SDK for atomic filesystem + memory rollback.
        """,
    )
    parser.add_argument("--container-name", default="rewind_sandbox", help="Target sandbox instance name")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output on stdout")
    parser.add_argument("--quiet", action="store_true", help="Minimize logging to stderr")

    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    parser_init = subparsers.add_parser("init", help="Initialize sandbox")
    parser_init.add_argument("path", help="Path to base host directory")
    parser_init.add_argument("--force", action="store_true", help="Wipe and re-initialize if exists")

    parser_exec = subparsers.add_parser("exec", help="Run command inside workspace")
    parser_exec.add_argument("cmd", help="Shell command to execute")

    parser_read = subparsers.add_parser("read", help="Read a workspace file")
    parser_read.add_argument("path", help="Path relative to the sandbox workspace")

    parser_write = subparsers.add_parser("write", help="Write a workspace file")
    parser_write.add_argument("path", help="Path relative to the sandbox workspace")
    parser_write.add_argument("content", help="Text content to write")

    parser_checkpoint = subparsers.add_parser("checkpoint", help="Freeze current filesystem state")
    parser_checkpoint.add_argument("name", help="Checkpoint label")

    parser_rollback = subparsers.add_parser("rollback", help="Restore filesystem state")
    parser_rollback.add_argument("name", help="Checkpoint label to restore")

    subparsers.add_parser("status", help="Query layer depth and usage")
    subparsers.add_parser("destroy", help="Destroy sandbox")

    parser_ledger = subparsers.add_parser("ledger", help="Show verification ledger history")
    parser_ledger.add_argument(
        "--checkpoint", default=None, help="Filter entries by checkpoint label"
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(EXIT_SUCCESS)

    cli = AgentNativeCLI(args.container_name, use_json=args.json, quiet=args.quiet)

    if args.command == "init":
        cli.cmd_init(args.path, force=args.force)
    elif args.command == "exec":
        cli.cmd_exec(args.cmd)
    elif args.command == "read":
        cli.cmd_read(args.path)
    elif args.command == "write":
        cli.cmd_write(args.path, args.content)
    elif args.command == "checkpoint":
        cli.cmd_checkpoint(args.name)
    elif args.command == "rollback":
        cli.cmd_rollback(args.name)
    elif args.command == "status":
        cli.cmd_status()
    elif args.command == "destroy":
        cli.cmd_destroy()
    elif args.command == "ledger":
        cli.cmd_ledger(checkpoint=getattr(args, "checkpoint", None))


if __name__ == "__main__":
    main()
