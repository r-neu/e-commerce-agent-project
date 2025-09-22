"""
Validate chat-style JSONL files for SFT.
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple


def validate_row(obj: Dict) -> Tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Row is not a JSON object"
    msgs = obj.get("messages")
    if not isinstance(msgs, list):
        return False, "Missing or invalid 'messages' list"
    if len(msgs) < 3:
        return False, "'messages' must have at least 3 entries (system, user, assistant)"
    roles = [m.get("role") for m in msgs]
    contents = [m.get("content") for m in msgs]
    # Check role ordering and presence
    if roles[0] != "system" or roles[1] != "user" or roles[2] != "assistant":
        return False, "First three roles must be system, user, assistant"
    # Check string content
    for i, (r, c) in enumerate(zip(roles, contents)):
        if not isinstance(r, str) or not isinstance(c, str):
            return False, f"Message {i} missing string role/content"
        if not c.strip():
            return False, f"Message {i} content is empty"
    return True, ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate chat-style JSONL for SFT")
    parser.add_argument("--input", required=True)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    total = ok = 0
    errors: List[str] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"L{lineno}: invalid JSON: {e}")
                continue
            valid, msg = validate_row(obj)
            if valid:
                ok += 1
            else:
                errors.append(f"L{lineno}: {msg}")
            if len(errors) >= args.max_errors:
                break

    print(f"Checked {total} rows: OK={ok}, ERRORS={len(errors)}")
    if errors:
        print("Sample errors:")
        for e in errors:
            print("  ", e)
    # Exit code nonzero on errors
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main() 