from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any


def dump_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def write_json_report(path: str | Path, payload: Any) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_json(payload) + "\n")
    return output_path


def read_json_report(
    path: str | Path,
    *,
    retries: int = 6,
    delay_sec: float = 0.05,
) -> Any:
    target = Path(path)
    attempts = max(retries, 1)
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return json.loads(target.read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                raise
            time.sleep(delay_sec)
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Failed to read JSON report: {target}")
