#!/usr/bin/env python3
"""Write the scenario set to scenarios/scenarios_v1.jsonl and record its SHA-256.

The paper cites this hash. Any change to a scenario changes the hash, so a
result can always be traced to the exact set it was produced on."""
import hashlib, json, pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))
from mandatebench.scenarios import build

out = pathlib.Path(__file__).resolve().parent.parent / "scenarios" / "scenarios_v1.jsonl"
lines = [json.dumps(s.to_dict(), sort_keys=True) for s in build()]
out.write_text("\n".join(lines) + "\n")
digest = hashlib.sha256(out.read_bytes()).hexdigest()
(out.parent / "scenarios_v1.sha256").write_text(f"{digest}  scenarios_v1.jsonl\n")
print(f"{len(lines)} scenarios -> {out.name}  sha256 {digest}")
