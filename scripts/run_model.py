#!/usr/bin/env python3
"""Run one model over the frozen scenario set. Resumable: scenario ids already
present in the output file are skipped, and harness errors are retried on the
next invocation rather than kept.

Two backends:
  ollama  (default) a model served by a local Ollama, as in the paper
  openai  any OpenAI-compatible chat-completions endpoint, given by --base-url;
          the API key, if one is needed, is read from MANDATEBENCH_API_KEY
"""
import argparse
import json
import os
import pathlib
import sys
import time
import urllib.request
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from mandatebench.runner import ollama_chat, openai_chat, run_episode
from mandatebench.scenarios import Request, Scenario

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", type=pathlib.Path, required=True)
ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
ap.add_argument("--base-url", help="for --backend openai, e.g. http://localhost:8000/v1")
ap.add_argument("--think", choices=["default", "on", "off"], default="default",
                help="Ollama reasoning switch; ignored by the openai backend")
ap.add_argument("--only", nargs="*", help="restrict to these scenario ids")
args = ap.parse_args()

if args.backend == "ollama":
    think = {"default": None, "on": True, "off": False}[args.think]
    # Record the serving stack. Runs that predate this field were all served by
    # Ollama 0.20.7.
    with urllib.request.urlopen("http://localhost:11434/api/version", timeout=10) as r:
        stack = {"ollama_version": json.load(r)["version"]}
    chat = lambda model, messages: ollama_chat(model, messages, think=think)
else:
    if not args.base_url:
        ap.error("--backend openai needs --base-url")
    key = os.environ.get("MANDATEBENCH_API_KEY")
    stack = {"backend": "openai", "base_url": args.base_url}
    chat = lambda model, messages: openai_chat(model, messages, base_url=args.base_url, api_key=key)

rows = [json.loads(l) for l in (ROOT / "scenarios" / "scenarios_v1.jsonl").read_text().splitlines()]
scen = [Scenario(**{**r, "request": Request(**r["request"])}) for r in rows]
if args.only:
    scen = [s for s in scen if s.id in set(args.only)]

done = set()
if args.out.exists():
    keep = []
    for l in args.out.read_text().splitlines():
        e = json.loads(l)
        if e["ended"] != "harness_error":
            done.add(e["scenario"]["id"]); keep.append(l)
    args.out.write_text("".join(k + "\n" for k in keep))
todo = [s for s in scen if s.id not in done]
print(f"model={args.model} backend={args.backend} think={args.think} {stack} todo={len(todo)} done={len(done)}",
      flush=True)

t0 = time.time()
with args.out.open("a") as fh:
    for i, s in enumerate(todo, 1):
        ep = run_episode(s, args.model, chat)
        ep["think"] = args.think
        ep.update(stack)
        fh.write(json.dumps(ep) + "\n"); fh.flush()
        el = time.time() - t0
        print(f"[{i}/{len(todo)}] {s.id:28s} {ep['ended']:14s} {ep['wall_seconds']:7.1f}s  eta {el/i*(len(todo)-i)/60:6.1f}m", flush=True)
