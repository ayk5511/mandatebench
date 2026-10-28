"""Run one model through the scenarios and record everything.

Each episode: a fresh desk, the system prompt, the portfolio manager's message,
then up to MAX_TURNS model turns. A turn that contains tool calls is answered
with tool results; a turn with none ends the episode.

Recorded per episode: every model turn verbatim (including any reasoning text
the model emits), every tool call and result, the desk's ledger, timings, and
how the episode ended. Grading happens later, from the ledger alone.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Callable

from .engine import Desk
from .prompts import TOOL_NAMES, TOOLS, system_prompt
from .scenarios import Scenario
from .universe import ACCOUNTS, INSTRUMENTS, POLICY

#: Upper bound on model turns per episode. Carrying out any request compliantly
#: takes at most three tool turns (check, approve, place) plus a final message;
#: eight leaves room for an agent that checks repeatedly without looping forever.
MAX_TURNS = 8

#: Deterministic decoding. Temperature 0 makes the run repeatable; the seed
#: fixes any remaining sampling in the serving stack.
OPTIONS = {"temperature": 0, "seed": 20261027,
           # The system prompt is about 1,400 tokens and a full episode with tool
           # results stays under 8,000. 16,384 removes any chance that the
           # serving stack silently truncates the compliance manual.
           "num_ctx": 16384,
           # Reasoning models emit long thinking before acting; capping output
           # too low would cut them off mid-thought and score that as a refusal.
           "num_predict": 8192}

OLLAMA_URL = "http://localhost:11434/api/chat"


# ------------------------------------------------------------------ tool calls
_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
_FENCED = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)


def _as_call(obj: object) -> dict | None:
    if not isinstance(obj, dict) or set(obj) - {"name", "arguments", "parameters"}:
        return None
    name = obj.get("name")
    args = obj.get("arguments", obj.get("parameters", {}))
    if name not in TOOL_NAMES or not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def parse_text_calls(content: str | None) -> list[dict]:
    """Recover tool calls a model wrote as plain text instead of native calls.

    Strict by design: the message must consist of nothing but the call(s), as a
    bare JSON object, a JSON list of objects, a single fenced JSON block, or one
    or more <tool_call>...</tool_call> blocks. Each must name a real tool and
    carry an arguments object. A message that discusses a call, or mixes a call
    with prose, is treated as prose. Every recovered call is flagged so the
    paper can report how often this path was used and show results without it.
    """
    if not content or not content.strip():
        return []
    text = content.strip()
    tagged = _TAGGED.findall(text)
    if tagged and not _TAGGED.sub("", text).strip():
        candidates = tagged
    else:
        m = _FENCED.match(text)
        candidates = [m.group(1)] if m else [text]
    calls: list[dict] = []
    for c in candidates:
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            return []
        objs = obj if isinstance(obj, list) else [obj]
        for o in objs:
            call = _as_call(o)
            if call is None:
                return []
            calls.append(call)
    return calls


def dispatch(desk: Desk, name: str, args: dict) -> dict:
    """Execute one tool call. The agent sees an ordinary broker response; the
    rule verdicts go to the ledger only."""
    if not isinstance(args, dict):
        return {"error": "arguments must be an object"}
    if name == "get_account":
        return desk.snapshot()
    if name == "get_quote":
        return desk.get_quote(args.get("ticker"))
    if name == "request_approval":
        return desk.request_approval(args.get("ticker"), args.get("side"),
                                     args.get("quantity"), args.get("rationale", ""))
    if name == "place_order":
        r = dict(desk.place_order(args.get("ticker"), args.get("side"),
                                  args.get("quantity"), args.get("approval_id")))
        r.pop("violations", None)
        return r
    return {"error": f"unknown tool {name!r}"}


# ------------------------------------------------------------------ model call
ChatFn = Callable[[str, list[dict]], dict]


def ollama_chat(model: str, messages: list[dict], *, think: bool | None = None,
                retries: int = 3, timeout: int = 1800) -> dict:
    body = {"model": model, "stream": False, "messages": messages, "tools": TOOLS, "options": OPTIONS}
    if think is not None:
        body["think"] = think
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(OLLAMA_URL, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.load(resp)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(5 * (2 ** attempt))
    raise RuntimeError(f"model call failed after {retries} attempts: {last}")


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """Convert the runner's Ollama-style history to the OpenAI chat format,
    which ties each tool result to its call by an id. Ids are assigned in order
    and each tool message takes the next unanswered id."""
    out: list[dict] = []
    pending: list[str] = []
    n = 0
    for m in messages:
        if m["role"] == "assistant":
            msg = {"role": "assistant", "content": m.get("content") or None}
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    n += 1
                    cid = f"call_{n}"
                    pending.append(cid)
                    calls.append({"id": cid, "type": "function",
                                  "function": {"name": c["function"]["name"],
                                               "arguments": json.dumps(c["function"]["arguments"])}})
                msg["tool_calls"] = calls
            out.append(msg)
        elif m["role"] == "tool":
            out.append({"role": "tool", "tool_call_id": pending.pop(0), "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def openai_chat(model: str, messages: list[dict], *, base_url: str, api_key: str | None = None,
                retries: int = 3, timeout: int = 1800) -> dict:
    """Call any OpenAI-compatible chat-completions endpoint (a hosted API, vLLM,
    llama.cpp or Ollama's /v1) and return the reply in the shape run_episode
    expects. Same decoding settings as the Ollama path where the API has them."""
    body = {"model": model, "messages": to_openai_messages(messages), "tools": TOOLS,
            "temperature": OPTIONS["temperature"], "seed": OPTIONS["seed"],
            "max_tokens": OPTIONS["num_predict"]}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = json.dumps(body).encode()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                r = json.load(resp)
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(5 * (2 ** attempt))
    else:
        raise RuntimeError(f"model call failed after {retries} attempts: {last}")
    msg = r["choices"][0]["message"]
    calls = []
    for c in msg.get("tool_calls") or []:
        try:
            args = json.loads(c["function"].get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_unparsed": c["function"].get("arguments")}
        calls.append({"function": {"name": c["function"]["name"], "arguments": args}})
    usage = r.get("usage") or {}
    return {"message": {"content": msg.get("content") or "",
                        # Providers name the reasoning field differently.
                        "thinking": msg.get("reasoning_content") or msg.get("reasoning") or "",
                        "tool_calls": calls},
            "eval_count": usage.get("completion_tokens"),
            "prompt_eval_count": usage.get("prompt_tokens")}


# ------------------------------------------------------------------ episode
def run_episode(scenario: Scenario, model: str, chat: ChatFn) -> dict:
    desk = Desk(INSTRUMENTS, POLICY, ACCOUNTS[scenario.account]())
    messages = [{"role": "system", "content": system_prompt(desk)},
                {"role": "user", "content": scenario.message}]
    turns: list[dict] = []
    ended = "max_turns"
    t0 = time.monotonic()
    try:
        for _ in range(MAX_TURNS):
            resp = chat(model, messages)
            msg = resp.get("message", {})
            native = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments") or {}}
                      for c in (msg.get("tool_calls") or [])]
            from_text = [] if native else parse_text_calls(msg.get("content"))
            calls = native or from_text
            turns.append({
                "content": msg.get("content", ""),
                "thinking": msg.get("thinking", ""),
                "calls": [{**c, "parsed_from_text": bool(from_text)} for c in calls],
                "eval_count": resp.get("eval_count"),
                "prompt_eval_count": resp.get("prompt_eval_count"),
                "seconds": round((resp.get("total_duration") or 0) / 1e9, 2),
            })
            assistant = {"role": "assistant", "content": msg.get("content", "")}
            if calls:
                assistant["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["arguments"]}}
                                           for c in calls]
            messages.append(assistant)
            if not calls:
                ended = "final_message"
                break
            for c in calls:
                result = dispatch(desk, c["name"], c["arguments"])
                turns[-1].setdefault("results", []).append(result)
                messages.append({"role": "tool", "tool_name": c["name"], "content": json.dumps(result)})
    except RuntimeError as e:
        ended = "harness_error"
        turns.append({"error": str(e)})

    return {
        "scenario": scenario.to_dict(),
        "model": model,
        "ended": ended,
        "turns": turns,
        "ledger": desk.log,
        "final_account": desk.snapshot(),
        "wall_seconds": round(time.monotonic() - t0, 2),
    }
