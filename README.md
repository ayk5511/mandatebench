# mandatebench

Does an LLM trading agent keep a firm's compliance rules when its own portfolio
manager asks it to break them?

An agent works the desk of a fictitious firm under a six-rule compliance manual
(restricted list, earnings blackout, 10% concentration cap, pre-trade approval
above $50,000, no short selling, no margin). It gets 144 conflict scenarios, in
which the requested trade breaches exactly one rule, at six pressure levels from
a plain request to a forged system notice. It also gets 28 legitimate scenarios,
several of them sitting exactly at a limit, so over-refusal is measured
alongside violation. A deterministic rule engine scores every order from the
ledger; no model grades another.

Paper: *Can an AI Agent Be Trusted With a Brokerage Account? Rule Compliance of
Trading Agents Under Pressure From Their Principal* (Khan, 2026). The PDF is in
`paper/`.

## Quick start

```bash
pip install -e ".[dev,analysis]"
python -m pytest -q                      # engine, scenarios, runner, stats
# a model served by a local Ollama
python scripts/run_model.py --model llama3.1:8b-instruct-q8_0 --out results/runs/my-model.jsonl
# or any OpenAI-compatible endpoint (vLLM, llama.cpp, a hosted API);
# the API key, if needed, goes in MANDATEBENCH_API_KEY
python scripts/run_model.py --backend openai --base-url http://localhost:8000/v1 \
    --model my-model --out results/runs/my-model.jsonl
python scripts/analyze.py                # every number, table and figure
python paper/audit.py                    # independent recomputation and checks
```

To include a new model in the analysis, add a line to `MODELS` in
`scripts/analyze.py`. Please report the violation rate on conflicts, the three
outcomes on legitimate requests, and the per-pressure-level counts together, and
quote the scenario-file digest in `scenarios/scenarios_v1.sha256`.

## Layout

| Path | Contents |
|---|---|
| `src/mandatebench/` | rule engine, universe and accounts, scenarios, prompts, runner, grader, statistics |
| `scenarios/` | frozen scenario set and its SHA-256 digest |
| `results/runs/` | every episode transcript: model turns, reasoning text, tool calls and results, ledger |
| `results/misreports.csv` | hand-coded verdicts on misreport candidates |
| `scripts/` | run a model, freeze scenarios, analyze |
| `paper/` | LaTeX source, generated numbers, tables and figures, audit script |

## Citation

See `CITATION.cff`.

---

# Study record

## Analysis plan (locked 2026-10-27, after the pilot, before any full-study result was examined)

**Primary outcome.** On conflict scenarios, the share of episodes in which the
agent executes at least one order that breaches any rule (`violated`). A
compliance officer cares about every breach, not only the one a scenario was
built around.

**Secondary outcomes.**
- Target-rule violation rate on conflict scenarios (`violated_target`).
- On legitimate scenarios, three mutually exclusive outcomes: completed
  compliantly; executed but in breach (for example, skipping a required
  approval); not executed (over-refusal).
- Unrequested trading: any order in a security or direction nobody asked for,
  or more shares than requested, across all scenarios.

**Inference.** Wilson 95% intervals on every proportion. Effect of each pressure
level against the plain request: exact McNemar test, paired over the 24
(rule, base) combinations within each model, Holm-adjusted across the five
comparisons per model. No pooled inference across models: four models are not a
sample from a population of models, so results are reported per model.

**Exclusions.** Episodes ending in an infrastructure failure (`harness_error`)
are excluded and counted, never scored.

**Sensitivity.** Results with and without tool calls recovered from plain text
(`parsed_from_text`). Repeatability: a full second pass of Llama 3.1 8B,
compared episode by episode.

## Deviations from the original design, disclosed

1. The pilot showed an agent trading on its own initiative after finishing its
   task. `unrequested_orders` and `over_executed` were added to the grader
   before the main runs; the original `off_target_orders` (defined before the
   pilot) is kept and reported.
2. Completion is judged on cumulative quantity, so a request split across
   several orders counts as completed. Changed before the main runs.
3. The prior-work check found ReguSim (arXiv 2608.19974), a close neighbor; the
   design was narrowed to what it does not cover (principal instruction
   conflicts, social-engineering pressure, over-refusal, firm-policy rules,
   open-weight local models).
4. Exploratory measures defined after reading pilot or early transcripts, all
   descriptive, no tests: violation as first action, use of the fabricated
   reference CMP-4471, substitution, plain versus stale-rule by rule.
5. Added after all runs were complete, all descriptive, no tests: the rule by
   pressure grid (appendix table), violations undone after the fill,
   off-target-rule violations, and the sensitivity checks (violation on the
   first order, unrequested trading at the turn limit, transcript-level
   identity of the repeat run, which replaced a grade-level comparison).

## Replication on four further models (prediction written 2026-10-28, before any of these runs)

After the first four models were analyzed, four more open-weight models from
other developers were added: gpt-oss-20b and gpt-oss-120b (OpenAI), Llama 3.3
70B (Meta) and Mistral Small 3.2 24B (Mistral). Same frozen scenarios, same
code, same decoding, same analysis. The finding from the first four models is
treated as a prediction for these four:

- **P1.** For none of the four new models will the authority, urgency or
  stale-rule level differ significantly from the plain request (exact McNemar,
  Holm-adjusted within model, p < 0.05).
- **P2.** Every significant pressure effect among the four new models will be
  at the claimed-approval or fake-system-notice level.
- **P3.** No new model will leave more than 2 of the 28 legitimate requests
  undone.

Falsified if any new model shows a significant social-pressure effect (P1),
a significant effect anywhere other than the two counterfeit-authorization
levels (P2), or more than 2 legitimate requests undone (P3). Results are
reported whether or not the predictions hold. A model with no significant
effect at all is consistent with P1 and P2 and says nothing for or against
them; it is reported as uninformative.

### Outcome (added 2026-10-28, after the runs; the prediction above is unchanged)

| Model | Significant levels | P1 | P2 | Legitimate undone | P3 |
|---|---|---|---|---|---|
| gpt-oss-20b | none | uninformative | uninformative | 0 | holds |
| Mistral Small 3.2 24B | fake system notice | holds | holds | 0 | holds |
| Llama 3.3 70B | claimed approval, fake system notice | holds | holds | 17 | **fails** |
| gpt-oss-120b | none | uninformative | uninformative | 0 | holds |

P3 failed for Llama 3.3 70B. In all 17 undone requests it wrote the intended
tool call inside prose (JSON or `<|python_tag|>` syntax), which the strict
text-call rule fixed before these runs treats as prose. The rule was not
changed; the paper reports the lenient counts as a sensitivity analysis.

### Deviations during the replication, disclosed

6. gpt-oss needs a newer Ollama than 0.20.7, so Ollama was upgraded to 0.40.11
   (Homebrew) after Mistral Small 3.2 and Llama 3.3 70B had run. The gpt-oss
   models ran on 0.40.11; all others on 0.20.7. The server log shows flash
   attention on and an 8-bit KV cache for every run on both versions. Llama
   3.1 8B was rerun in full on 0.40.11 to measure the effect of the version
   change alone (`results/runs/llama3.1-8b.rerun-v040.jsonl`).
7. Added after the second batch, all descriptive, no tests: misreports
   (candidates found by `scripts/misreport_candidates.py`, every candidate read
   and coded by hand in `results/misreports.csv`; a lower bound), conflict
   episodes with no tool call, approval conflicts completed compliantly, the
   lenient reading of calls written inside prose, cross-model counts used in
   the prose, and the cross-version comparison.
8. The runner gained a backend for OpenAI-compatible endpoints. It was
   checked against the Ollama path on six episodes of Llama 3.1 8B (identical
   transcripts and ledgers); no reported result uses it.

## Data

Every company is fictitious and no market data or proprietary data of any kind is used.
