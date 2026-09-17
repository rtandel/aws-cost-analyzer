# Progress — Ops Intelligence Agent

An autonomous agent that monitors AWS cost and infrastructure, decides which tools to call, and answers operational questions — a portfolio project for moving from federal-contracting infra work toward agentic AI engineering roles.

## Current State

- **Phase 0 (single tool-calling script)** — complete.
- **Phase 1 (multi-tool agent loop)** — complete. 4 tools: `get_aws_cost`, `get_cost_by_service`, `get_s3_storage_cost`, `list_ec2_instances`.
- **Phase 2 (memory + RAG)** — complete. Added `search_runbooks` (Chroma vector store, 3 incident docs, OpenAI `text-embedding-3-small` embeddings) as a 5th tool, plus persistent conversation memory via `conversation_history` and `reset_conversation()`.
- **Phase 3 (evaluation + guardrails)** — complete. Built an automated eval harness (`evals/run_evals.py`), root-caused every failure it surfaced, and got the 8-case suite to a stable 8/8. Details below.
- **Phase 4 (deploy)** — not started. ECS/Fargate planned, reusing existing AWS infra experience.

## Structure

- `tools.py` — tool functions, JSON schema (`tools`), dispatch dict (`TOOL_FUNCTIONS`), and the Chroma client/collection `search_runbooks` queries.
- `agent.py` — `SYSTEM_PROMPT`, `conversation_history`, `ask_agent()`, `reset_conversation()`. Also records `last_tool_calls` after each `ask_agent()` call so callers (the eval harness) can inspect tool selection without changing `ask_agent`'s return type.
- `app.ipynb` — the working notebook; imports from `tools.py`/`agent.py` rather than duplicating them, so the notebook and the harness share one implementation. Runs clean top-to-bottom (verified via `jupyter nbconvert --execute`).
- `evals/eval_cases.json` — source of truth for eval cases, 8 logged.
- `evals/run_evals.py` — the automated harness (see below).
- `evals/results/latest.json` — output of the most recent harness run, tracked in git so `git diff` shows regressions after any prompt/tool/model change. Timestamped per-run snapshots also land in `evals/results/` but are gitignored.

Run the eval suite with:
```bash
python evals/run_evals.py                    # all cases
python evals/run_evals.py --case cost_driver_01   # one case
python evals/run_evals.py --verbose           # print each tool call as it happens
```
Uses the `llms` conda env (`chromadb`, `openai`, `boto3`, `python-dotenv` all present there; the base `python3`/`.venv` do not have these).

## Phase 3: What the Harness Does and What It Found

The harness reruns every case in `eval_cases.json` against the live agent and grades two independent axes:
1. **Tool selection** — actual tools called vs. `acceptable_tool_sets` (falls back to exact-match on `expected_tools` if that field is absent). Several questions are legitimately answerable with more than one valid tool combination; the field exists specifically so that isn't graded as a failure.
2. **Answer quality** — an LLM-as-judge (`gpt-4o`) call, grounded in the actual raw tool results from that run (not just the historical reference text), scoring whether the answer's cited facts are real and its conclusion is reasonable given today's data.

Plus a deterministic guardrail: any `YYYY-MM-DD` date cited in the final answer must appear in the tool results actually used that turn — this operationalizes the previously-logged `s3_date_hallucination_01` bug (model cited real Cost Explorer data but transcribed the wrong year).

**First run: 1/8 passed.** Root-causing each failure (not just re-running until green) surfaced four distinct classes of issue:

1. **Real, still-open agent gap** — `last_month_spend_01`: the agent used trailing-30-days as a stand-in for "last month" without flagging the substitution. This was already known and logged in Phase 1 but never fixed. **Fix:** added a system prompt instruction to explicitly flag calendar-period approximations and name the window used.
2. **Harness gap, not an agent bug** — `runbook_retrieval_01` and `memory_followup_recall_01` both consistently failed when run via `reset_conversation()` in isolation. Root cause: both questions depend on prior conversation context ("**this** spend pattern", "how would I actually fix **that**") that only exists mid-conversation — they were originally tested as a second turn, not standalone. **Fix:** added a `setup_turns` field so a case can prime the conversation with an earlier question before the one actually being graded.
3. **Stale eval reference, not an agent bug** — `cost_driver_01`'s reference required the answer to cite "AWS Cost Explorer API usage" as the top cost driver. Pulling the raw `get_cost_by_service` output directly confirmed "AWS Cost Explorer" no longer appears as a line item in the account's current cost breakdown at all — the account's real spend profile has drifted since the original test, so that insight is now unrecoverable from live data. **Fix:** loosened the reference to the general property (correctly names today's actual top cost driver) instead of a frozen historical fact. **Lesson:** any eval graded against live AWS billing data needs periodic revalidation — the "known good" answer can go stale silently as the account's real spend changes, with no code or prompt change involved.
4. **Judge calibration issue, not an agent bug** — the first-draft judge was graded only against narrative reference text with no ground truth, so it called real tool figures (e.g. an actual `$0.0000183863` line item) "fabricated," and separately kept nitpicking rounding/precision in summed totals even when told not to. **Fix:** feed the judge the raw tool results from that run as ground truth, instruct it explicitly not to fail on rounding/precision, and switch the judge model from `gpt-4o-mini` to `gpt-4o` (mini kept ignoring the "don't nitpick precision" instruction even when given the same grounding).

Also hit and fixed an infra issue: this org's `gpt-4o` TPM limit (30k) is tight enough that a full 8-case suite run can sustain the cap rather than just spike through it, which the OpenAI SDK's default retry count (2) couldn't recover from. Fixed with `max_retries=6` on both OpenAI clients plus a 5s pace between cases in the harness loop.

**Current state: 8/8 passing, confirmed stable across two independent full-suite runs.**

## Known Open Items

- The account's real AWS spend is currently near-zero across the board (fractions of a cent daily). This makes some eval questions ("what's driving my spend") structurally low-signal — there's no real spike to reason about right now. Worth revisiting once/if there's genuine cost variation to test against, or consider synthetic cost injection for eval purposes.
- `search_runbooks` has only 3 documents; `n_results=2` will nearly always return 2 of 3 regardless of true relevance strength (noted since Phase 2). Not yet a strong test of retrieval quality at scale.
- The date-hallucination guardrail is deterministic and narrow (only catches `YYYY-MM-DD` patterns). It hasn't tripped since the original bug was found; no evidence yet on how it behaves against a genuinely new hallucination.
- No dedicated eval case yet for a completely out-of-scope question (e.g. Lambda/RDS costs) to confirm the agent says "I don't have a tool for that" rather than guessing.

## Patterns Worth Preserving

- **Tool pattern**: every tool needs three things kept in sync — a schema entry in `tools`, the actual function, and an entry in `TOOL_FUNCTIONS` (the dispatch dict, name → function). Order doesn't matter in either structure.
- **Dispatch is defensive**: unknown tool names or exceptions during a tool call return an error string as the tool result rather than crashing the loop.
- **`messages.append(msg)` happens exactly once per loop iteration**, before iterating over that turn's tool calls.
- **System prompt is the fix mechanism of choice**: four real reliability/quality gaps now (EC2 cost/instance-count hedging, memory follow-up not triggering `search_runbooks`, missing calendar-approximation caveat, and indirectly the runbook-retrieval context issue) have been root-caused through actual testing and fixed with a targeted system prompt sentence, not new code. Check whether a new bug fits this pattern before assuming you need new tools or logic.
- **`chroma_db/` is gitignored** — regenerated from the runbook source data, never commit the binary vector store itself.
- **New (Phase 3): don't trust a failing eval at face value — root-cause it before touching the agent.** Of 7 failures on the first harness run, only 1 was a real agent gap; the rest were harness/reference/judge issues that would have led to bad "fixes" (or wasted tuning) if patched reactively. Always check: (a) is this flaky across reruns or consistent? (b) does the raw tool data actually support what the reference/judge expects? (c) does the question depend on context the case isn't providing?
- **New (Phase 3): an LLM-as-judge needs the same ground truth the agent had**, not just a narrative description of a past "good" answer — otherwise it grades vibes, not facts, and produces its own false positives/negatives.

## Next Steps

With Phase 3 closed out, in rough priority order:

1. **Re-run the eval suite after any future prompt, tool, or model change** — `python evals/run_evals.py`, then `git diff -- evals/results/latest.json` to see exactly what shifted. This is the harness's whole reason for existing; use it before assuming a change is safe.
2. **Add 2-3 new eval cases** covering the gaps noted above: an out-of-scope question (Lambda/RDS), and something that would actually exercise the date-hallucination guardrail on purpose (e.g. a multi-week query spanning a year boundary) to confirm it fires when it should, not just when it doesn't.
3. **Consider whether `search_runbooks`'s `n_results=2` needs revisiting** now that it's a known weak point — either more runbook documents, or a relevance-threshold cutoff instead of a fixed top-k.
4. **Decide on the trailing-days-vs-calendar-period gap long-term**: the system prompt caveat is a real fix, but the original idea of a calendar-aware cost tool (`get_cost_for_calendar_month`) is still on the table if you want a cleaner answer than "approximating with a 30-day window."
5. **When ready, move to Phase 4 (deploy)** — see below, unchanged from the original plan.

## Phase 4 (Deploy) — Not Started

ECS/Fargate planned, reusing existing AWS infra experience. `tools.py`/`agent.py` are already plain importable modules (extracted in Phase 3 specifically to avoid a bigger rewrite here), so this phase should be able to build directly on them rather than starting from notebook code. Nothing else about this phase has been scoped yet.

## Repo Hygiene

- Commits per logical unit of work, pushed at least once per session.
- IAM user scoped to read-only permissions for Cost Explorer, EC2 describe, S3 — not admin access.
- `.gitignore` covers `chroma_db/`, `__pycache__/`, and timestamped eval result snapshots (`evals/results/run_*.json`); `evals/results/latest.json` is intentionally tracked for diffing.
