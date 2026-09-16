# Handoff Brief — Ops Intelligence Agent

Paste this as your opening message in Claude Code, in your project repo, so it has full context without re-deriving anything.

---

I'm building an autonomous agent that monitors AWS cost and infrastructure, decides which tools to call, and answers operational questions — a portfolio project for moving from federal-contracting infra work toward agentic AI engineering roles. I've been building this in a Jupyter notebook with help from another Claude session, which has full history but no direct file access to this repo. You have direct file/execution access, which is why I'm moving the remaining work here.

**Read `PROGRESS.md` in this repo first** — it's a full log of every phase, bug, and fix so far. This message is the condensed version to orient you fast.

## Current State

- **Phase 0 (single tool-calling script)** — complete.
- **Phase 1 (multi-tool agent loop)** — complete. 4 tools: `get_aws_cost`, `get_cost_by_service`, `get_s3_storage_cost`, `list_ec2_instances`.
- **Phase 2 (memory + RAG)** — complete. Added `search_runbooks` (Chroma vector store, 3 incident docs, OpenAI `text-embedding-3-small` embeddings) as a 5th tool, plus persistent conversation memory via a module-level `conversation_history` list and a `reset_conversation()` helper.
- **Phase 3 (evaluation + guardrails)** — just starting. This is the immediate work.
- **Phase 4 (deploy)** — not started. ECS/Fargate planned, reusing existing AWS infra experience.

## Immediate Next Task

Build an actual eval harness script that reruns every case in `eval_cases.json` automatically — currently I've been eyeballing outputs by hand and logging findings manually, which doesn't scale. The script should:

1. Reset conversation state before each case (`reset_conversation()`), so cases don't bleed into each other.
2. Run the question through `ask_agent`, capture which tools actually got called.
3. Compare against `expected_tools` in each case — **decide and implement**: exact match, or "any tool from an acceptable set" (tool-selection non-determinism has already been observed — same question, different valid tool across runs — so an exact-match requirement will produce false failures).
4. For answer *quality* (not just tool selection), either simple keyword/assertion checks per case, or an LLM-as-judge call scoring against `actual_answer_summary`-style criteria.
5. Output a pass/fail summary, ideally in a format that's easy to diff between runs (so regressions are visible after any prompt/tool change).

## Known Open Bug (already logged, not yet fixed)

`s3_date_hallucination_01` in `eval_cases.json` — the model cited real Cost Explorer data but reported every date as year 2023 when the actual tool data was 2026. Root cause not yet investigated. Worth a guardrail: assert any date cited in a final answer actually appears in the tool result that was used.

## Patterns Worth Preserving

- **Tool pattern**: every tool needs three things kept in sync — a schema entry in `tools`, the actual function, and an entry in `TOOL_FUNCTIONS` (the dispatch dict, name → function). Order doesn't matter in either structure.
- **Dispatch is defensive**: unknown tool names or exceptions during a tool call return an error string as the tool result rather than crashing the loop — this was a real bug earlier (an unhandled `KeyError` mid-loop caused an OpenAI 400 error about orphaned `tool_calls`).
- **`messages.append(msg)` happens exactly once per loop iteration**, before iterating over that turn's tool calls — duplicating it inside the per-tool-call loop was an earlier bug.
- **System prompt is the fix mechanism of choice**: two real reliability bugs so far (EC2 cost/instance-count hedging, and a memory follow-up not triggering `search_runbooks`) were both root-caused through actual testing and fixed with one added sentence to the system prompt, not code changes. Check whether a new bug fits this pattern before assuming you need new tools or logic.
- **`chroma_db/` is gitignored** — regenerated from the runbook source data, never commit the binary vector store itself.

## Suggested First Structural Step

Before or alongside the eval harness, consider extracting the notebook's reusable code into plain `.py` modules (e.g. `tools.py` for the tool functions/schema/dispatch dict, `agent.py` for `ask_agent`/`conversation_history`/`reset_conversation`). The notebook currently holds everything; Phase 4 deployment (ECS/Fargate) needs this to run as a script, not a notebook, so doing the extraction now avoids a bigger rewrite later. Not mandatory before starting the eval harness, but worth doing before Phase 4.

## Repo Hygiene Already in Place

- Commits per logical unit of work, pushed at least once per session.
- IAM user scoped to read-only permissions for Cost Explorer, EC2 describe, S3 — not admin access.
- `eval_cases.json` at repo root (or wherever you've placed it) is the source of truth for eval cases — 7 logged so far.