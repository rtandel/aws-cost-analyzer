"""
Eval harness for the Ops Intelligence Agent.

Reruns every case in eval_cases.json against the live agent and reports
pass/fail on two independent axes:

  - tool selection: did the agent call an acceptable set of tools?
  - answer quality: does the answer meet the same bar as the reference
    (actual_answer_summary), judged by an LLM, plus a deterministic
    guardrail that any date cited in the answer must actually appear in
    the tool results the agent used (catches the s3_date_hallucination_01
    class of bug: real data, transcribed with the wrong year).

Usage:
    python run_evals.py                  # run all cases
    python run_evals.py --case cost_driver_01   # run one case
    python run_evals.py --verbose        # print each tool call as it happens
"""

import argparse
import datetime
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from openai import OpenAI

JUDGE_MODEL = "gpt-4o"
EVAL_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

judge_client = OpenAI(max_retries=6)


def check_tool_selection(case: dict, actual_tools: list[str]) -> tuple[bool, list[list[str]]]:
    acceptable_sets = case.get("acceptable_tool_sets", [case["expected_tools"]])
    actual_set = set(actual_tools)
    ok = any(actual_set == set(s) for s in acceptable_sets)
    return ok, acceptable_sets


def judge_answer_quality(question: str, answer: str, reference_summary: str, notes: str, tool_results_text: str) -> dict:
    prompt = f"""You are grading an AWS operations assistant's answer for a regression eval suite.

Question asked: {question}

Reference summary of what a passing answer looked like on a PAST run (the account's real
AWS data changes over time, so treat this as an example of the right kind of reasoning and
tone, not a literal script the new answer must match):
{reference_summary}

Notes from that prior grading (may include known caveats, e.g. dataset-size limitations
that also apply to this run -- do not fail the answer for a caveat already noted here):
{notes}

Raw tool results actually returned during THIS run (ground truth for this run -- use this,
not the reference summary above, to check whether any figure/date/fact in the new answer is
real or fabricated):
{tool_results_text or "(no tools were called this run)"}

New answer to grade:
{answer}

Judge only:
1. Is every specific figure, date, or fact the answer cites actually present in or a fair
   summary of the raw tool results above? A number that looks unusually precise is NOT a
   fabrication if it matches the tool data -- only flag it if it does NOT appear there.
2. Does the answer reach a conclusion that is reasonable given THIS run's actual tool
   results (not necessarily the same conclusion as the old reference, if the underlying
   data has genuinely changed)?
Do not fail the answer for different wording, a different but equally valid framing, or for
omitting narrative flourishes the reference happened to include. Do not perform your own
precise arithmetic on the raw values and fail the answer over rounding or small floating-point
differences -- a qualitative or range-based claim (e.g. "very minimal", "less than $0.001",
"about $0.0002") is correct as long as it does not contradict the actual order of magnitude in
the tool data. Only fail on a genuine, unambiguous mismatch (a fact that is flatly absent from
or contradicted by the tool data), not on imprecision.

Respond with strict JSON only: {{"meets_bar": true or false, "reasoning": "one or two sentences"}}"""

    resp = judge_client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def check_date_guardrail(answer: str, tool_results_text: str) -> dict | None:
    answer_dates = set(DATE_RE.findall(answer))
    if not answer_dates:
        return None
    tool_dates = set(DATE_RE.findall(tool_results_text))
    hallucinated = sorted(answer_dates - tool_dates)
    return {"ok": not hallucinated, "hallucinated_dates": hallucinated}


def run_case(case: dict, verbose: bool = False) -> dict:
    agent.reset_conversation()

    for setup_question in case.get("setup_turns", []):
        agent.ask_agent(setup_question, verbose=verbose)

    prompt = case.get("prompt", case["question"])
    answer = agent.ask_agent(prompt, verbose=verbose)
    actual_tools = list(agent.last_tool_calls)

    tool_ok, acceptable_sets = check_tool_selection(case, actual_tools)

    tool_results_text = "\n".join(
        m["content"] for m in agent.conversation_history if isinstance(m, dict) and m.get("role") == "tool"
    )

    judge = judge_answer_quality(
        case["question"],
        answer,
        case.get("actual_answer_summary", ""),
        case.get("notes", ""),
        tool_results_text,
    )

    date_check = check_date_guardrail(answer, tool_results_text)

    overall_pass = tool_ok and judge["meets_bar"] and (date_check is None or date_check["ok"])

    return {
        "id": case["id"],
        "question": case["question"],
        "acceptable_tool_sets": acceptable_sets,
        "actual_tools_called": actual_tools,
        "tool_selection_pass": tool_ok,
        "answer": answer,
        "quality_judge": judge,
        "date_guardrail": date_check,
        "pass": overall_pass,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Ops Intelligence Agent eval cases")
    parser.add_argument("--case", help="Only run the case with this id")
    parser.add_argument("--verbose", action="store_true", help="Print each tool call as it happens")
    args = parser.parse_args()

    data = json.loads(EVAL_CASES_PATH.read_text())
    cases = data["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"No case with id '{args.case}'")
            sys.exit(1)

    results = []
    for i, case in enumerate(cases):
        if i > 0:
            # This org's gpt-4o TPM limit is tight enough that back-to-back cases
            # (each making several agent + judge calls) can sustain the cap rather
            # than just spike through it, which the SDK's own retry/backoff can't
            # recover from -- so pace our own request rate instead.
            time.sleep(5)
        print(f"Running: {case['id']} — {case['question']}")
        result = run_case(case, verbose=args.verbose)
        status = "PASS" if result["pass"] else "FAIL"
        print(f"  [{status}] tools={result['actual_tools_called']} quality_ok={result['quality_judge']['meets_bar']}")
        if not result["pass"]:
            if not result["tool_selection_pass"]:
                print(f"    tool mismatch: expected one of {result['acceptable_tool_sets']}, got {result['actual_tools_called']}")
            if not result["quality_judge"]["meets_bar"]:
                print(f"    quality: {result['quality_judge']['reasoning']}")
            if result["date_guardrail"] and not result["date_guardrail"]["ok"]:
                print(f"    date guardrail: hallucinated dates {result['date_guardrail']['hallucinated_dates']}")
        results.append(result)

    passed = sum(1 for r in results if r["pass"])
    summary = {
        "run_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "results": results,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"run_{ts}.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    latest_path = RESULTS_DIR / "latest.json"
    latest_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(f"\n{passed}/{len(results)} passed. Results written to {out_path}")
    print(f"Diff against the previous run with: git diff -- {latest_path.relative_to(Path.cwd()) if latest_path.is_relative_to(Path.cwd()) else latest_path}")


if __name__ == "__main__":
    main()
