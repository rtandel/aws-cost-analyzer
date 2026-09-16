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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import agent
from openai import OpenAI

JUDGE_MODEL = "gpt-4o-mini"
EVAL_CASES_PATH = Path(__file__).resolve().parent / "eval_cases.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")

judge_client = OpenAI()


def check_tool_selection(case: dict, actual_tools: list[str]) -> tuple[bool, list[list[str]]]:
    acceptable_sets = case.get("acceptable_tool_sets", [case["expected_tools"]])
    actual_set = set(actual_tools)
    ok = any(actual_set == set(s) for s in acceptable_sets)
    return ok, acceptable_sets


def judge_answer_quality(question: str, answer: str, reference_summary: str, notes: str) -> dict:
    prompt = f"""You are grading an AWS operations assistant's answer for a regression eval suite.

Question asked: {question}

Reference summary of what a passing answer looked like on a previously graded run:
{reference_summary}

Notes from that prior grading (may include known caveats):
{notes}

New answer to grade:
{answer}

Judge whether the new answer meets the same quality bar as the reference: reaches a
similar conclusion, is consistent with the kind of data described, and contains no
fabricated specifics (e.g. wrong dates, invented numbers). Minor wording differences
are fine and should not fail the answer.

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

    judge = judge_answer_quality(
        case["question"],
        answer,
        case.get("actual_answer_summary", ""),
        case.get("notes", ""),
    )

    tool_results_text = "\n".join(
        m["content"] for m in agent.conversation_history if isinstance(m, dict) and m.get("role") == "tool"
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
    for case in cases:
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
