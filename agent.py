import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from tools import tools, TOOL_FUNCTIONS

load_dotenv(override=True)

MODEL = "gpt-4o"
openai = OpenAI()

SYSTEM_PROMPT = (
    "You are an AWS operations assistant. When interpreting get_cost_by_service results, "
    "a service that does not appear in the breakdown had zero cost for that period -- "
    "treat this as a confident zero, not as missing or anomalous data. When you have "
    "multiple tool results, state your conclusion plainly when the data supports it, "
    "rather than speculating about billing or configuration issues. When asked how to fix, "
    "resolve, or address a cost or infrastructure issue, always check search_runbooks for "
    "a documented resolution first -- prefer a specific, already-known fix over generic "
    "best-practice advice."
)

conversation_history = [{"role": "system", "content": SYSTEM_PROMPT}]

# Tool names called during the most recent ask_agent() invocation, in call order.
# Reset at the start of each call so callers (e.g. the eval harness) can inspect
# it right after ask_agent() returns without changing ask_agent's return type.
last_tool_calls = []


def reset_conversation():
    conversation_history.clear()
    conversation_history.append({"role": "system", "content": SYSTEM_PROMPT})


def ask_agent(question: str, verbose: bool = True) -> str:
    global last_tool_calls
    last_tool_calls = []

    conversation_history.append({"role": "user", "content": question})

    while True:
        resp = openai.chat.completions.create(model=MODEL, messages=conversation_history, tools=tools)
        msg = resp.choices[0].message
        conversation_history.append(msg)

        if not msg.tool_calls:
            return msg.content

        for call in msg.tool_calls:
            name = call.function.name
            args = json.loads(call.function.arguments)
            last_tool_calls.append(name)
            if verbose:
                print(f"  → calling {name}({args})")
            if name not in TOOL_FUNCTIONS:
                result = f"Error: model requested unknown tool '{name}'"
            else:
                try:
                    result = TOOL_FUNCTIONS[name](**args)
                except Exception as e:
                    result = f"Error running '{name}': {e}"
            conversation_history.append({"role": "tool", "tool_call_id": call.id, "content": result})
