"""
PCUV MBO Appointment Agent
Answers questions about PCUV new dealer appointment planning from
data/pcuv_mbo/pcuv_mbo_planning.parquet using GPT-4.1-mini with ReAct tool use.

Tools: think, run_pandas
"""

import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "pcuv_mbo" / "pcuv_mbo_planning.parquet"

_df: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_parquet(DATA_PATH)
    return _df


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _transfer_to_dc(question: str) -> dict:
    from agents.district_coverage_agent import run_agent as _dc_run
    answer, _ = _dc_run(question, history=None, verbose=False)
    return {"answer": answer}


def run_pandas(code: str) -> dict:
    """
    Execute pandas code against the PCUV MBO planning dataframe.
    Variable `df` is pre-bound. Result must be assigned to `result`.

    Key columns:
    - Zone Description, Region Description, State, Territory, District, Town
    - Metro - Yes/No  (Yes / No)
    - Special Channel - CS/SIS/MBO/Other  (CS, SIS, MBO, Others, DL)
    - Prospect Code, Customer Name
    - Month of Appointment for New Channel Partner  (e.g. Apr-26, May-26)
    - FY26 DAP Potential, FY26 AVG (PCUV)
    - Apr-26 … Mar-27  (monthly planned sales)
    - FY27 Q1 Avg, FY27 Q2 Avg, FY27 Q3 Avg, FY27 Q4 Avg
    - FY27 H1 Avg, FY27 H2 Avg, FY27 Avg
    """
    df = _get_df().copy()  # noqa: F841
    local_vars: dict = {"df": df, "pd": pd}
    try:
        exec(code, {"pd": pd, "__builtins__": {}}, local_vars)
    except Exception:
        return {"error": traceback.format_exc(), "code": code}

    result = local_vars.get("result")
    if result is None:
        return {"error": "Code did not assign anything to `result`.", "code": code}

    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "shape": list(result.shape), "data": result.where(result.notna(), None).to_dict(orient="records")}
    if isinstance(result, pd.Series):
        return {"type": "series", "data": result.where(result.notna(), None).to_dict()}
    return {"type": type(result).__name__, "data": result}


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "transfer_to_district_coverage_agent",
        "description": (
            "Use this tool when the user asks about district coverage data — DT counts, SD counts, "
            "DSE plans, feeder plans, coverage gaps, or DT appointment plans by zone/region/district. "
            "That data does not exist in this dataset. Forward the question verbatim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question to forward to the District Coverage agent, verbatim.",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "think",
        "description": (
            "Reason step-by-step before writing any pandas code. "
            "Think about: what the user is asking, which columns to use, "
            "what filters and aggregations are needed, and what to assign to `result`. "
            "Always call this before run_pandas."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Your step-by-step reasoning."},
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "run_pandas",
        "description": (
            "Execute pandas code against the PCUV MBO planning dataframe (`df`, 598 rows). "
            "Must assign the final result to `result`. "
            "Only `df` and `pd` are available. "
            "Channel types: CS (CEAT Shoppe), SIS (Shop-in-Shop), MBO (Multi-Brand Outlet), Others (Normal Dealer), DL. "
            "Appointment months are in Mon-YY format (e.g. 'Apr-26'). "
            "Example: result = df.groupby('Zone Description')['FY27 Avg'].sum()"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python/pandas code. Must assign output to `result`.",
                },
            },
            "required": ["code"],
        },
    },
]

_OAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    }
    for t in TOOLS
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

TOOL_FN_MAP = {
    "transfer_to_district_coverage_agent": lambda args: _transfer_to_dc(**args),
    "think": lambda args: {"ok": True},
    "run_pandas": lambda args: run_pandas(**args),
}


def dispatch_tool(name: str, args: dict) -> str:
    fn = TOOL_FN_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return json.dumps(fn(args), default=str)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the PCUV MBO Appointment Agent for RPG Enterprises GTM FY27.
You answer questions about the PCUV (Passenger Car & UV Radial Tyres) new dealer appointment plan.

The dataset has 598 planned new channel partners across 7 zones with these key attributes:
- Geography: Zone → Region → State → District → Town
- Channel type (Special Channel): CS (CEAT Shoppe), SIS (Shop-in-Shop), MBO (Multi-Brand Outlet), Others (Normal Dealer), DL
- Metro / Non-Metro classification
- Month of appointment (Mon-YY format, e.g. Apr-26)
- Monthly planned sales (Apr-26 to Mar-27) and period averages (Q1–Q4, H1, H2, FY27 Avg)
- FY26 DAP Potential (benchmark for target-setting)

## How to reason and act (ReAct pattern)
Always call `think` first to plan your pandas code, then call `run_pandas`.
After observing the result, reason again if needed before answering.
Never guess column values — use exact values from the descriptions above.

## Critical rules
- When the user says "X RO" or "X Regional Office", treat it as region = "X" — the suffix "RO" is not stored in the data. E.g. "Ernakulam RO" → filter Region Description = "Ernakulam".
- If the user asks about district coverage data (DT counts, SD counts, DSE/feeder plans, coverage gaps), call `transfer_to_district_coverage_agent` immediately — that data does not exist in this dataset.
- ALWAYS include `Customer Name` in results whenever showing dealer-level data
- NEVER rename, relabel, or paraphrase values from the data — use exact strings as they appear in the dataframe (e.g. zone names are exactly: East Zone, North Zone 1, North Zone 2, South Zone 1, South Zone 2, West Zone 1, West Zone 2)
- Always show values exactly as returned by run_pandas
- "Appointment plan" / "how many appointments" / "how many dealers appointed" = COUNT OF ROWS by filtering on `Month of Appointment for New Channel Partner`. NEVER use monthly sales columns for this.
- Monthly sales columns (Apr-26 … Mar-27) = planned SALES VOLUME only. Use them ONLY when the user explicitly asks for sales, volume, or potential — never for counting dealers.
- Some dealers have decimal sales in their first month — counting via sales column will miss them. Always count via the appointment month column.
- Correct zone-wise appointment count pattern (e.g. MBO in May-26):
  result = df[(df['Month of Appointment for New Channel Partner'] == 'May-26') & (df['Special Channel - CS/SIS/MBO/Other'] == 'MBO')].groupby('Zone Description').size().reset_index(name='count')
- Correct all-channel zone-wise count pattern:
  result = df[df['Month of Appointment for New Channel Partner'] == 'May-26'].groupby('Zone Description').size().reset_index(name='count')

## Formatting (Slack mrkdwn)
- Use *bold* for headers and key labels
- Use - for bullet lists
- NEVER use markdown pipe tables (| col | col |) — Slack does not render them
- For tabular output, generate an aligned string using pandas and wrap in triple backticks:
  result = df[cols].to_string(index=False)
  Then in your answer: wrap that string in triple backticks. pandas to_string() handles column widths automatically — never hand-pad columns yourself.
- Append a TOTAL row as a plain-text line after the table (outside the pandas output), e.g.:
  TOTAL    | 3700.0 | 418.75
- Never use ## headers or **double asterisk bold** inside code blocks — plain text only inside backticks"""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run_agent(user_query: str, history: list = None, verbose: bool = True) -> tuple[str, list]:
    """
    Run one turn of the PCUV MBO agent.
    Pass `history` from the previous turn to maintain conversation context.
    Returns (answer, updated_history).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if history is None:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    history.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            tools=_OAI_TOOLS,
            tool_choice="auto",
            messages=history,
        )

        msg = response.choices[0].message
        history.append(msg)

        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                if verbose:
                    if tc.function.name == "think":
                        print(f"  [pcuv:think] {args.get('reasoning', '')[:200]}")
                    elif tc.function.name == "transfer_to_district_coverage_agent":
                        print(f"  [pcuv:handoff->dc] {args.get('question', '')[:120]}")
                    else:
                        print(f"  [pcuv:tool] {tc.function.name}({json.dumps(args, default=str)[:120]})")
                result_str = dispatch_tool(tc.function.name, args)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
        else:
            return msg.content or "", history


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("PCUV MBO Appointment Agent — FY27")
    print("Type your question (or 'exit' to quit)\n")

    history = None
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "bye"}:
            print("Bye.")
            break
        print("Agent: ", end="", flush=True)
        answer, history = run_agent(query, history=history, verbose=True)
        print(answer)
        print()
