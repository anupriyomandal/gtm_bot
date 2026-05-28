"""
District Coverage Agent
Answers questions from data/district_coverage/district_coverage.parquet
using OpenAI GPT-4.1 with tool use.
"""

import json
import os
import traceback
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
import pandas as pd

load_dotenv(Path(__file__).parent / ".env")

from agents.pcuv_mbo_agent import run_agent as _pcuv_run_agent  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data" / "district_coverage" / "district_coverage.parquet"

_df: pd.DataFrame | None = None


def _get_df() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_parquet(DATA_PATH)
    return _df


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def load_data() -> dict:
    df = _get_df()
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "zones": sorted(df["Zone Description"].dropna().unique().tolist()),
        "regions": sorted(df["Region Description"].dropna().unique().tolist()),
    }


def get_summary() -> dict:
    df = _get_df()
    dt_plan_yes = df["DT Appointment Plan (Y/N)"].str.strip().str.upper().isin(["Y", "YES", "1"]).sum()
    feeder_plan = int((df["Feeder Appointment Plan"] == 1).sum())
    dse_plan = df["No of DSEs Addition Plan (1,2,3 etc)"].fillna(0) > 0
    metro_yes = df["Metro DT Appointment Plan (Y/N)"].str.strip().str.upper().isin(["Y", "YES"]).sum()
    super_yes = df["Super DT Appointment Plan (Y/N)"].str.strip().str.upper().isin(["Y", "YES"]).sum()
    return {
        "total_districts": len(df),
        "total_zones": df["Zone Description"].nunique(),
        "total_regions": df["Region Description"].nunique(),
        "districts_with_DT_appointment_plan": int(dt_plan_yes),
        "districts_with_feeder_plan": feeder_plan,
        "districts_with_DSE_addition_plan": int(dse_plan.sum()),
        "districts_with_metro_DT_plan": int(metro_yes),
        "districts_with_super_DT_plan": int(super_yes),
        "avg_SDs_FY26": round(df["Current No of SDs FY26"].mean(), 1),
        "avg_planned_SDs_FY27": round(df["Planned No of SDs FY27"].mean(), 1),
    }


def filter_by_geography(zone: str = None, region: str = None, district_name: str = None) -> dict:
    df = _get_df().copy()
    if zone:
        df = df[df["Zone Description"].str.contains(zone, case=False, na=False)]
    if region:
        df = df[df["Region Description"].str.contains(region, case=False, na=False)]
    if district_name:
        df = df[df["District Name"].str.contains(district_name, case=False, na=False)]

    cols = [
        "Zone Description", "Region Description", "District Name",
        "Current Status(No of DTs)", "Current No of SDs FY26",
        "Planned No of SDs FY27", "Current No of DSEs",
        "DT Appointment Plan (Y/N)", "Metro DT Appointment Plan (Y/N)",
        "Super DT Appointment Plan (Y/N)",
    ]
    records = df[cols].where(df[cols].notna(), None).to_dict(orient="records")
    return {"matched": len(records), "data": records}


def get_dt_appointment_plan(zone: str = None, region: str = None, month: str = None) -> dict:
    df = _get_df().copy()
    mask = df["DT Appointment Plan (Y/N)"].str.strip().str.upper().isin(["Y", "YES", "1"])
    df = df[mask]
    if zone:
        df = df[df["Zone Description"].str.contains(zone, case=False, na=False)]
    if region:
        df = df[df["Region Description"].str.contains(region, case=False, na=False)]
    if month:
        df = df[df["Month of DT Appoinment Plan"].str.contains(month, case=False, na=False)]

    cols = [
        "Zone Description", "Region Description", "District Name",
        "DT Appointment Plan (Y/N)", "No of DTs Appointment Plan (1,2,3 etc)",
        "Month of DT Appoinment Plan", "Metro DT Appointment Plan (Y/N)",
        "Super DT Appointment Plan (Y/N)", "DT Code",
    ]
    records = df[cols].where(df[cols].notna(), None).to_dict(orient="records")
    return {"matched": len(records), "data": records}


def get_feeder_plan(zone: str = None, region: str = None) -> dict:
    df = _get_df().copy()
    df = df[df["Feeder Appointment Plan"] == 1]
    if zone and zone.strip():
        df = df[df["Zone Description"].str.contains(zone.strip(), case=False, na=False)]
    if region and region.strip():
        df = df[df["Region Description"].str.contains(region.strip(), case=False, na=False)]

    zone_summary = (
        df.groupby("Zone Description")["No of FDs Appointment Plan (1,2,3 etc)"]
        .agg(districts_with_plan="count", total_FDs_planned="sum")
        .reset_index()
        .to_dict(orient="records")
    )
    region_summary = (
        df.groupby(["Zone Description", "Region Description"])["No of FDs Appointment Plan (1,2,3 etc)"]
        .agg(districts_with_plan="count", total_FDs_planned="sum")
        .reset_index()
        .to_dict(orient="records")
    )

    cols = [
        "Zone Description", "Region Description", "District Name",
        "Feeder Appointment Plan", "No of FDs Appointment Plan (1,2,3 etc)",
        "Month of Appoinment Plan", "Current Count of Feeders",
    ]
    records = df[cols].where(df[cols].notna(), None).to_dict(orient="records")
    return {"matched": len(records), "zone_summary": zone_summary, "region_summary": region_summary, "data": records}


def get_dse_plan(zone: str = None, region: str = None, dse_count: int = None) -> dict:
    df = _get_df().copy()
    df["_dse_count"] = df["No of DSEs Addition Plan (1,2,3 etc)"].fillna(0)
    df = df[df["_dse_count"] > 0]
    if zone and zone.strip():
        df = df[df["Zone Description"].str.contains(zone.strip(), case=False, na=False)]
    if region and region.strip():
        df = df[df["Region Description"].str.contains(region.strip(), case=False, na=False)]
    if dse_count is not None:
        df = df[df["_dse_count"] == dse_count]

    zone_summary = (
        df.groupby("Zone Description")["_dse_count"]
        .agg(districts_with_plan="count", total_dses_planned="sum")
        .reset_index()
        .to_dict(orient="records")
    )
    region_summary = (
        df.groupby(["Zone Description", "Region Description"])["_dse_count"]
        .agg(districts_with_plan="count", total_dses_planned="sum")
        .reset_index()
        .to_dict(orient="records")
    )

    cols = [
        "Zone Description", "Region Description", "District Name",
        "Current No of DSEs", "No of DSEs Addition Plan (1,2,3 etc)",
        "Month of DSE Appoinment Plan",
    ]
    records = df[cols].where(df[cols].notna(), None).to_dict(orient="records")
    return {"matched": len(records), "zone_summary": zone_summary, "region_summary": region_summary, "data": records}


def aggregate_by_zone_or_region(group_by: str = "zone") -> dict:
    df = _get_df().copy()
    col = "Zone Description" if group_by.lower() == "zone" else "Region Description"

    agg = df.groupby(col).agg(
        total_districts=(col, "count"),
        total_DTs=("Current Status(No of DTs)", "sum"),
        avg_DTs=("Current Status(No of DTs)", "mean"),
        total_SDs_FY26=("Current No of SDs FY26", "sum"),
        total_planned_SDs_FY27=("Planned No of SDs FY27", "sum"),
        total_feeders=("Current Count of Feeders", "sum"),
        districts_with_feeder_plan=("Feeder Appointment Plan", "sum"),
        total_DSEs_current=("Current No of DSEs", "sum"),
        total_DSEs_addition_planned=("No of DSEs Addition Plan (1,2,3 etc)", "sum"),
        total_DTs_addition_planned=("No of DTs Appointment Plan (1,2,3 etc)", "sum"),
        total_FDs_addition_planned=("No of FDs Appointment Plan (1,2,3 etc)", "sum"),
    ).round(1).reset_index()

    dt_yes = df[df["DT Appointment Plan (Y/N)"] == "Y"]
    dt_counts = dt_yes.groupby(col).size().reset_index(name="districts_with_DT_plan")
    agg = agg.merge(dt_counts, on=col, how="left").fillna(0)

    return {"group_by": col, "data": agg.to_dict(orient="records")}


def search_district(query: str) -> dict:
    df = _get_df().copy()
    q = query.strip().lower()
    mask = (
        df["District Name"].str.lower().str.contains(q, na=False) |
        df["DT Code"].str.lower().str.contains(q, na=False) |
        df["District"].str.lower().str.contains(q, na=False)
    )
    results = df[mask].where(df[mask].notna(), None).to_dict(orient="records")
    return {"matched": len(results), "data": results}


def get_coverage_gaps(zone: str = None, region: str = None) -> dict:
    df = _get_df().copy()
    if zone:
        df = df[df["Zone Description"].str.contains(zone, case=False, na=False)]
    if region:
        df = df[df["Region Description"].str.contains(region, case=False, na=False)]

    no_dts = df[df["Current Status(No of DTs)"].isna() | (df["Current Status(No of DTs)"] == 0)]
    no_sds = df[df["Current No of SDs FY26"].isna() | (df["Current No of SDs FY26"] == 0)]
    no_dses = df[df["Current No of DSEs"].fillna(0) == 0]
    no_plan = df[
        ~df["DT Appointment Plan (Y/N)"].str.strip().str.upper().isin(["Y", "YES", "1"]) &
        (df["Feeder Appointment Plan"] == 0)
    ]

    def to_records(sub):
        cols = ["Zone Description", "Region Description", "District Name",
                "Current Status(No of DTs)", "Current No of SDs FY26", "Current No of DSEs"]
        return sub[cols].where(sub[cols].notna(), None).to_dict(orient="records")

    return {
        "districts_with_no_DTs": {"count": len(no_dts), "data": to_records(no_dts)},
        "districts_with_no_SDs_FY26": {"count": len(no_sds), "data": to_records(no_sds)},
        "districts_with_no_DSEs": {"count": len(no_dses), "data": to_records(no_dses)},
        "districts_with_no_appointment_plan": {"count": len(no_plan), "data": to_records(no_plan)},
    }


def run_pandas(code: str) -> dict:
    """
    Execute arbitrary pandas code against the loaded dataframe.
    The variable `df` is pre-bound to the full district coverage dataframe.
    The code must assign its result to a variable named `result`.

    Example:
        df[df['Zone Description'] == 'East Zone'].groupby('Region Description')['Current No of SDs FY26'].sum()
        result = ...
    """
    df = _get_df().copy()  # noqa: F841 — exposed to exec scope
    local_vars: dict = {"df": df, "pd": pd}
    try:
        exec(code, {"pd": pd, "__builtins__": {}}, local_vars)  # restricted builtins
    except Exception:
        return {"error": traceback.format_exc(), "code": code}

    result = local_vars.get("result", None)
    if result is None:
        return {"error": "Code did not assign anything to `result`.", "code": code}

    if isinstance(result, pd.DataFrame):
        return {"type": "dataframe", "shape": list(result.shape), "data": result.where(result.notna(), None).to_dict(orient="records")}
    if isinstance(result, pd.Series):
        return {"type": "series", "data": result.where(result.notna(), None).to_dict()}
    return {"type": type(result).__name__, "data": result}


# ---------------------------------------------------------------------------
# Tool schemas for Claude
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "think",
        "description": (
            "Use this tool to reason step-by-step before calling any data tool. "
            "Think about: what is the user actually asking, what filters apply (zone/region from prior context), "
            "which tool is most appropriate, and what parameters to pass. "
            "This reasoning is not shown to the user — it just helps you plan the next action correctly."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "reasoning": {"type": "string", "description": "Your step-by-step reasoning before taking action."},
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "load_data",
        "description": "Load the district coverage dataset and return its schema — column names, zones, and regions. Call this first to orient yourself.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_summary",
        "description": "Return high-level summary statistics for the entire dataset: total districts, zones, regions, and counts of districts with DT / feeder / DSE appointment plans.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "filter_by_geography",
        "description": "Filter districts by zone, region, and/or district name (all optional, case-insensitive substring match). Returns key metrics per matching district.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string", "description": "Zone name substring, e.g. 'East Zone'"},
                "region": {"type": "string", "description": "Region name substring, e.g. 'Asansol'"},
                "district_name": {"type": "string", "description": "District name substring, e.g. 'Barddhaman'"},
            },
            "required": [],
        },
    },
    {
        "name": "get_dt_appointment_plan",
        "description": "Return districts that have a DT Appointment Plan, with number of DTs planned, month, Metro DT and Super DT flags. Filterable by zone, region, or month.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string", "description": "Zone name substring"},
                "region": {"type": "string", "description": "Region name substring"},
                "month": {"type": "string", "description": "Month substring, e.g. \"Apr'26\""},
            },
            "required": [],
        },
    },
    {
        "name": "get_feeder_plan",
        "description": "Return districts with a Feeder Appointment Plan. Returns `zone_summary` (zone-level totals: districts_with_plan, total_FDs_planned), `region_summary`, and `data` (district rows). Filterable by zone or region.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string"},
                "region": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "get_dse_plan",
        "description": "Return districts with a DSE addition plan. Returns `zone_summary`, `region_summary` (pre-aggregated totals), and `data` (district rows). Filter by zone, region, or exact dse_count (e.g. dse_count=1 to find districts adding exactly 1 DSE). Always pass zone/region as non-empty strings when filtering.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string", "description": "Zone name substring, e.g. 'East Zone'. Omit or leave null if not filtering by zone."},
                "region": {"type": "string", "description": "Region name substring. Omit or leave null if not filtering by region."},
                "dse_count": {"type": "integer", "description": "Filter for districts planning to add exactly this many DSEs, e.g. 1."},
            },
            "required": [],
        },
    },
    {
        "name": "aggregate_by_zone_or_region",
        "description": "Aggregate key metrics grouped by zone or region: total_districts, total_DTs, total_SDs_FY26, total_planned_SDs_FY27, total_feeders, total_DSEs_current, total_DSEs_addition_planned, total_DTs_addition_planned, districts_with_DT_plan, districts_with_feeder_plan. Use this for any zone/region comparison or ranking question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "group_by": {
                    "type": "string",
                    "enum": ["zone", "region"],
                    "description": "Group results by 'zone' or 'region'. Default: 'zone'.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_district",
        "description": "Free-text search on District Name or DT Code. Returns all matching rows with full details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search string, e.g. 'Pune' or '50004191'"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_coverage_gaps",
        "description": "Identify coverage gaps: districts with no DTs, no SDs in FY26, no DSEs, or no appointment plans. Filterable by zone or region.",
        "input_schema": {
            "type": "object",
            "properties": {
                "zone": {"type": "string"},
                "region": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "run_pandas",
        "description": (
            "Execute arbitrary pandas code against the district coverage dataframe (variable `df`). "
            "Use for complex queries not covered by other tools. "
            "The code MUST assign the final result to a variable named `result`. "
            "Example: result = df.groupby('Zone Description')['Current No of SDs FY26'].sum()"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python/pandas code. Must assign output to `result`. Only `df` and `pd` are available.",
                },
            },
            "required": ["code"],
        },
    },
    {
        "name": "transfer_to_pcuv_mbo_agent",
        "description": (
            "ALWAYS use this tool — without exception — when the user asks anything about: "
            "SIS (Shop-in-Shop), CS (CEAT Shoppe), MBO (Multi-Brand Outlet), normal dealer appointments, "
            "PCUV new dealer / channel partner appointments, new dealer appointment months, "
            "PCUV sales plan for new partners, or Passenger Car & UV tyre new channel partner planning. "
            "Do NOT try to answer these from the district coverage dataset — that data does not exist there. "
            "Forward the question verbatim to this agent immediately."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The user's question to forward to the PCUV MBO agent, verbatim.",
                },
            },
            "required": ["question"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_pcuv_history: dict[str, list] = {}  # keyed by channel to preserve PCUV context per conversation


def _transfer_to_pcuv(question: str) -> dict:
    answer, _ = _pcuv_run_agent(question, history=None, verbose=False)
    return {"answer": answer}


TOOL_FN_MAP = {
    "think": lambda args: {"ok": True},
    "load_data": lambda args: load_data(),
    "get_summary": lambda args: get_summary(),
    "filter_by_geography": lambda args: filter_by_geography(**args),
    "get_dt_appointment_plan": lambda args: get_dt_appointment_plan(**args),
    "get_feeder_plan": lambda args: get_feeder_plan(**args),
    "get_dse_plan": lambda args: get_dse_plan(**args),
    "aggregate_by_zone_or_region": lambda args: aggregate_by_zone_or_region(**args),
    "search_district": lambda args: search_district(**args),
    "get_coverage_gaps": lambda args: get_coverage_gaps(**args),
    "run_pandas": lambda args: run_pandas(**args),
    "transfer_to_pcuv_mbo_agent": lambda args: _transfer_to_pcuv(**args),
}


def dispatch_tool(name: str, args: dict) -> str:
    fn = TOOL_FN_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        result = fn(args)
        return json.dumps(result, default=str)
    except Exception:
        return json.dumps({"error": traceback.format_exc()})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the District Coverage Agent for RPG Enterprises GTM FY27.
You have access to the district coverage dataset (618 districts across 7 zones and multiple regions).

The dataset contains:
- Zone / Region / District hierarchy
- Current DT count, SDs FY26, feeders, DSEs per district
- DT Appointment Plans (Y/N, count, month, Metro/Super flags)
- Feeder and DSE addition plans with months
- DT Codes

## How to reason and act (ReAct pattern)
Before calling any data tool, ALWAYS call `think` first to:
1. Identify what the user is asking — including implied context from prior messages (e.g. if they said "East Zone" earlier, carry that forward).
2. Decide which tool to call and what exact parameters to pass.
3. Never pass empty strings as filter values — omit the parameter entirely if not filtering.

Then call the appropriate data tool. After observing the result, reason again if needed before answering.

## Guidelines
- Use `run_pandas` only for queries no other tool can handle.
- NEVER answer questions about SIS / CS / MBO / PCUV dealer appointments yourself — ALWAYS call `transfer_to_pcuv_mbo_agent` immediately for these.
- Be concise and specific. Format numbers clearly.

## Formatting (Slack mrkdwn)
- Use *bold* for headers and key labels (NOT **double asterisk**)
- Use - for bullet lists
- Use `backticks` for codes or numbers to highlight
- NEVER use markdown pipe tables (| col | col |) — Slack does not render them
- ALWAYS wrap tabular data in triple backticks so columns align in monospace, like this:
  ```
  Name                | Col1  | Col2
  --------------------|-------|------
  Row 1               | 100   | 200
  TOTAL               | 100   | 200
  ```
- Never use ## headers — use *bold* labels instead
- Never use **double asterisk bold** inside code blocks — plain text only inside backticks
- ALWAYS add a TOTAL row at the bottom of every table"""

# Convert Anthropic-style tool schemas → OpenAI function format
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


def run_agent(user_query: str, history: list = None, verbose: bool = True) -> tuple[str, list]:
    """
    Run one turn of the agent.
    Pass `history` from the previous turn to maintain conversation context.
    Returns (answer, updated_history) — store and pass back on the next call.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    if history is None:
        history = [{"role": "system", "content": SYSTEM_PROMPT}]

    history.append({"role": "user", "content": user_query})

    while True:
        response = client.chat.completions.create(
            model="gpt-4.1",
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
                        print(f"  [think] {args.get('reasoning', '')[:200]}")
                    elif tc.function.name == "transfer_to_pcuv_mbo_agent":
                        print(f"  [handoff->pcuv_mbo_agent] {args.get('question', '')[:120]}")
                    else:
                        print(f"  [tool] {tc.function.name}({json.dumps(args, default=str)[:120]})")
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
    print("District Coverage Agent — FY27")
    print("Type your question (or 'exit' to quit)\n")

    history = None  # persists across the entire session
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
