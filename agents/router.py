"""
Router Agent — classifies user intent and dispatches to the correct specialist agent.
Add new agents here as categories grow; cross-dataset queries are handled by each agent's transfer tools.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

_CLASSIFY_SYSTEM = """You are a routing classifier for the RPG Enterprises GTM FY27 Slack bot.

Classify the user's question into exactly one category:

- district_coverage: Questions about district-level coverage — DT counts, SD counts, DSE plans, feeder plans, DT appointment plans, coverage gaps, zone/region/district hierarchy, DT codes.
- pcuv_mbo: Questions about PCUV new dealer appointments — SIS (Shop-in-Shop), CS (CEAT Shoppe), MBO (Multi-Brand Outlet), normal dealers, DL channel, appointment months, prospect names, PCUV sales potential/volume.

For questions spanning both datasets, pick the primary domain. The routed agent will pull data from the other agent via its transfer tool if needed.

Respond with ONLY one of: district_coverage, pcuv_mbo"""


def _classify(question: str, history: list | None, client: OpenAI) -> str:
    messages = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
    # Include recent text-only context for follow-up awareness
    if history:
        for m in history[-8:]:
            if isinstance(m, dict) and m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str):
                messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model="gpt-5.4-mini",
        messages=messages,
        max_tokens=10,
    )
    label = resp.choices[0].message.content.strip().lower()
    return label if label in ("district_coverage", "pcuv_mbo") else "district_coverage"


def _clean_for_pcuv(history: list | None) -> list | None:
    """Strip system prompts and tool messages — PCUV agent only needs text turns."""
    if not history:
        return None
    clean = [
        m for m in history
        if isinstance(m, dict)
        and m.get("role") in ("user", "assistant")
        and isinstance(m.get("content"), str)
    ]
    return clean or None


def _prepare_for_dc(history: list | None) -> list | None:
    """Ensure DC history starts with the DC system prompt."""
    from agents.district_coverage_agent import SYSTEM_PROMPT as _DC_SYSTEM

    if history is None:
        return None  # DC agent self-initialises with its prompt
    if history and history[0].get("role") == "system":
        # Replace whatever system prompt is there with DC's own
        if history[0].get("content") == _DC_SYSTEM:
            return history
        return [{"role": "system", "content": _DC_SYSTEM}] + list(history[1:])
    # History exists but has no system prompt — inject DC's
    return [{"role": "system", "content": _DC_SYSTEM}] + list(history)


def run_routed_agent(question: str, history: list | None = None, verbose: bool = False) -> tuple[str, list]:
    """
    Classify and route the question, return (answer, updated_history).
    Drop-in replacement for district_coverage_agent.run_agent in slack_bot.py.
    Cross-dataset queries are handled by each agent's transfer tools, not the router.
    """
    from agents.district_coverage_agent import run_agent as _dc_run
    from agents.pcuv_mbo_agent import run_agent as _pcuv_run

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    label = _classify(question, history, client)

    if label == "pcuv_mbo":
        pcuv_history = _clean_for_pcuv(history)
        answer, _ = _pcuv_run(question, history=pcuv_history, verbose=verbose)
        # Append plain exchange to shared history so follow-ups have context
        updated = list(history) if history else []
        updated.append({"role": "user", "content": question})
        updated.append({"role": "assistant", "content": answer})
        return answer, updated

    # Default: district_coverage
    dc_history = _prepare_for_dc(history)
    return _dc_run(question, history=dc_history, verbose=verbose)
