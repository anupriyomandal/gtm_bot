"""
Slack output formatter — adds a bold summary line and ⚠️ gap highlights.
Applied as a lightweight post-processing step after the agent produces its answer.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).parent / ".env")

_FORMAT_SYSTEM = """Format this GTM bot answer for Slack. Instructions:
1. Add a one-line bold summary at the very top using SINGLE asterisks: *<key finding in ≤15 words>* — SKIP step 1 if the answer already starts with *bold text*
2. Prefix any line that describes zero or missing coverage (0 DTs, no SDs, no appointment plans, zero FY27 plan) with ⚠️
3. If the answer contains a markdown pipe table (lines starting with |), convert it into a triple-backtick code block with plain-text aligned columns — Slack does not render markdown tables. Remove any **bold** markers inside the table, use plain text only.
4. If a table (in triple backticks) has no TOTAL row, add one summing all numeric columns.
5. Keep all data values and numbers exactly as-is.
6. ALWAYS use single asterisks for bold (*text*), never double asterisks (**text**).
7. Do not rewrite, reorder, or truncate the body."""


def format_for_slack(answer: str) -> str:
    """Post-process agent answer: add summary + gap highlights."""
    if len(answer) < 150:
        return answer  # short answers don't need formatting

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": _FORMAT_SYSTEM},
            {"role": "user", "content": answer},
        ],
    )
    return resp.choices[0].message.content.strip()
