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

_FORMAT_SYSTEM = """Format this GTM bot answer for better Slack readability. Instructions:
1. Add a one-line bold summary at the very top using SINGLE asterisks: *<key finding in ≤15 words>* — SKIP step 1 if the answer already starts with *bold text*
2. Prefix any line that describes zero or missing coverage (e.g. 0 DTs, no SDs in a zone, districts with no coverage, no appointment plans, coverage gaps) with ⚠️
3. If the answer contains a table (inside triple backticks) and it has no grand total row, add one at the bottom of the table summing all numeric columns
4. Keep all data values, numbers, and existing Slack formatting exactly as-is
5. Do not rewrite, reorder, summarize, or truncate the body — only add the summary header, ⚠️ markers, and grand total row where appropriate
6. IMPORTANT: always use single asterisks for bold (*text*), never double asterisks (**text**) — this is Slack mrkdwn not markdown"""


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
