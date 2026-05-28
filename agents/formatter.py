"""
Slack output formatter — adds summary, gap highlights, and converts tables to Block Kit.
Uses structured output (Pydantic + .parse) to extract table data reliably.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")

# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

class _TableData(BaseModel):
    headers: list[str]
    rows: list[list[str]]  # all cell values as strings, including TOTAL row


class _Parsed(BaseModel):
    summary: str           # one-line key finding; empty string if answer already starts with *bold*
    text: str              # formatted answer; table replaced with literal "[TABLE]" placeholder
    table: Optional[_TableData] = None  # None if no tabular data present


_SYSTEM = """You process GTM sales bot answers for Slack display. Return JSON with these fields:

summary: One-line key finding (≤15 words). Return "" if the answer already starts with *bold text*.

text: The full answer with these changes applied:
  - Prefix any line describing zero or missing data (0 DTs, no SDs, no appointments, zero plan, NaN) with ⚠️
  - Use single asterisks *bold* only — never **double asterisks**
  - CRITICAL: If a table exists anywhere (code block with ```, pipe-separated rows, or space-aligned columns), you MUST remove the entire table from this field and replace it with the single literal string [TABLE] — nothing else, just [TABLE]. The table will be rendered separately. Do NOT leave any table content in text.
  - Keep all non-table content exactly as-is

table: If a table exists, extract {"headers": [...], "rows": [[...], ...]} with every cell value as a string.
  CRITICAL: You MUST include the TOTAL or GRAND TOTAL row as the last entry in rows if it appears in the original answer. Do not skip it.
  Return null if no table."""


# ---------------------------------------------------------------------------
# Block Kit builders
# ---------------------------------------------------------------------------

def _clean_num(val: str) -> str:
    """Convert '44.0' → '44'; leave non-numeric strings unchanged."""
    try:
        f = float(val)
        if f == int(f):
            return str(int(f))
    except (ValueError, OverflowError):
        pass
    return val


def _section(text: str) -> dict:
    return {"type": "section", "text": {"type": "mrkdwn", "text": text}}


def _table_block(table: _TableData) -> dict:
    header_row = [{"type": "raw_text", "text": h} for h in table.headers]
    data_rows = [
        [{"type": "raw_text", "text": _clean_num(str(cell))} for cell in row]
        for row in table.rows
    ]
    return {"type": "table", "rows": [header_row] + data_rows}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def format_for_slack(answer: str) -> tuple[str, list]:
    """
    Post-process agent answer for Slack.
    Returns (fallback_text, blocks).
    - fallback_text: plain mrkdwn string (used for push notifications and non-block clients)
    - blocks: list of Slack Block Kit block dicts; empty list if no table present
    """
    if len(answer) < 150:
        return answer, []

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    result = client.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": answer},
        ],
        response_format=_Parsed,
    )

    parsed = result.choices[0].message.parsed
    if parsed is None:
        return answer, []

    # Force single-asterisk Slack bold — GPT often ignores the instruction
    parsed.text = re.sub(r"\*\*(.+?)\*\*", r"*\1*", parsed.text)
    parsed.summary = re.sub(r"\*\*(.+?)\*\*", r"*\1*", parsed.summary)

    # Strip trailing .0 from whole-number floats in prose text (e.g. "32.0 DTs" → "32 DTs")
    parsed.text = re.sub(r"\b(\d+)\.0\b", r"\1", parsed.text)
    parsed.summary = re.sub(r"\b(\d+)\.0\b", r"\1", parsed.summary)

    # Safety net: if GPT extracted a table but forgot to put [TABLE] in text,
    # strip code blocks from text programmatically rather than showing both.
    if parsed.table and "[TABLE]" not in parsed.text:
        parsed.text = re.sub(r"```[\s\S]*?```", "[TABLE]", parsed.text)
        if "[TABLE]" not in parsed.text:
            parsed.text = parsed.text  # leave as-is; table block appends below

    prefix = f"*{parsed.summary}*\n\n" if parsed.summary else ""

    # --- Build fallback plain text ---
    if parsed.table:
        table_plaintext = "\n".join(
            " | ".join(_clean_num(cell) for cell in row)
            for row in [parsed.table.headers] + parsed.table.rows
        )
        fallback = prefix + parsed.text.replace("[TABLE]", table_plaintext)
    else:
        fallback = prefix + parsed.text

    # --- Build Block Kit blocks ---
    blocks: list[dict] = []

    if parsed.table:
        body = prefix + parsed.text
        parts = body.split("[TABLE]")
        before = parts[0].strip()
        after = parts[1].strip() if len(parts) > 1 else ""

        if before:
            blocks.append(_section(before))
        blocks.append(_table_block(parsed.table))
        if after:
            blocks.append(_section(after))
    else:
        blocks.append(_section(prefix + parsed.text))

    return fallback, blocks
