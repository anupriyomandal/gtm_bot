"""
District Coverage Slack Bot
Receives Slack events via HTTP webhook, runs the District Coverage agent,
and posts the answer back. Deployed on Railway.
"""

import os
import re
import threading
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from openai import OpenAI
from pydantic import BaseModel
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

load_dotenv(Path(__file__).parent / ".env")

from agents.router import run_routed_agent  # noqa: E402 — after load_dotenv
from agents.formatter import format_for_slack  # noqa: E402

_oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# ---------------------------------------------------------------------------
# Slack app
# ---------------------------------------------------------------------------

slack_app = App(
    token=os.environ["SLACK_BOT_TOKEN"],
    signing_secret=os.environ["SLACK_SIGNING_SECRET"],
)

# Per-channel conversation history (in-memory, resets on redeploy)
_histories: dict[str, list] = {}
_histories_lock = threading.Lock()


def _get_history(channel: str) -> list | None:
    with _histories_lock:
        return _histories.get(channel)


def _set_history(channel: str, history: list) -> None:
    with _histories_lock:
        _histories[channel] = history


class _ChannelPost(BaseModel):
    wants_channel_post: Literal["yes", "no"]


def _wants_channel_post(text: str) -> bool:
    """Detect if the user wants the answer posted to the main channel."""
    resp = _oai.beta.chat.completions.parse(
        model="gpt-5.4-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You detect whether a user wants a bot response posted to the main Slack channel "
                    "(visible to everyone) rather than as a private thread reply."
                ),
            },
            {"role": "user", "content": text},
        ],
        response_format=_ChannelPost,
    )
    parsed = resp.choices[0].message.parsed
    return parsed.wants_channel_post == "yes" if parsed else False


def _process(text: str, user: str, channel: str, thread_ts: str, post_to_channel: bool, client) -> None:
    """Run the agent in a background thread and post the result back."""
    try:
        history = _get_history(channel)
        answer, new_history = run_routed_agent(text, history=history, verbose=False)
        fallback_text, blocks = format_for_slack(answer)
        _set_history(channel, new_history)
    except Exception as exc:
        fallback_text = f"Sorry, something went wrong: {exc}"
        blocks = []

    if post_to_channel:
        mention = f"<@{user}>"
        if blocks:
            mention_block = {"type": "section", "text": {"type": "mrkdwn", "text": mention}}
            client.chat_postMessage(
                channel=channel,
                text=f"{mention} {fallback_text}",
                blocks=[mention_block] + blocks,
            )
        else:
            client.chat_postMessage(
                channel=channel,
                text=f"{mention} {fallback_text}",
            )
    else:
        if blocks:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=fallback_text,
                blocks=blocks,
            )
        else:
            client.chat_postMessage(
                channel=channel,
                thread_ts=thread_ts,
                text=fallback_text,
            )

    try:
        client.reactions_remove(channel=channel, timestamp=thread_ts, name="thinking_face")
    except Exception:
        pass


def _handle(event: dict, client) -> None:
    """Shared handler for mentions and DMs."""
    raw_text = event.get("text", "")
    # Strip @mention tags so the agent only sees the question
    text = re.sub(r"<@[A-Z0-9]+>", "", raw_text).strip()
    if not text:
        return

    channel = event["channel"]
    user = event.get("user", "")
    thread_ts = event.get("thread_ts") or event["ts"]
    post_to_channel = _wants_channel_post(text)

    try:
        client.reactions_add(channel=channel, timestamp=event["ts"], name="thinking_face")
    except Exception:
        pass

    threading.Thread(
        target=_process,
        args=(text, user, channel, thread_ts, post_to_channel, client),
        daemon=True,
    ).start()


@slack_app.event("app_mention")
def handle_mention(event, client):
    _handle(event, client)


@slack_app.event("message")
def handle_dm(event, client):
    # Only respond to direct messages (channel type "im"), ignore bot messages
    if event.get("channel_type") == "im" and not event.get("bot_id"):
        _handle(event, client)


# ---------------------------------------------------------------------------
# Flask webhook server
# ---------------------------------------------------------------------------

flask_app = Flask(__name__)
handler = SlackRequestHandler(slack_app)


@flask_app.route("/slack/events", methods=["POST"])
def slack_events():
    return handler.handle(request)


@flask_app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3000))
    flask_app.run(host="0.0.0.0", port=port)
