"""
District Coverage Slack Bot
Receives Slack events via HTTP webhook, runs the District Coverage agent,
and posts the answer back. Deployed on Railway.
"""

import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, request, jsonify
from slack_bolt import App
from slack_bolt.adapter.flask import SlackRequestHandler

load_dotenv(Path(__file__).parent / ".env")

from agents.district_coverage_agent import run_agent  # noqa: E402 — after load_dotenv

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


def _process(text: str, channel: str, thread_ts: str, client) -> None:
    """Run the agent in a background thread and post the result back."""
    try:
        history = _get_history(channel)
        answer, new_history = run_agent(text, history=history, verbose=False)
        _set_history(channel, new_history)
    except Exception as exc:
        answer = f"Sorry, something went wrong: {exc}"

    client.chat_postMessage(
        channel=channel,
        thread_ts=thread_ts,
        text=answer,
    )
    # Remove the thinking emoji once done
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
    # Reply in thread if already in one, otherwise start a new thread
    thread_ts = event.get("thread_ts") or event["ts"]

    # Acknowledge immediately with a thinking reaction
    try:
        client.reactions_add(channel=channel, timestamp=event["ts"], name="thinking_face")
    except Exception:
        pass

    threading.Thread(
        target=_process,
        args=(text, channel, thread_ts, client),
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
