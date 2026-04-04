#!/usr/bin/env python3
"""Send tmux job completion emails via SMTP."""

from __future__ import annotations

import argparse
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path
import dotenv

dotenv.load_dotenv()


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def read_log_tail(log_file: str, tail_lines: int) -> str:
    path = Path(log_file)
    if not path.exists():
        return f"Log file not found: {log_file}"

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        return f"Could not read log file {log_file}: {exc}"

    if not lines:
        return "(log file is empty)"

    return "\n".join(lines[-tail_lines:])


def build_message(args: argparse.Namespace) -> EmailMessage:
    smtp_user = get_required_env("SMTP_USERNAME")
    smtp_to = get_required_env("SMTP_TO")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)

    status_text = "SUCCESS" if args.exit_code == 0 else "FAILURE"
    subject = f"[{status_text}] tmux job {args.session_name}"

    body = (
        f"Status: {status_text}\n"
        f"Session: {args.session_name}\n"
        f"Args: {args.args_text}\n"
        f"Exit code: {args.exit_code}\n"
        f"Start time: {args.start_time}\n"
        f"End time: {args.end_time}\n"
        f"Log file: {args.log_file}\n"
        "\n"
        f"Last {args.tail_lines} log lines:\n"
        f"{read_log_tail(args.log_file, args.tail_lines)}\n"
    )

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = smtp_to
    reply_to = os.getenv("SMTP_REPLY_TO")
    if reply_to:
        message["Reply-To"] = reply_to
    message.set_content(body)
    return message


def send_message(message: EmailMessage) -> None:
    smtp_host = get_required_env("SMTP_HOST")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = get_required_env("SMTP_USERNAME")
    smtp_password = get_required_env("SMTP_PASSWORD")
    use_ssl = parse_bool(os.getenv("SMTP_USE_SSL"), default=False)
    use_starttls = parse_bool(os.getenv("SMTP_USE_STARTTLS"), default=not use_ssl)

    if use_ssl:
        with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=30) as server:
            server.login(smtp_user, smtp_password)
            server.send_message(message)
        return

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        if use_starttls:
            server.starttls()
            server.ehlo()
        server.login(smtp_user, smtp_password)
        server.send_message(message)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send a tmux job completion email through SMTP."
    )
    parser.add_argument("--session-name", required=True)
    parser.add_argument("--args-text", required=True)
    parser.add_argument("--exit-code", required=True, type=int)
    parser.add_argument("--start-time", required=True)
    parser.add_argument("--end-time", required=True)
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--tail-lines", type=int, default=40)
    args = parser.parse_args()

    try:
        message = build_message(args)
        send_message(message)
    except Exception as exc:
        print(f"Failed to send email: {exc}", file=sys.stderr)
        return 1

    print("Email notification sent.")
    return 0


if __name__ == "__main__":
    print("Sending email notification...")
    raise SystemExit(main())
