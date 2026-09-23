#!/usr/bin/env python3
"""Manual check that email delivery is configured correctly.
Usage: TEST_NOTIFY_EMAIL=you@example.com python scripts/check_email.py
"""
import os
import sys

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    recipient = os.environ.get("TEST_NOTIFY_EMAIL")
    if not recipient:
        raise EnvironmentError("Set TEST_NOTIFY_EMAIL to the address you want the test email sent to")

    from app.crawl import send_notification

    result = send_notification(
        subject="[Test] Related Work Agent email check",
        message="If you see this, email delivery is working.",
        project_emails=recipient,
    )
    print("Email sent:", result)
