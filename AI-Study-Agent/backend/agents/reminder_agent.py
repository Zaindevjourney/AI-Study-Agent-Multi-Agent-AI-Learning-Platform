"""
reminder_agent.py
-----------------
Sends study reminders (email / push / daily reminder). This standalone
project logs reminders and includes a ready-to-use SMTP email sender -
plug in real SMTP credentials or a push-notification service (e.g. FCM)
to make it live.
"""

import os
import smtplib
from email.mime.text import MIMEText
from typing import Optional


class ReminderAgent:
    def __init__(self):
        self.smtp_host = os.getenv("SMTP_HOST")
        self.smtp_port = int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = os.getenv("SMTP_USER")
        self.smtp_pass = os.getenv("SMTP_PASS")

    def send_email(self, to_email: str, subject: str, body: str) -> str:
        if not all([self.smtp_host, self.smtp_user, self.smtp_pass]):
            return f"[LOGGED - no SMTP configured] Would email '{to_email}': {subject} - {body}"

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.smtp_user
        msg["To"] = to_email

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.sendmail(self.smtp_user, [to_email], msg.as_string())
        return "sent"

    def send_push(self, device_token: str, title: str, message: str) -> str:
        # Placeholder for FCM / APNs integration.
        return f"[LOGGED - no push provider configured] Would push to {device_token}: {title} - {message}"

    def daily_reminder_text(self, subject: str, hours: float) -> str:
        return f"📚 Reminder: Study '{subject}' for {hours} hour(s) today. You've got this!"
