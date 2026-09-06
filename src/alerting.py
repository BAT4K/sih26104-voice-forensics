import os
import logging
import requests
import smtplib
from email.message import EmailMessage
from typing import Optional, Dict, Any

log = logging.getLogger("vfd.alerting")

class AlertManager:
    """Dispatches webhook and email alerts without blocking the main event loop."""
    
    def __init__(self):
        self.webhook_url = os.environ.get("VFD_WEBHOOK_URL", "")
        self.smtp_server = os.environ.get("VFD_SMTP_SERVER", "")
        self.smtp_port = int(os.environ.get("VFD_SMTP_PORT", "587"))
        self.smtp_user = os.environ.get("VFD_SMTP_USER", "")
        self.smtp_pass = os.environ.get("VFD_SMTP_PASS", "")
        self.alert_email = os.environ.get("VFD_ALERT_EMAIL", "")

    def dispatch_webhook(self, payload: Dict[str, Any], webhook_url: Optional[str] = None):
        url = webhook_url or self.webhook_url
        if not url or "v1/analyze" in url:
            return
        try:
            log.info(f"Dispatching webhook to {url}")
            requests.post(url, json=payload, timeout=5)
        except Exception as e:
            log.error(f"Failed to dispatch webhook: {e}")

    def dispatch_email(self, subject: str, body: str, alert_email: Optional[str] = None):
        email_to = alert_email or self.alert_email
        if not self.smtp_server or not email_to:
            return
        try:
            log.info(f"Dispatching email alert to {email_to}")
            msg = EmailMessage()
            msg.set_content(body)
            msg['Subject'] = subject
            msg['From'] = self.smtp_user or "vfd-alerts@localhost"
            msg['To'] = email_to

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                if self.smtp_user and self.smtp_pass:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_pass)
                server.send_message(msg)
        except Exception as e:
            log.error(f"Failed to dispatch email: {e}")

class WorkflowEngine:
    def __init__(self, green_max: float = 30.0, amber_max: float = 70.0):
        self.green_max = float(os.environ.get("VFD_GREEN_MAX", green_max))
        self.amber_max = float(os.environ.get("VFD_AMBER_MAX", amber_max))
        self.alert_manager = AlertManager()

    def evaluate(self, request_id: str, risk_score: float, is_fake: bool, context: Optional[Any] = None) -> None:
        """
        Evaluate the analysis result and trigger non-blocking alerts if needed.
        This method is meant to be called within a FastAPI BackgroundTask.
        """
        log.info(f"Evaluating workflow for {request_id}, risk={risk_score}")
        
        # Trigger condition: High risk OR explicitly Fake
        if risk_score > self.amber_max or is_fake:
            log.warning(f"[{request_id}] HIGH RISK DETECTED. Triggering alerts.")
            
            payload = {
                "request_id": request_id,
                "risk_score": risk_score,
                "is_fake": is_fake,
                "status": "BLOCKED" if risk_score > self.amber_max else "REVIEW",
                "message": "High impersonation risk detected in voice stream."
            }
            
            # Dispatch Webhook
            webhook_url = context.get("webhook_url") if context else None
            self.alert_manager.dispatch_webhook(payload, webhook_url=webhook_url)
            
            # Dispatch Email
            smtp_email = context.get("smtp_email") if context else None
            subject = f"🚨 VFD ALERT: High Risk Voice Clone Detected ({request_id})"
            body = f"Risk Score: {risk_score}\nFake Probability > 50%: {is_fake}\nAction Recommended: Verify via secondary channel immediately."
            self.alert_manager.dispatch_email(subject, body, alert_email=smtp_email)
