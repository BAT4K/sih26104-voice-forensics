"use client";

import { useState, useEffect } from "react";
import SmokeWarp from "@/components/SmokeWarp";
import Link from "next/link";

export default function SettingsPage() {
  const [webhookUrl, setWebhookUrl] = useState("");
  const [smtpEmail, setSmtpEmail] = useState("");
  const [speakerId, setSpeakerId] = useState("");
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    if (typeof window !== "undefined") {
      setWebhookUrl(localStorage.getItem("vfd_webhook_url") || "");
      setSmtpEmail(localStorage.getItem("vfd_smtp_email") || "");
      setSpeakerId(localStorage.getItem("vfd_speaker_id") || "");
    }
  }, []);

  const handleSave = () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("vfd_webhook_url", webhookUrl);
      localStorage.setItem("vfd_smtp_email", smtpEmail);
      localStorage.setItem("vfd_speaker_id", speakerId);
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    }
  };

  return (
    <>
      <SmokeWarp />
      <main className="mx-auto w-full max-w-[720px] px-14 pt-32 pb-40">
        <div className="mb-12 flex items-center justify-between">
          <Link href="/" className="eyebrow hover:text-white transition-colors">
            ← Back to Dashboard
          </Link>
          <div className="eyebrow">Workflows & Alerts</div>
        </div>

        <div className="scrim mb-10">
          <h1 className="caps t1 text-[42px]">
            Alerting <span className="accent">Infrastructure</span>
          </h1>
          <p className="t3 mt-4">
            Configure dynamic workflows. Settings are injected directly into the analysis payload and consumed by the backend Workflow Engine.
          </p>
        </div>

        <div className="plate px-8 py-8 space-y-8">
          <div>
            <label className="eyebrow block mb-3">Webhook URL</label>
            <input
              type="text"
              value={webhookUrl}
              onChange={(e) => setWebhookUrl(e.target.value)}
              placeholder="https://api.example.com/alerts"
              className="control"
            />
            <p className="t4 mt-2 text-[11px]">
              Triggered automatically when risk exceeds the amber threshold or upon confident synthetic detection.
            </p>
          </div>

          <div className="rule" />

          <div>
            <label className="eyebrow block mb-3">SMTP Notification Email</label>
            <input
              type="email"
              value={smtpEmail}
              onChange={(e) => setSmtpEmail(e.target.value)}
              placeholder="analyst@bank.com"
              className="control"
            />
            <p className="t4 mt-2 text-[11px]">
              Sends a formatted risk report to this address. Requires VFD_SMTP_SERVER to be configured on the backend.
            </p>
          </div>

          <div className="rule" />

          <div>
            <label className="eyebrow block mb-3">Target Speaker ID</label>
            <input
              type="text"
              value={speakerId}
              onChange={(e) => setSpeakerId(e.target.value)}
              placeholder="e.g. modi"
              className="control"
            />
            <p className="t4 mt-2 text-[11px]">
              If set, the forensic engine will run ECAPA-TDNN Cross-Speaker Verification against the enrolled voiceprint for this ID.
            </p>
          </div>

          <div className="pt-4 flex items-center gap-4">
            <button
              onClick={handleSave}
              className="px-6 py-2 rounded-md bg-white text-black font-semibold text-sm hover:bg-gray-200 transition-colors"
            >
              Save Configuration
            </button>
            {saved && <span className="text-[var(--state-genuine)] text-sm">Saved to local storage!</span>}
          </div>
        </div>
      </main>
    </>
  );
}
