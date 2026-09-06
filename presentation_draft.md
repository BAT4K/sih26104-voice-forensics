# SIH 2026 Presentation Draft (Cipher UI / Voice Forensics)

Hand this document directly to your teammates making the PowerPoint. It contains the exact bullet points and technical details they need to fill out all 6 required slides for the SIH 2026 pitch.

---

## Slide 1: Title Page
*(Fill these in with your exact registration details)*
- **Hackathon:** Smart India Hackathon 2026
- **Problem Statement ID:** 26104
- **Problem Statement Title:** AI Voice Clone Detection for Indic Telephony
- **Theme:** Security & Surveillance (or relevant theme)
- **PS Category:** Software
- **Team ID:** [Your Team ID]
- **Team Name:** [Your Team Name]

---

## Slide 2: Idea Title & Proposed Solution
**Idea Title:** Indic Voice Forensic Inspector (VFD)

**Proposed Solution:**
- An enterprise-grade, end-to-end voice forensic platform designed to intercept and analyze live phone calls to detect AI-generated voice clones in real-time.
- **How it addresses the problem:** Unlike standard deepfake detectors that fail on low-quality phone calls, our system dynamically degrades audio during training to survive 8kHz telecom compression (AMR-NB).
- **Innovation & Uniqueness:** We go beyond a simple "Fake/Real" binary. We use a **4-Tier Forensic Architecture**:
  1. Identifies if the voice is AI-generated.
  2. Pinpoints exactly *which* neural vocoder family generated it (e.g., HiFi-GAN, Vocos).
  3. Uses mathematical distance tracking to detect "Zero-Day" (never-before-seen) AI models.
  4. Cross-verifies the caller's identity against past biometric voiceprints to catch human impersonators.

---

## Slide 3: Technical Approach
**Technologies Used:**
- **Machine Learning Core:** PyTorch, Hugging Face Transformers (`wav2vec2-xls-r-300m`), SpeechBrain (`ECAPA-TDNN`).
- **Backend & Enterprise Integration:** FastAPI, Python, **gRPC** (for high-throughput telecom streaming), Webhooks & SMTP for dynamic alerting.
- **Frontend Dashboard:** Next.js (React), TailwindCSS, custom WebGL/Canvas rendering.

**Methodology / Implementation Flow:**
1. **Ingestion:** Audio is streamed via REST or gRPC to the backend.
2. **Analysis:** The audio passes through the Wav2Vec2 transformer backbone, extracting phonetic features specialized for Indic languages (Hindi, Marathi, etc.).
3. **Forensic Scoring:** The system calculates a unified Risk Score (0-100) based on AI artifacts and biological speaker consistency.
4. **Automated Workflows:** If the risk exceeds configurable thresholds, the `WorkflowEngine` instantly fires Webhooks and SMTP emails to bank analysts.

*(Tip for Teammates: Put a flowchart image on this slide showing Audio -> FastAPI/gRPC -> Wav2Vec2 -> Next.js Dashboard & Alerts)*

---

## Slide 4: Feasibility and Viability
**Feasibility:**
- The architecture is highly scalable. The forensic engine is decoupled from the UI, allowing telecom providers to route millions of calls directly into the gRPC microservice without overhead.

**Potential Challenges & Risks:**
- **Telephony Compression:** Phone networks destroy the high-frequency artifacts that most AI detectors rely on.
- **Zero-Day Models:** Scammers constantly invent new AI architectures that our model wasn't trained on.

**Strategies for Overcoming Challenges:**
- **Custom Telephony Degrader:** We built a custom audio degrader that simulates G.711 / AMR-NB packet loss during training, forcing the model to learn robust, compression-resistant features.
- **Mahalanobis Novelty Detection:** We fitted a Mahalanobis distance scorer. If an audio file lands mathematically outside our known AI clusters, it is instantly flagged as an "Unknown System", protecting against future, unreleased AI models.

---

## Slide 5: Impact and Benefits
**Potential Impact on Target Audience:**
- **Banking Sector:** Prevents devastating financial losses by intercepting AI-driven social engineering attacks before authorization is granted.
- **Telecom Providers:** Allows networks to flag suspicious calls in real-time, protecting vulnerable demographics (like the elderly) from impersonation scams.

**Benefits of the Solution:**
- **Economic:** Drastically reduces fraud liability for Indian financial institutions.
- **Social:** Restores trust in digital and telephonic communications across India. 
- **Security:** The integration of cross-session speaker verification means we don't just catch AI; we catch human fraudsters trying to mimic legitimate customers.

---

## Slide 6: Research and References
- **Wav2Vec2-XLS-R:** Babu et al., "XLS-R: Self-supervised Cross-lingual Speech Representation Learning at Scale" (Meta AI).
- **Speaker Verification:** SpeechBrain ECAPA-TDNN (Desplanques et al., "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation").
- **Dataset:** Google Fleurs (High-quality Indic read-speech) mixed with SOTA synthetic generation (XTTS-v2, IndicF5, Suno Bark).
- **Novelty Detection:** Mahalanobis Distance for Out-of-Distribution Detection.
