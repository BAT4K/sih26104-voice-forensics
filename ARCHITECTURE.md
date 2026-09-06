# Indic Voice Forensic Inspector (SIH26104)

## 1. System Overview
This project is an AI-powered voice clone detection dashboard optimized for Indian telephony environments. It detects synthetic audio (deepfakes/voice clones) that have been transmitted over degraded cellular networks and messaging apps.

## 2. The 4-Tier Architecture
*   **Tier 0 (The Scanner):** Fast-pass detection for known commercial watermarks (e.g., ElevenLabs SynthID, Meta AudioSeal).
*   **Tier 1 (The Brain - MVP Focus):** A binary classifier (`wav2vec2-xls-r-300m` backbone) trained on heavily compressed, noisy phone-call data to detect AI manipulation in Hindi, Hinglish, and English dialects.
*   **Tier 2 (The Explainer):** A multi-class head that identifies the specific vocoder family (e.g., HiFi-GAN, Vocos, EnCodec, RVC) used to generate the fake audio, providing forensic explainability.
*   **Tier 3 (The Novelty Radar):** An anomaly detection layer measuring Mahalanobis distance from known engine clusters to flag unseen/zero-day AI synthesis methods.

## 3. Data Pipeline & Telephony Degradation
Unlike standard lab-trained models, this system simulates real-world telecommunications. All training data passes through an augmentation pipeline before hitting the model:
1.  **Resampling:** Downsampled to 8 kHz.
2.  **Compression:** Processed through G.711 (µ-law) or AMR-NB codecs.
3.  **Noise Injection:** Ambient street and babble noise added (e.g., MUSAN dataset).

## 4. MVP (v1.0) Implementation Scope
The current iteration delivers a fully functional end-to-end system:
*   **Frontend:** Next.js web dashboard with real-time inference visualizations.
*   **Audio Processing:** `torchaudio` and `librosa` for spectral feature extraction and codec simulation.
*   **Inference:** Custom-trained `wav2vec2` checkpoint running on a FastAPI backend.

## 5. Development Guidelines
*   All audio must be processed in 4.04-second windows (64,600 samples at 16kHz) to match the standard AASIST architectural constraints.
*   Do not rely on pristine high-frequency spectral artifacts above 4kHz, as telephony compression destroys them.