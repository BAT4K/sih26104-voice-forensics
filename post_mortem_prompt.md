# Critical Review: ML Generalization & Zero-Day Vulnerabilities

## Core Philosophy: "No Stone Unturned"
Our architecture was built on a singular premise: **Deepfake detection must happen at every possible level.** Relying solely on a single Machine Learning model is a known vulnerability in cybersecurity, which is why our system fuses together:
1. **Tier 1 & 2 Neural Networks:** To detect known acoustic anomalies and classify specific synthetic families.
2. **Deterministic DSP Math Engine:** A mathematical fail-safe that inspects the raw frequencies, noise floors, and physical physics of the audio for microscopic digital artifacts.
3. **Provenance Watermark Checking:** Cryptographic verification of known benign audio.

## The Core Problem We Are Facing Today
Despite this robust architecture, our current `vfd-v1` machine learning checkpoint is suffering from **severe overfitting and a failure to generalize to Zero-Day Attacks.** 

### 1. The ML Model Fails on Analysis-Synthesis Attacks
The model successfully detects traditional Text-to-Speech (TTS) architectures (like standard HiFi-GAN or Legacy Autoregressive models) because it has learned to listen for unnatural pacing, robotic intonation, and weird breathing gaps. 

However, it completely fails against **Zero-Day Analysis-Synthesis Attacks**. When we strip a real human voice down to phonetic tokens and reconstruct it using modern neural codecs (like EnCodec, DAC, or Vocos), the pacing, emotion, and breathing remain 100% human. Because the model overfitted to *pacing* rather than *microscopic digital artifacts*, it gets fooled and blindly classifies these studio-grade fakes as `GENUINE HUMAN VOICE`.

### 2. The Multi-Class Family Attribution is Uncalibrated
Even when the Tier 1 model correctly identifies that an audio file is spoofed, the Tier 2 network (which tries to identify the specific AI family) frequently panics when faced with an architecture it hasn't seen in its training data (like proprietary Azure Edge TTS). Because it doesn't recognize the specific synthetic fingerprint, it defaults to its highest-confidence fallback bucket, which is incorrectly labeled as `"real"`.

### 3. The DSP Engine is Hyper-Sensitive to Baselines
Our deterministic DSP engine successfully catches the anomalies that the ML model misses, proving our "no stones unturned" philosophy works. However, the DSP engine is extremely sensitive to its mathematical baseline. If a genuine human caller happens to have a very choppy connection (high pause rate) or a perfectly silent background (low noise floor), the DSP math violently penalizes the audio, boosting the Risk Score to 100/100 and causing a **False Positive on genuine human speech**.

## The Required Fix (Prompt for the Next Phase)
To permanently solve this without relying on hardcoded overrides, we must rebuild the training pipeline with the following instructions:

> **System Prompt / Engineering Goal:**
> We need to harden our Voice Forensics model to ensure it leaves no stones unturned. Currently, our model is overfitting to traditional TTS artifacts and failing to catch Zero-Day Analysis-Synthesis attacks (like DAC and Vocos). 
> 
> **Phase 1: Dataset Augmentation**
> We must aggressively augment our training dataset. Do not just train on TTS; we need to run our genuine human datasets through EnCodec, DAC, and Vocos to create a massive corpus of high-quality Analysis-Synthesis fakes. The model must learn to detect the microscopic codec artifacts, not just human pacing.
> 
> **Phase 2: DSP Calibration Pipeline**
> The DSP Engine must be dynamically calibrated. We need to implement a dynamic baseline that adjusts its expected standard deviations based on the telephony channel (Wideband vs. Narrowband G.711). We need a softer log-odds penalty for standard statistical outliers to prevent false positives on genuine callers with choppy connections.
> 
> **Phase 3: Tier 2 Fallback Class**
> The Tier 2 Family Classification network must be retrained to include an `unknown_synthetic` bucket. When it detects a spoofed file but doesn't recognize the specific architecture, it must confidently output `unknown_synthetic` rather than panicking and defaulting to `real`.
