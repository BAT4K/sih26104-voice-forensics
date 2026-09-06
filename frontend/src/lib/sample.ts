/**
 * A fixture for design review only.
 *
 * It is reachable exclusively at `/?sample=1`, and whenever it is on screen the
 * page shows a persistent SAMPLE marker. Nothing here is ever used as a
 * fallback when the detector is unreachable — an unreachable detector renders
 * the unavailable state with every number blank. Inventing a verdict is worse
 * than showing none.
 */
export const SAMPLE = {
  verdict: "ai_clone" as const,
  risk: 98,
  band: "red",
  recommendedAction: "Do not act on this call. Call back on a number already on file before approving any transfer or disclosing anything.",
  family: "hifigan",
  familyConfidence: 0.921,
  probabilities: {
    real: 0.021, hifigan: 0.921, bigvgan: 0.006, vocos: 0.019,
    encodec: 0.008, modern_codec: 0.004, legacy: 0.017, diffusion: 0.000,
  },
  scores: [0.94, 0.97, 0.96, 0.98, 0.97, 0.99, 0.98, 0.97, 0.98],
  reasons: [
    { name: "pause rate",          value: "134.1", unit: "/ min",
      note: "pauses and breath gaps per minute sit 20.0 standard deviations above genuine telephone speech",
      sigma: 20.0 },
    { name: "hf energy ratio",     value: "0.00",  unit: "",
      note: "share of energy in the upper band sits 1.5 standard deviations below genuine telephone speech",
      sigma: -1.5 },
    { name: "noise floor flatness", value: "0.139", unit: "",
      note: "texture of the background between words sits 1.4 standard deviations below genuine telephone speech",
      sigma: -1.4 },
  ],
  durationSec: 8.5,
  elapsedMs: 16768,
  model: "v0.3-telephony-xlsr",
};

export type Result = typeof SAMPLE;
