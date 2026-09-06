export interface BackendEvidence {
  name: string;
  value: number | string;
  unit: string;
  reading: string;
}

export interface BackendSpeaker {
  decision: "accept" | "reject";
  score: number;
}

export interface BackendResponse {
  request_id: string;
  verdict: string;
  risk_score: number;
  band: string;
  recommended_action: string;
  confidence: number;
  tier1_spoof_probability: number;
  tier2_family: string;
  tier2_family_label: string;
  tier2_probabilities: Record<string, number>;
  tier3_novelty_distance: number | null;
  tier3_is_unknown_system: boolean | null;
  speaker: BackendSpeaker | null;
  evidence: BackendEvidence[];
  window_scores: number[];
  seconds_analysed: number;
  latency_ms: number;
}
