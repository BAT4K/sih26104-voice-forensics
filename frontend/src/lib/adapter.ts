import { BackendResponse, BackendEvidence } from './types';
import { Result } from './sample';

export function adaptAnalyzeResponse(data: BackendResponse): Result {
  const is_fake = data.risk_score >= 70;
  const is_unknown = data.tier3_is_unknown_system;
  
  let verdict: string = 'genuine';
  
  if (data.speaker?.decision === 'reject') {
    verdict = 'human_mimicry';
  } else if (data.risk_score >= 70) {
    verdict = 'ai_clone';
  } else if (data.risk_score > 30) {
    verdict = 'unclear';
  } else {
    verdict = 'genuine';
  }

  // Risk = probability of synthesis. 
  // The UI expects integer risk 0-100.
  const risk = Math.round(data.risk_score);

  // Re-map families to 8 keys
  const probs = data.tier2_probabilities || {};
  const mappedProbabilities = {
    real: probs['real'] || 0,
    hifigan: probs['hifigan'] || 0,
    bigvgan: probs['bigvgan'] || 0,
    vocos: probs['vocos'] || 0,
    encodec: probs['encodec'] || 0,
    modern_codec: probs['modern_codec'] || 0,
    legacy: probs['legacy_ar'] || 0,
    diffusion: probs['diffusion'] || 0,
  };

  const family = data.tier2_family === 'legacy_ar' ? 'legacy' : (data.tier2_family || 'real');
  const familyConfidence = mappedProbabilities[family as keyof typeof mappedProbabilities] || 0;

  const reasons = (data.evidence || []).map((e: BackendEvidence) => {
    // Attempt to extract sigma from reading for the sigma bar.
    let sigma = 0;
    const match = e.reading.match(/([<>]?\s*[\d\.]+)\s+standard deviations (higher|lower)/i);
    if (match) {
      let val = parseFloat(match[1].replace(/[<>]/g, '').trim());
      if (match[2].toLowerCase() === 'lower') {
        val = -val;
      }
      sigma = val;
    }
    
    return {
      name: e.name.replace(/_/g, ' '),
      value: String(e.value),
      unit: e.unit,
      note: e.reading,
      sigma: sigma
    };
  });

  return {
    verdict,
    risk,
    band: data.band || 'green',
    recommendedAction: data.recommended_action || 'No action. Proceed normally.',
    family,
    familyConfidence,
    probabilities: mappedProbabilities,
    scores: data.window_scores || [],
    reasons,
    durationSec: data.seconds_analysed || 0,
    elapsedMs: data.latency_ms || 0,
    model: 'v0.3-telephony-xlsr'
  } as unknown as Result;
}
