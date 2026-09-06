import { adaptAnalyzeResponse } from './adapter';
import { Result } from './sample';

export async function analyzeAudio(file: File, channel?: string): Promise<Result> {
  const formData = new FormData();
  formData.append('file', file);
  
  if (typeof window !== 'undefined') {
    const webhook = localStorage.getItem('vfd_webhook_url');
    if (webhook) formData.append('webhook_url', webhook);
    const smtp = localStorage.getItem('vfd_smtp_email');
    if (smtp) formData.append('smtp_email', smtp);
    const speakerId = localStorage.getItem('vfd_speaker_id');
    if (speakerId) formData.append('speaker_id', speakerId);
  }

  if (channel && channel !== 'none') {
    formData.append('simulate_codec', channel.replace('-', '_'));
  }

  try {
    const response = await fetch('http://localhost:8000/v1/analyze', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.statusText}`);
    }

    const json = await response.json();
    return adaptAnalyzeResponse(json);
  } catch (error) {
    console.error('API Error:', error);
    throw error;
  }
}
