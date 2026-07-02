const API_BASE = '/api';

export async function uploadVideo(file: File): Promise<{ job_id: string }> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function processVideo(
  jobId: string,
  poseModel: string = 'rtmpose',
  sampleRate: number = 0
): Promise<void> {
  const params = new URLSearchParams();
  params.append('pose_model', poseModel);
  if (sampleRate > 0) params.append('sample_rate', String(sampleRate));
  const res = await fetch(`${API_BASE}/jobs/${jobId}/process?${params.toString()}`, { method: 'POST' });
  if (!res.ok) throw new Error(await res.text());
}

export async function getJob(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}`);
  if (!res.ok) throw new Error('Job not found');
  return res.json();
}

export async function getReport(jobId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/report`);
  if (!res.ok) throw new Error('Report not found');
  return res.json();
}

export async function getPlayerProgress(playerKey: string, window: number = 5): Promise<any> {
  const res = await fetch(`${API_BASE}/players/${encodeURIComponent(playerKey)}/progress?window=${window}`);
  if (!res.ok) throw new Error('Progress not found');
  return res.json();
}

export async function getFrame(jobId: string): Promise<Blob> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/frame`);
  if (!res.ok) throw new Error('Frame not found');
  return res.blob();
}

export async function setCourtCorners(jobId: string, corners: number[][]): Promise<void> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/court-corners`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ corners }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function getCourtCorners(jobId: string): Promise<{ corners: number[][] | null; source: string }> {
  const res = await fetch(`${API_BASE}/jobs/${jobId}/court-corners`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
