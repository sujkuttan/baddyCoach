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
