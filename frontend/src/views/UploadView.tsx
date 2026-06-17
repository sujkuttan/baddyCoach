import { useState, useRef, useEffect } from 'react';
import { uploadVideo, processVideo, getJob } from '../utils/api';
import { setVideoFile } from '../utils/videoStore';

interface UploadViewProps {
  onJobCreated: (jobId: string) => void;
  onLoadReport?: (report: any) => void;
}

interface Job {
  id: string;
  filename: string;
  status: string;
  created_at: string;
  stages_completed?: string[];
}

export function UploadView({ onJobCreated, onLoadReport }: UploadViewProps) {
  const [file, setFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [dragActive, setDragActive] = useState(false);
  const [jobs, setJobs] = useState<Job[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const [stagedReport, setStagedReport] = useState<any>(null);
  const [stagedVideoName, setStagedVideoName] = useState<string>('');

  useEffect(() => {
    fetch('/api/jobs')
      .then(r => r.json())
      .then(data => setJobs((data.jobs || []).reverse()))
      .catch(() => {});
  }, []);

  const validateAndSet = (f: File) => {
    const ext = f.name.split('.').pop()?.toLowerCase();
    if (!['mp4', 'mov', 'avi'].includes(ext || '')) {
      setError('Unsupported format. Use MP4, MOV, or AVI.');
      return;
    }
    if (f.size > 2 * 1024 * 1024 * 1024) {
      setError('File too large. Maximum 2GB.');
      return;
    }
    setFile(f);
    setError('');
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    const dropped = e.dataTransfer.files[0];
    if (dropped) validateAndSet(dropped);
  };

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError('');
    try {
      const { job_id } = await uploadVideo(file);
      localStorage.setItem('baddycoach_job_id', job_id);
      await processVideo(job_id);
      onJobCreated(job_id);
    } catch (e: any) {
      setError(e.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  const handleResume = (job: Job) => {
    localStorage.setItem('baddycoach_job_id', job.id);
    if (job.status === 'completed') {
      onJobCreated(job.id);
    } else {
      onJobCreated(job.id);
    }
  };

  const statusColor = (s: string) => {
    if (s === 'completed') return 'text-feather-green';
    if (s === 'processing') return 'text-shuttle-lime';
    if (s === 'error') return 'text-error-red';
    return 'text-text-muted';
  };

  const statusLabel = (s: string) => s.toUpperCase();

  return (
    <div className="min-h-screen court-pattern flex items-center justify-center p-6">
      <div className="w-full max-w-2xl animate-entrance">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 mb-4 px-4 py-1.5 rounded-full border border-court-line/30 bg-court-surface/50">
            <div className="w-2 h-2 rounded-full bg-shuttle-lime pulse-live" />
            <span className="font-mono text-xs text-text-secondary tracking-widest uppercase">AI-Powered Analysis</span>
          </div>
          <h1 className="font-display text-7xl md:text-8xl tracking-tight text-text-primary leading-none">
            COURT<span className="text-shuttle-lime">VISION</span>
          </h1>
          <p className="font-body text-text-secondary mt-3 text-lg">
            Upload a match. Get coach-grade insights.
          </p>
        </div>

        {/* Upload Zone */}
        <div
          onDrop={handleDrop}
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onClick={() => inputRef.current?.click()}
          className={`
            relative cursor-pointer rounded-2xl border-2 border-dashed p-16 text-center
            transition-all duration-300 group
            ${dragActive
              ? 'border-shuttle-lime bg-shuttle-lime/5 scale-[1.02]'
              : file
                ? 'border-feather-green/40 bg-feather-green/5'
                : 'border-court-line/40 bg-court-surface/30 hover:border-shuttle-lime/50 hover:bg-court-surface/50'
            }
          `}
        >
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.mov,.avi"
            onChange={e => e.target.files?.[0] && validateAndSet(e.target.files[0])}
            className="hidden"
          />

          {/* Corner brackets */}
          <div className="bracket-frame absolute inset-4 pointer-events-none" />

          {file ? (
            <div className="space-y-3">
              <div className="w-16 h-16 mx-auto rounded-xl bg-feather-green/10 flex items-center justify-center">
                <svg className="w-8 h-8 text-feather-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <p className="font-mono text-sm text-feather-green">{file.name}</p>
              <p className="font-mono text-xs text-text-muted">{(file.size / 1024 / 1024).toFixed(1)} MB</p>
            </div>
          ) : (
            <div className="space-y-4">
              <div className="w-16 h-16 mx-auto rounded-xl bg-court-line/20 flex items-center justify-center group-hover:bg-shuttle-lime/10 transition-colors">
                <svg className="w-8 h-8 text-text-muted group-hover:text-shuttle-lime transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <div>
                <p className="font-body text-text-secondary text-lg">Drop your match video here</p>
                <p className="font-mono text-xs text-text-muted mt-2">MP4, MOV, AVI — up to 2GB</p>
              </div>
            </div>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="mt-4 p-4 rounded-xl bg-error-red/10 border border-error-red/20">
            <p className="font-mono text-sm text-error-red">{error}</p>
          </div>
        )}

        {/* Action */}
        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className={`
            w-full mt-6 py-4 rounded-xl font-display text-2xl tracking-wider
            transition-all duration-300 relative overflow-hidden
            ${file && !uploading
              ? 'bg-shuttle-lime text-court-dark hover:bg-shuttle-lime-dim hover:scale-[1.01] active:scale-[0.99]'
              : 'bg-court-surface text-text-muted cursor-not-allowed'
            }
          `}
        >
          {uploading ? (
            <span className="flex items-center justify-center gap-3">
              <svg className="animate-spin h-5 w-5" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
              UPLOADING...
            </span>
          ) : (
            'START ANALYSIS'
          )}
        </button>

        {/* Job History */}
        {jobs.length > 0 && (
          <div className="mt-10">
            <div className="flex items-center justify-between mb-4">
              <h2 className="font-display text-lg text-text-primary tracking-wide">RECENT MATCHES</h2>
              <span className="font-mono text-[10px] text-text-muted">{jobs.length} TOTAL</span>
            </div>
            <div className="space-y-2">
              {jobs.slice(0, 5).map(job => (
                <button
                  key={job.id}
                  onClick={() => handleResume(job)}
                  className="w-full flex items-center justify-between px-4 py-3 rounded-xl bg-court-mid/60 border border-court-line/15 hover:border-shuttle-lime/30 transition-colors text-left group"
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className={`w-2 h-2 rounded-full flex-shrink-0 ${
                      job.status === 'completed' ? 'bg-feather-green' :
                      job.status === 'processing' ? 'bg-shuttle-lime pulse-live' :
                      job.status === 'error' ? 'bg-error-red' : 'bg-text-muted'
                    }`} />
                    <div className="min-w-0">
                      <p className="font-mono text-sm text-text-primary truncate">{job.filename}</p>
                      <p className="font-mono text-[10px] text-text-muted">{job.id}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 flex-shrink-0">
                    <span className={`font-mono text-[10px] tracking-wider ${statusColor(job.status)}`}>
                      {statusLabel(job.status)}
                    </span>
                    <svg className="w-4 h-4 text-text-muted group-hover:text-shuttle-lime transition-colors" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Load Report + Video */}
        {onLoadReport && (
          <div className="mt-8 text-center">
            <div className="flex items-center gap-3 mb-3">
              <div className="flex-1 h-px bg-court-line/20" />
              <span className="font-mono text-[10px] text-text-muted tracking-widest">OR</span>
              <div className="flex-1 h-px bg-court-line/20" />
            </div>

            {stagedReport ? (
              <div className="space-y-4">
                <div className="p-4 rounded-xl bg-court-mid/60 border border-court-line/15">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className="w-2 h-2 rounded-full bg-feather-green" />
                      <span className="font-mono text-xs text-feather-green">Report loaded</span>
                    </div>
                    <button onClick={() => { setStagedReport(null); setStagedVideoName(''); setVideoFile(null); }} className="font-mono text-[10px] text-text-muted hover:text-error-red transition-colors">REMOVE</button>
                  </div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${stagedVideoName ? 'bg-feather-green' : 'bg-text-muted'}`} />
                      <span className="font-mono text-xs text-text-secondary">{stagedVideoName || 'No video selected'}</span>
                    </div>
                    <label className="font-mono text-[10px] text-shuttle-lime hover:underline cursor-pointer">
                      {stagedVideoName ? 'CHANGE' : 'ADD VIDEO'}
                      <input type="file" accept=".mp4,.mov,.avi" className="hidden" onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (!f) return;
                        setVideoFile(f);
                        setStagedVideoName(f.name);
                      }} />
                    </label>
                  </div>
                </div>
                <button
                  onClick={() => onLoadReport(stagedReport)}
                  className="w-full py-4 rounded-xl font-display text-2xl tracking-wider bg-shuttle-lime text-court-dark hover:bg-shuttle-lime-dim hover:scale-[1.01] active:scale-[0.99] transition-all duration-300"
                >
                  VIEW REPORT
                </button>
              </div>
            ) : (
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <label className="inline-flex items-center gap-2 px-4 py-2 rounded-xl border border-court-line/30 bg-court-surface/30 hover:border-shuttle-lime/30 hover:bg-court-surface/50 transition-colors cursor-pointer">
                  <svg className="w-4 h-4 text-text-muted" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                  <span className="font-mono text-xs text-text-secondary">Report JSON</span>
                  <input
                    type="file"
                    accept=".json"
                    className="hidden"
                    onChange={async (e) => {
                      const f = e.target.files?.[0];
                      if (!f) return;
                      try {
                        const text = await f.text();
                        setStagedReport(JSON.parse(text));
                      } catch {
                        setError('Invalid report file');
                      }
                    }}
                  />
                </label>
              </div>
            )}
          </div>
        )}

        {/* Footer info */}
        <div className="mt-8 flex items-center justify-center gap-6 text-text-muted">
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span className="font-mono text-xs">GPU accelerated</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            <span className="font-mono text-xs">Local processing</span>
          </div>
          <div className="flex items-center gap-2">
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span className="font-mono text-xs">14-stage pipeline</span>
          </div>
        </div>
      </div>
    </div>
  );
}
