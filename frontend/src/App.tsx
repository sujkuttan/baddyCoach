import { useState, useEffect } from 'react';
import { UploadView } from './views/UploadView';
import { ProcessingView } from './views/ProcessingView';
import { ReportView } from './views/ReportView';
import { ProgressView } from './views/ProgressView';
import { CourtCornerSetup } from './views/CourtCornerSetup';
import { getJob } from './utils/api';

type AppState = 'upload' | 'setup_court' | 'processing' | 'report' | 'progress';

const STORAGE_KEY = 'baddycoach_job_id';

function App() {
  const [state, setState] = useState<AppState>('upload');
  const [jobId, setJobId] = useState<string | null>(null);
  const [poseModel, setPoseModel] = useState<string>('rtmpose');
  const [sampleRate, setSampleRate] = useState<number>(0);
  const [loadedReport, setLoadedReport] = useState<any>(null);
  const [restoring, setRestoring] = useState(true);
  const [progressPlayerKey, setProgressPlayerKey] = useState<string>('player_1');

  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) {
      setRestoring(false);
      return;
    }

    getJob(saved).then(job => {
      if (job.status === 'completed') {
        setJobId(saved);
        setState('report');
      } else if (job.status === 'processing' || job.status === 'uploaded') {
        setJobId(saved);
        setState('processing');
      } else {
        localStorage.removeItem(STORAGE_KEY);
      }
    }).catch(() => {
      localStorage.removeItem(STORAGE_KEY);
    }).finally(() => setRestoring(false));
  }, []);

  const handleJobCreated = (id: string, opts?: { poseModel?: string; sampleRate?: number }) => {
    localStorage.setItem(STORAGE_KEY, id);
    setJobId(id);
    setPoseModel(opts?.poseModel ?? 'rtmpose');
    setSampleRate(opts?.sampleRate ?? 0);
    setState('setup_court');
  };

  const handleLoadReport = (report: any) => {
    setLoadedReport(report);
    setJobId(null);
    setState('report');
  };

  const handleCourtComplete = () => {
    setState('processing');
  };

  const handleBack = () => {
    localStorage.removeItem(STORAGE_KEY);
    setJobId(null);
    setLoadedReport(null);
    setState('upload');
  };

  const handleViewProgress = (playerKey: string = 'player_1') => {
    setProgressPlayerKey(playerKey);
    setState('progress');
  };

  if (restoring) {
    return (
      <div className="min-h-screen court-pattern flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-shuttle-lime/30 border-t-shuttle-lime rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-court-dark">
      {state === 'upload' && (
        <UploadView onJobCreated={handleJobCreated} onLoadReport={handleLoadReport} />
      )}
      {state === 'setup_court' && jobId && (
        <CourtCornerSetup jobId={jobId} poseModel={poseModel} sampleRate={sampleRate} onComplete={handleCourtComplete} onBack={handleBack} />
      )}
      {state === 'processing' && jobId && (
        <ProcessingView jobId={jobId} onComplete={() => setState('report')} />
      )}
      {state === 'report' && (
        <ReportView jobId={jobId} reportData={loadedReport} onBack={handleBack} onViewProgress={handleViewProgress} />
      )}
      {state === 'progress' && (
        <ProgressView playerKey={progressPlayerKey} onBack={() => setState('report')} />
      )}
    </div>
  );
}

export default App;
