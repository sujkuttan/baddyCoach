import { useState, useEffect } from 'react';
import { UploadView } from './views/UploadView';
import { ProcessingView } from './views/ProcessingView';
import { ReportView } from './views/ReportView';
import { ProgressView } from './views/ProgressView';
import { CourtCornerSetup } from './views/CourtCornerSetup';
import { LabelingView } from './views/LabelingView';
import { getJob } from './utils/api';
import { getVideoObjectURL } from './utils/videoStore';

type AppState = 'upload' | 'setup_court' | 'processing' | 'report' | 'progress' | 'labeling';

const STORAGE_KEY = 'baddycoach_job_id';

function App() {
  const [state, setState] = useState<AppState>('upload');
  const [jobId, setJobId] = useState<string | null>(null);
  const [poseModel, setPoseModel] = useState<string>('rtmpose');
  const [sampleRate, setSampleRate] = useState<number>(0);
  const [loadedReport, setLoadedReport] = useState<any>(null);
  const [restoring, setRestoring] = useState(true);
  const [progressPlayerKey, setProgressPlayerKey] = useState<string>('player_1');
  const [labelingReport, setLabelingReport] = useState<any>(null);
  const [redoCourt, setRedoCourt] = useState(false);

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
      } else if (job.status === 'uploaded') {
        setJobId(saved);
        setState('setup_court');
      } else if (job.status === 'processing') {
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

  const handleLabeling = (report: any) => {
    setLabelingReport(report);
    setState('labeling');
  };

  const handleRedoCourt = (jobId: string) => {
    setRedoCourt(true);
    setState('setup_court');
  };

  const handleRedoCourtComplete = () => {
    setRedoCourt(false);
    setState('report');
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
        <CourtCornerSetup
          jobId={jobId}
          poseModel={poseModel}
          sampleRate={sampleRate}
          onComplete={redoCourt ? handleRedoCourtComplete : handleCourtComplete}
          onBack={redoCourt ? () => { setRedoCourt(false); setState('report'); } : handleBack}
          redoMode={redoCourt}
        />
      )}
      {state === 'processing' && jobId && (
        <ProcessingView jobId={jobId} onComplete={() => setState('report')} />
      )}
      {state === 'report' && (
        <ReportView jobId={jobId} reportData={loadedReport} onBack={handleBack} onViewProgress={handleViewProgress} onLabeling={(report: any) => { setLabelingReport(report); setState('labeling'); }} onRedoCourt={handleRedoCourt} />
      )}
      {state === 'labeling' && labelingReport && (
        <LabelingView shots={labelingReport.shots || []} videoUrl={!jobId ? getVideoObjectURL() : null} jobId={jobId} onBack={() => setState('report')} />
      )}
      {state === 'progress' && (
        <ProgressView playerKey={progressPlayerKey} onBack={() => setState('report')} />
      )}
    </div>
  );
}

export default App;
