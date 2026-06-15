import { useState, useEffect } from 'react';
import { UploadView } from './views/UploadView';
import { ProcessingView } from './views/ProcessingView';
import { ReportView } from './views/ReportView';
import { getJob } from './utils/api';

type AppState = 'upload' | 'processing' | 'report';

const STORAGE_KEY = 'baddycoach_job_id';

function App() {
  const [state, setState] = useState<AppState>('upload');
  const [jobId, setJobId] = useState<string | null>(null);
  const [loadedReport, setLoadedReport] = useState<any>(null);
  const [restoring, setRestoring] = useState(true);

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

  const handleJobCreated = (id: string) => {
    localStorage.setItem(STORAGE_KEY, id);
    setJobId(id);
    setState('processing');
  };

  const handleLoadReport = (report: any) => {
    setLoadedReport(report);
    setJobId(null);
    setState('report');
  };

  const handleBack = () => {
    localStorage.removeItem(STORAGE_KEY);
    setJobId(null);
    setLoadedReport(null);
    setState('upload');
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
      {state === 'processing' && jobId && (
        <ProcessingView jobId={jobId} onComplete={() => setState('report')} />
      )}
      {state === 'report' && (
        <ReportView jobId={jobId} reportData={loadedReport} onBack={handleBack} />
      )}
    </div>
  );
}

export default App;
