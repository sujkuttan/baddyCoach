import { useState } from 'react';
import { UploadView } from './views/UploadView';
import { ProcessingView } from './views/ProcessingView';
import { ReportView } from './views/ReportView';

type AppState = 'upload' | 'processing' | 'report';

function App() {
  const [state, setState] = useState<AppState>('upload');
  const [jobId, setJobId] = useState<string | null>(null);

  return (
    <div className="min-h-screen bg-court-dark">
      {state === 'upload' && (
        <UploadView onJobCreated={(id) => { setJobId(id); setState('processing'); }} />
      )}
      {state === 'processing' && jobId && (
        <ProcessingView jobId={jobId} onComplete={() => setState('report')} />
      )}
      {state === 'report' && jobId && (
        <ReportView jobId={jobId} onBack={() => { setState('upload'); setJobId(null); }} />
      )}
    </div>
  );
}

export default App;
