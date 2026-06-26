import { useEffect, useState, useCallback } from 'react';
import { useWebSocket } from '../hooks/useWebSocket';
import { StageProgress } from '../components/StageProgress';

const STAGES = [
  'court_detection', 'player_tracking', 'shuttle_tracking', 'pose_estimation',
  'hit_frame_localization', 'stroke_classification', 'player_attribution',
  'rally_segmentation', 'court_position_analytics', 'footwork_analytics',
  'fitness_analytics', 'tactical_analytics', 'technical_analytics',
  'coach_recommendations',
];

interface ProcessingViewProps {
  jobId: string;
  onComplete: () => void;
}

export function ProcessingView({ jobId, onComplete }: ProcessingViewProps) {
  const { events, connected } = useWebSocket(jobId);
  const [completedStages, setCompletedStages] = useState<string[]>([]);
  const [currentStage, setCurrentStage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleComplete = useCallback(() => {
    onComplete();
  }, [onComplete]);

  useEffect(() => {
    for (const event of events) {
      if (event.status === 'complete') {
        setCompletedStages(prev => {
          if (!prev.includes(event.stage)) return [...prev, event.stage];
          return prev;
        });
        setCurrentStage(null);
      } else if (event.status === 'running') {
        setCurrentStage(event.stage);
      } else if (event.status === 'failed') {
        setError(event.error || 'Processing failed');
      }
    }
  }, [events]);

  useEffect(() => {
    if (completedStages.length === STAGES.length && !error) {
      handleComplete();
    }
  }, [completedStages.length, error, handleComplete]);

  const progress = (completedStages.length / STAGES.length) * 100;

  return (
    <div className="min-h-screen court-pattern flex items-center justify-center p-6">
      <div className="w-full max-w-3xl animate-entrance">
        {/* Broadcast header bar */}
        <div className="flex items-center justify-between mb-8 px-2">
          <div className="flex items-center gap-3">
            <div className={`w-2.5 h-2.5 rounded-full ${connected ? 'bg-shuttle-lime pulse-live' : 'bg-error-red'}`} />
            <span className="font-mono text-xs text-text-secondary tracking-widest uppercase">
              {connected ? 'Processing Live' : 'Reconnecting...'}
            </span>
          </div>
          <span className="font-mono text-xs text-text-muted">
            JOB {jobId.toUpperCase()}
          </span>
        </div>

        {/* Main progress card */}
        <div className="bg-court-mid/80 backdrop-blur-sm rounded-2xl border border-court-line/20 overflow-hidden">
          {/* Top accent bar */}
          <div className="h-1 bg-court-line/20 relative">
            <div
              className="h-full bg-gradient-to-r from-shuttle-lime to-feather-green transition-all duration-700 ease-out progress-active"
              style={{ width: `${progress}%` }}
            />
          </div>

          <div className="p-8">
            {/* Progress header */}
            <div className="flex items-end justify-between mb-6">
              <div>
                <h2 className="font-display text-4xl text-text-primary tracking-wide">
                  {currentStage
                    ? (currentStage.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()))
                    : completedStages.length === STAGES.length
                      ? 'ANALYSIS COMPLETE'
                      : 'INITIALIZING'
                  }
                </h2>
                <p className="font-mono text-sm text-text-muted mt-1">
                  Stage {completedStages.length} of {STAGES.length}
                </p>
              </div>
              <div className="text-right">
                <span className="font-display text-5xl text-shuttle-lime">
                  {Math.round(progress)}
                </span>
                <span className="font-display text-2xl text-text-muted">%</span>
              </div>
            </div>

            {/* Progress bar */}
            <div className="w-full h-2 bg-court-surface rounded-full overflow-hidden mb-8">
              <div
                className="h-full bg-gradient-to-r from-shuttle-lime via-feather-green to-shuttle-lime rounded-full transition-all duration-700 ease-out"
                style={{ width: `${progress}%` }}
              />
            </div>

            {/* Stage list */}
            <div className="bg-court-dark/50 rounded-xl p-4 border border-court-line/10">
              <StageProgress
                stages={STAGES}
                completedStages={completedStages}
                currentStage={currentStage}
              />
            </div>
          </div>
        </div>

        {/* Error display */}
        {error && (
          <div className="mt-4 p-4 rounded-xl bg-error-red/10 border border-error-red/20 animate-entrance">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-error-red flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              <p className="font-mono text-sm text-error-red">{error}</p>
            </div>
          </div>
        )}

        {/* Bottom info */}
        <div className="mt-6 text-center">
          <p className="font-mono text-xs text-text-muted">
            14-stage ML pipeline — Court → Players → Shuttle → Pose → Strokes → Analytics → Coach
          </p>
        </div>
      </div>
    </div>
  );
}
