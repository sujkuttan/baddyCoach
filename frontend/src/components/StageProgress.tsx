interface StageProgressProps {
  stages: string[];
  completedStages: string[];
  currentStage: string | null;
}

const STAGE_LABELS: Record<string, string> = {
  court_detection: 'Court Detection',
  player_tracking: 'Player Tracking',
  shuttle_tracking: 'Shuttle Tracking',
  pose_estimation: 'Pose Estimation',
  hit_frame_localization: 'Hit Detection',
  stroke_classification: 'Stroke Classification',
  player_attribution: 'Player Attribution',
  rally_segmentation: 'Rally Segmentation',
  court_position_analytics: 'Court Analytics',
  footwork_analytics: 'Footwork',
  fitness_analytics: 'Fitness',
  tactical_analytics: 'Tactical',
  technical_analytics: 'Technical',
  coach_recommendations: 'Coach Engine',
};

export function StageProgress({ stages, completedStages, currentStage }: StageProgressProps) {
  return (
    <div className="space-y-1.5">
      {stages.map((stage, idx) => {
        const isComplete = completedStages.includes(stage);
        const isRunning = currentStage === stage;
        const isPending = !isComplete && !isRunning;
        const label = STAGE_LABELS[stage] || stage.replace(/_/g, ' ');

        return (
          <div
            key={stage}
            className={`
              flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-300
              ${isRunning ? 'bg-shuttle-lime/10 border border-shuttle-lime/20' : ''}
              ${isComplete ? 'opacity-60' : ''}
            `}
            style={{ animationDelay: `${idx * 50}ms` }}
          >
            {/* Status indicator */}
            <div className="relative flex-shrink-0">
              {isComplete ? (
                <div className="w-5 h-5 rounded-full bg-feather-green/20 flex items-center justify-center">
                  <svg className="w-3 h-3 text-feather-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                  </svg>
                </div>
              ) : isRunning ? (
                <div className="w-5 h-5 rounded-full bg-shuttle-lime/20 flex items-center justify-center pulse-live">
                  <div className="w-2 h-2 rounded-full bg-shuttle-lime" />
                </div>
              ) : (
                <div className="w-5 h-5 rounded-full border border-court-line/40" />
              )}
            </div>

            {/* Stage number */}
            <span className="font-mono text-[10px] text-text-muted w-5 text-right">
              {String(idx + 1).padStart(2, '0')}
            </span>

            {/* Stage name */}
            <span className={`
              font-body text-sm flex-1
              ${isRunning ? 'text-shuttle-lime font-medium' : ''}
              ${isComplete ? 'text-text-muted line-through' : ''}
              ${isPending ? 'text-text-secondary' : ''}
            `}>
              {label}
            </span>

            {/* Status badge */}
            {isRunning && (
              <span className="font-mono text-[10px] text-shuttle-lime bg-shuttle-lime/10 px-2 py-0.5 rounded-full">
                RUNNING
              </span>
            )}
            {isComplete && (
              <span className="font-mono text-[10px] text-feather-green/60">
                DONE
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
