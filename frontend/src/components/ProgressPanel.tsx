

interface SparklineProps {
  values: number[];
  width?: number;
  height?: number;
  improving?: boolean;
}

function Sparkline({ values, width = 80, height = 24, improving = true }: SparklineProps) {
  if (values.length < 2) return null;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * width;
    const y = height - ((v - min) / range) * (height - 4) - 2;
    return `${x},${y}`;
  });
  const color = improving ? '#6ee7b7' : '#f59e0b';
  return (
    <svg width={width} height={height} className="flex-shrink-0">
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

interface Headline {
  metric: string;
  label: string;
  pct_change: number;
  direction: string;
  detail: string;
  sparkline: number[];
}

interface ProgressPanelProps {
  headlines: Headline[];
  sparklines: Record<string, number[]>;
  nSessions: number;
  playerKey: string;
}

export function ProgressPanel({ headlines, sparklines, nSessions, playerKey }: ProgressPanelProps) {
  if (nSessions < 2) {
    return (
      <div className="px-4 py-6 text-center">
        <p className="font-body text-sm text-text-muted">
          Analyze at least 2 sessions for <span className="text-shuttle-lime font-semibold">{playerKey}</span> to see progress trends.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 mb-4">
        <div className="w-6 h-6 rounded-md bg-shuttle-lime/15 flex items-center justify-center">
          <svg className="w-3.5 h-3.5 text-shuttle-lime" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
          </svg>
        </div>
        <h3 className="font-display text-xl text-shuttle-lime tracking-wide">PROGRESS TRENDS</h3>
        <span className="font-mono text-xs text-text-muted ml-auto">{nSessions} sessions</span>
      </div>

      <div className="space-y-2">
        {headlines.map((h, i) => {
          const isImproving = h.direction === 'improving';
          const isDeclining = h.direction === 'declining';
          const sparkVals = sparklines[h.metric] || h.sparkline;
          return (
            <div
              key={i}
              className={`flex items-center gap-3 px-4 py-3 rounded-xl border ${
                isImproving ? 'bg-feather-green/5 border-feather-green/10' :
                isDeclining ? 'bg-net-amber/5 border-net-amber/10' :
                'bg-court-surface border-court-border'
              }`}
            >
              <div className="flex-1 min-w-0">
                <p className="font-body text-sm text-text-primary">{h.detail}</p>
              </div>
              {sparkVals.length >= 2 && (
                <Sparkline values={sparkVals} improving={isImproving || h.direction === 'stable'} />
              )}
              <span className={`font-mono text-xs font-semibold ${
                isImproving ? 'text-feather-green' :
                isDeclining ? 'text-net-amber' :
                'text-text-muted'
              }`}>
                {h.pct_change > 0 ? '+' : ''}{(h.pct_change * 100).toFixed(0)}%
              </span>
            </div>
          );
        })}
      </div>

      {headlines.length === 0 && (
        <p className="font-body text-sm text-text-muted text-center py-4">
          No significant trends detected yet. Keep playing!
        </p>
      )}
    </div>
  );
}
