import { useState, useEffect } from 'react';
import { getPlayerProgress } from '../utils/api';
import { ProgressPanel } from '../components/ProgressPanel';

interface ProgressViewProps {
  playerKey?: string;
  onBack: () => void;
}

export function ProgressView({ playerKey = 'player_1', onBack }: ProgressViewProps) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [key, setKey] = useState(playerKey);

  useEffect(() => {
    setLoading(true);
    setError(null);
    getPlayerProgress(key, 5)
      .then(setData)
      .catch(err => setError(err.message))
      .finally(() => setLoading(false));
  }, [key]);

  return (
    <div className="min-h-screen bg-court-dark">
      {/* Header */}
      <header className="border-b border-court-border/30 bg-court-dark/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-4">
          <button
            onClick={onBack}
            className="flex items-center gap-2 px-3 py-2 rounded-lg text-text-muted hover:text-text-primary hover:bg-court-surface/50 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            <span className="font-body text-sm">Back</span>
          </button>
          <h1 className="font-display text-xl text-text-primary tracking-wide">PROGRESS</h1>

          {/* Player selector */}
          <div className="ml-auto flex gap-2">
            {['player_1', 'player_2'].map(p => (
              <button
                key={p}
                onClick={() => setKey(p)}
                className={`px-3 py-1.5 rounded-lg font-mono text-xs transition-colors ${
                  key === p
                    ? 'bg-shuttle-lime/15 text-shuttle-lime border border-shuttle-lime/30'
                    : 'text-text-muted border border-court-border/30 hover:bg-court-surface/30'
                }`}
              >
                {p === 'player_1' ? 'YOU' : 'OPPONENT'}
              </button>
            ))}
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-3xl mx-auto px-4 py-8">
        {loading && (
          <div className="flex items-center justify-center py-16">
            <div className="w-8 h-8 border-2 border-shuttle-lime/30 border-t-shuttle-lime rounded-full animate-spin" />
          </div>
        )}

        {error && (
          <div className="px-4 py-3 rounded-xl bg-net-amber/10 border border-net-amber/20 text-net-amber font-body text-sm">
            {error}
          </div>
        )}

        {data && (
          <ProgressPanel
            headlines={data.headlines || []}
            sparklines={data.sparklines || {}}
            nSessions={data.n_sessions || 0}
            playerKey={key}
          />
        )}
      </main>
    </div>
  );
}
