import { useState } from 'react';

interface Evidence {
  finding: string;
  metrics: string[];
}

interface CoachPanelProps {
  strengths: string[];
  weaknesses: string[];
  improvements: string[];
  drills: string[];
  evidence: Evidence[];
}

export function CoachPanel({ strengths, weaknesses, drills, evidence }: CoachPanelProps) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);

  return (
    <div className="space-y-8">
      {/* Strengths */}
      {strengths.length > 0 && (
        <div className="animate-entrance" style={{ animationDelay: '100ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-md bg-feather-green/15 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-feather-green" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M5 13l4 4L19 7" />
              </svg>
            </div>
            <h3 className="font-display text-xl text-feather-green tracking-wide">STRENGTHS</h3>
          </div>
          <div className="space-y-2">
            {strengths.map((s, i) => (
              <div key={i} className="flex items-start gap-3 px-4 py-3 rounded-xl bg-feather-green/5 border border-feather-green/10">
                <span className="font-mono text-xs text-feather-green/60 mt-0.5">{String(i + 1).padStart(2, '0')}</span>
                <p className="font-body text-sm text-text-primary leading-relaxed">{s}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Weaknesses */}
      {weaknesses.length > 0 && (
        <div className="animate-entrance" style={{ animationDelay: '200ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-md bg-net-amber/15 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-net-amber" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h3 className="font-display text-xl text-net-amber tracking-wide">AREAS FOR IMPROVEMENT</h3>
          </div>
          <div className="space-y-2">
            {weaknesses.map((w, i) => (
              <div key={i} className="flex items-start gap-3 px-4 py-3 rounded-xl bg-net-amber/5 border border-net-amber/10">
                <span className="font-mono text-xs text-net-amber/60 mt-0.5">{String(i + 1).padStart(2, '0')}</span>
                <p className="font-body text-sm text-text-primary leading-relaxed">{w}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommended Drills */}
      {drills.length > 0 && (
        <div className="animate-entrance" style={{ animationDelay: '300ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-md bg-shuttle-lime/15 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-shuttle-lime" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="font-display text-xl text-shuttle-lime tracking-wide">RECOMMENDED DRILLS</h3>
          </div>
          <div className="space-y-2">
            {drills.map((d, i) => (
              <div key={i} className="flex items-start gap-3 px-4 py-3 rounded-xl bg-shuttle-lime/5 border border-shuttle-lime/10">
                <div className="w-6 h-6 rounded-full bg-shuttle-lime/15 flex items-center justify-center flex-shrink-0 mt-0.5">
                  <span className="font-mono text-xs text-shuttle-lime font-semibold">{i + 1}</span>
                </div>
                <p className="font-body text-sm text-text-primary leading-relaxed">{d}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Evidence */}
      {evidence.length > 0 && (
        <div className="animate-entrance" style={{ animationDelay: '400ms' }}>
          <div className="flex items-center gap-2 mb-3">
            <div className="w-6 h-6 rounded-md bg-smash-magenta/15 flex items-center justify-center">
              <svg className="w-3.5 h-3.5 text-smash-magenta" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
              </svg>
            </div>
            <h3 className="font-display text-xl text-smash-magenta tracking-wide">SUPPORTING EVIDENCE</h3>
          </div>
          <div className="space-y-2">
            {evidence.map((e, i) => (
              <div
                key={i}
                className="rounded-xl border border-court-line/15 overflow-hidden transition-all duration-200"
              >
                <button
                  onClick={() => setExpandedIdx(expandedIdx === i ? null : i)}
                  className="w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-court-surface/30 transition-colors"
                >
                  <svg
                    className={`w-4 h-4 text-text-muted transition-transform duration-200 ${expandedIdx === i ? 'rotate-90' : ''}`}
                    fill="none" viewBox="0 0 24 24" stroke="currentColor"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                  <p className="font-body text-sm text-text-primary flex-1">{e.finding}</p>
                </button>
                {expandedIdx === i && (
                  <div className="px-4 pb-3 pt-1 border-t border-court-line/10 bg-court-dark/30">
                    {e.metrics.map((m, j) => (
                      <p key={j} className="font-mono text-xs text-text-secondary py-1">
                        <span className="text-shuttle-lime mr-2">→</span>{m}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
