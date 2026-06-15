import { useEffect, useState, useMemo } from 'react';
import { getReport } from '../utils/api';
import { VideoPlayer } from '../components/VideoPlayer';
import { ShotChart } from '../components/ShotChart';
import { CoachPanel } from '../components/CoachPanel';
import { CourtHeatmap } from '../components/CourtHeatmap';

interface ReportViewProps {
  jobId: string;
  onBack: () => void;
}

function getPlayers(report: any): string[] {
  const players = new Set<string>();
  for (const source of [report.footwork, report.fitness, report.tactical, report.technical]) {
    if (source && typeof source === 'object') {
      Object.keys(source).forEach(k => players.add(k));
    }
  }
  return Array.from(players).sort();
}

function playerLabel(id: string): string {
  if (id === 'player_1') return 'Near';
  if (id === 'player_2') return 'Far';
  return id.replace('player_', 'P');
}

export function ReportView({ jobId, onBack }: ReportViewProps) {
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'tactical' | 'technical' | 'fitness'>('overview');
  const [selectedPlayer, setSelectedPlayer] = useState<string>('');

  useEffect(() => {
    getReport(jobId).then(data => {
      setReport(data);
      const players = getPlayers(data);
      if (players.length > 0) setSelectedPlayer(players[0]);
    }).catch(console.error).finally(() => setLoading(false));
  }, [jobId]);

  const players = useMemo(() => report ? getPlayers(report) : [], [report]);

  if (loading) {
    return (
      <div className="min-h-screen court-pattern flex items-center justify-center">
        <div className="text-center animate-entrance">
          <div className="w-12 h-12 mx-auto mb-4 border-2 border-shuttle-lime/30 border-t-shuttle-lime rounded-full animate-spin" />
          <p className="font-mono text-sm text-text-muted">Loading report...</p>
        </div>
      </div>
    );
  }

  if (!report) {
    return (
      <div className="min-h-screen court-pattern flex items-center justify-center">
        <div className="text-center">
          <p className="font-display text-3xl text-text-muted mb-4">NO REPORT FOUND</p>
          <button onClick={onBack} className="font-mono text-sm text-shuttle-lime hover:underline">
            ← Back to upload
          </button>
        </div>
      </div>
    );
  }

  const shotDistribution = report.tactical?.[selectedPlayer]?.shot_distribution || report.shot_distribution || {};
  const commonPatterns = report.tactical?.[selectedPlayer]?.common_patterns || [];
  const technicalAssessments = report.technical?.[selectedPlayer] || null;
  const rallies = report.rallies || [];

  return (
    <div className="min-h-screen court-pattern">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-court-dark/90 backdrop-blur-md border-b border-court-line/15">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-text-muted hover:text-shuttle-lime transition-colors"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
              </svg>
              <span className="font-mono text-xs">NEW</span>
            </button>
            <div className="h-6 w-px bg-court-line/20" />
            <div>
              <h1 className="font-display text-2xl text-text-primary tracking-wide">
                MATCH <span className="text-shuttle-lime">REPORT</span>
              </h1>
              <p className="font-mono text-[10px] text-text-muted tracking-widest">JOB {jobId.toUpperCase()}</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* Player selector */}
            {players.length > 1 && (
              <div className="flex items-center gap-2">
                <span className="font-mono text-[10px] text-text-muted tracking-widest">PLAYER</span>
                <div className="flex items-center gap-1 bg-court-surface/50 rounded-lg p-1">
                  {players.map(p => (
                    <button
                      key={p}
                      onClick={() => setSelectedPlayer(p)}
                      className={`
                        px-3 py-1.5 rounded-md font-mono text-xs tracking-wider transition-all duration-200
                        ${selectedPlayer === p
                          ? 'bg-shuttle-lime text-court-dark'
                          : 'text-text-muted hover:text-text-primary'
                        }
                      `}
                    >
                      {playerLabel(p)}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Tab navigation */}
            <div className="flex items-center gap-1 bg-court-surface/50 rounded-lg p-1">
              {(['overview', 'tactical', 'technical', 'fitness'] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`
                    px-4 py-1.5 rounded-md font-mono text-xs tracking-wider uppercase transition-all duration-200
                    ${activeTab === tab
                      ? 'bg-shuttle-lime text-court-dark'
                      : 'text-text-muted hover:text-text-primary'
                    }
                  `}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>
        </div>
      </header>

      {/* Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'overview' && (
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 animate-entrance">
            {/* Video Player - 8 cols */}
            <div className="lg:col-span-8">
              <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
                <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                  <h2 className="font-display text-lg text-text-primary tracking-wide">MATCH FOOTAGE</h2>
                  <span className="font-mono text-[10px] text-text-muted">{rallies.length} RALLIES</span>
                </div>
                <div className="p-4">
                  <VideoPlayer jobId={jobId} />
                </div>
              </div>
            </div>

            {/* Shot Distribution - 4 cols */}
            <div className="lg:col-span-4">
              <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden h-full">
                <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                  <h2 className="font-display text-lg text-text-primary tracking-wide">SHOT DISTRIBUTION</h2>
                  <span className="font-mono text-[10px] text-shuttle-lime">{playerLabel(selectedPlayer)}</span>
                </div>
                <div className="p-5">
                  <ShotChart distribution={shotDistribution} />
                </div>
              </div>
            </div>

            {/* Coach Recommendations - full width */}
            <div className="lg:col-span-12">
              <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
                <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                  <h2 className="font-display text-lg text-text-primary tracking-wide">COURT HEATMAP</h2>
                  <span className="font-mono text-[10px] text-shuttle-lime">{playerLabel(selectedPlayer)}</span>
                </div>
                <div className="p-4">
                  <CourtHeatmap
                    zoneTransitions={report.court_analytics?.zone_transitions || []}
                    courtDimensions={report.court_analytics?.court_dimensions || { length: 13.4, width: 5.18 }}
                    selectedPlayer={selectedPlayer}
                  />
                </div>
              </div>
            </div>

            {/* Coach Recommendations - full width */}
            <div className="lg:col-span-12">
              <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
                <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                  <h2 className="font-display text-lg text-text-primary tracking-wide">COACH RECOMMENDATIONS</h2>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-shuttle-lime pulse-live" />
                    <span className="font-mono text-[10px] text-text-muted">AI-GENERATED</span>
                  </div>
                </div>
                <div className="p-6">
                  <CoachPanel
                    strengths={report.strengths || []}
                    weaknesses={report.weaknesses || []}
                    improvements={report.top_3_improvements || []}
                    drills={report.recommended_drills || []}
                    evidence={report.evidence || []}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'tactical' && (
          <div className="animate-entrance">
            <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
              <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                <h2 className="font-display text-lg text-text-primary tracking-wide">TACTICAL ANALYSIS</h2>
                <span className="font-mono text-[10px] text-shuttle-lime">{playerLabel(selectedPlayer)}</span>
              </div>
              <div className="p-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <h3 className="font-display text-sm text-text-muted tracking-widest mb-4">SHOT DISTRIBUTION</h3>
                    <ShotChart distribution={shotDistribution} />
                  </div>
                  <div>
                    <h3 className="font-display text-sm text-text-muted tracking-widest mb-4">COMMON PATTERNS</h3>
                    <div className="space-y-2">
                      {commonPatterns.length > 0 ? commonPatterns.map((p: any, i: number) => (
                        <div key={i} className="flex items-center justify-between px-4 py-2.5 rounded-lg bg-court-dark/40">
                          <span className="font-mono text-xs text-text-secondary">{p.pattern}</span>
                          <span className="font-mono text-xs text-shuttle-lime">{p.count}×</span>
                        </div>
                      )) : (
                        <p className="font-mono text-xs text-text-muted">No pattern data available</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'technical' && (
          <div className="animate-entrance">
            <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
              <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                <h2 className="font-display text-lg text-text-primary tracking-wide">TECHNICAL ASSESSMENT</h2>
                <span className="font-mono text-[10px] text-shuttle-lime">{playerLabel(selectedPlayer)}</span>
              </div>
              <div className="p-6">
                {technicalAssessments ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                    {Object.entries(technicalAssessments).map(([stroke, data]: [string, any]) => (
                      <div key={stroke} className="px-4 py-3 rounded-xl bg-court-dark/40 border border-court-line/10">
                        <div className="flex items-center justify-between mb-2">
                          <span className="font-body text-sm text-text-primary capitalize">
                            {stroke.replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono text-xs text-text-muted">{data.shot_count}×</span>
                        </div>
                        <div className="w-full h-1.5 bg-court-surface rounded-full overflow-hidden">
                          <div
                            className="h-full bg-gradient-to-r from-shuttle-lime to-feather-green rounded-full"
                            style={{ width: `${(data.avg_score || 0) * 100}%` }}
                          />
                        </div>
                        <p className="font-mono text-[10px] text-text-muted mt-1.5">
                          Score: {(data.avg_score || 0).toFixed(2)}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-text-muted font-mono text-sm">No technical data available for this player</p>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'fitness' && (
          <div className="animate-entrance">
            <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
              <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
                <h2 className="font-display text-lg text-text-primary tracking-wide">FITNESS & FOOTWORK</h2>
                <span className="font-mono text-[10px] text-shuttle-lime">{playerLabel(selectedPlayer)}</span>
              </div>
              <div className="p-6">
                <p className="font-mono text-xs text-text-muted">Coming soon — fatigue trend chart and court coverage stats</p>
              </div>
            </div>
          </div>
        )}

        {/* Rally Breakdown - always visible at bottom */}
        {rallies.length > 0 && activeTab === 'overview' && (
          <div className="mt-6 bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden animate-entrance" style={{ animationDelay: '200ms' }}>
            <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
              <h2 className="font-display text-lg text-text-primary tracking-wide">RALLY BREAKDOWN</h2>
              <span className="font-mono text-[10px] text-text-muted">{rallies.length} RALLIES</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-court-line/10">
                    <th className="text-left px-5 py-3 font-mono text-[10px] text-text-muted tracking-widest">#</th>
                    <th className="text-left px-5 py-3 font-mono text-[10px] text-text-muted tracking-widest">START</th>
                    <th className="text-left px-5 py-3 font-mono text-[10px] text-text-muted tracking-widest">END</th>
                    <th className="text-left px-5 py-3 font-mono text-[10px] text-text-muted tracking-widest">SHOTS</th>
                  </tr>
                </thead>
                <tbody>
                  {rallies.map((r: any) => (
                    <tr key={r.rally_id} className="border-b border-court-line/5 hover:bg-court-surface/20 transition-colors">
                      <td className="px-5 py-3 font-mono text-xs text-shuttle-lime">{r.rally_id}</td>
                      <td className="px-5 py-3 font-mono text-xs text-text-secondary">{r.start_frame}</td>
                      <td className="px-5 py-3 font-mono text-xs text-text-secondary">{r.end_frame}</td>
                      <td className="px-5 py-3">
                        <span className="font-mono text-xs bg-court-surface/50 px-2 py-0.5 rounded-full text-text-primary">
                          {r.shot_count}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
