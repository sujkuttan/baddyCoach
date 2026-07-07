import { useRef, useCallback, useMemo } from 'react';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  Radar, ResponsiveContainer, Tooltip,
} from 'recharts';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyDict = Record<string, any>;

interface Props {
  technical: AnyDict | null;
  footwork: AnyDict | null;
  fitness: AnyDict | null;
  symmetryScore: number;
  playerLabel: string;
}

const DIM_LABELS: Record<string, string> = {
  arm_power: 'Arm Power',
  footwork: 'Footwork',
  core_stability: 'Core Stability',
  strike_precision: 'Precision',
  coordination: 'Coordination',
  consistency: 'Consistency',
};

function clamp(v: number): number {
  return Math.max(0, Math.min(100, Math.round(v)));
}

function computeDimensions(technical: AnyDict | null, footwork: AnyDict | null, fitness: AnyDict | null, symmetryScore: number): Record<string, number> {

  // arm_power: from technical stroke scores — avg max elbow extension
  let armPower = 50;
  if (technical) {
    const scores: number[] = [];
    for (const data of Object.values(technical)) {
      const d = data as AnyDict;
      if (d.avg_score != null) scores.push(d.avg_score * 100);
    }
    if (scores.length > 0) armPower = scores.reduce((a, b) => a + b, 0) / scores.length;
  }

  // footwork: from footwork distance and recovery
  let footworkScore = 50;
  if (footwork) {
    const dist = (footwork as AnyDict).distance_covered ?? 0;
    const recovery = (footwork as AnyDict).avg_recovery ?? 1;
    const distScore = Math.min(100, (dist / 50) * 100);
    const recoveryScore = Math.max(0, 100 - (recovery / 2) * 100);
    footworkScore = (distScore + recoveryScore) / 2;
  }

  // core_stability: from fitness fatigue_trend and intensity consistency
  let coreScore = 50;
  if (fitness) {
    const f = fitness as AnyDict;
    const trend = f.fatigue_trend === 'stable' ? 80 : f.fatigue_trend === 'improving' ? 90 : 40;
    const peak = f.peak_intensity ?? 1;
    const peakScore = Math.min(100, (peak / 3) * 100);
    coreScore = (trend + peakScore) / 2;
  }

  // strike_precision: from per-stroke technique score consistency
  let precision = 50;
  if (technical) {
    const highs: number[] = [];
    for (const data of Object.values(technical)) {
      const d = data as AnyDict;
      if (d.avg_score != null && d.avg_score > 0.5) highs.push(d.avg_score * 100);
    }
    if (highs.length > 0) precision = highs.reduce((a, b) => a + b, 0) / highs.length;
  }

  // coordination: from symmetry score (0-100)
  const coordination = symmetryScore;

  // consistency: inverse of score variance across stroke types
  let consistency = 50;
  if (technical) {
    const scores: number[] = [];
    for (const data of Object.values(technical)) {
      const d = data as any;
      if (d.avg_score != null) scores.push(d.avg_score);
    }
    if (scores.length > 1) {
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
      const variance = scores.reduce((sum, s) => sum + (s - mean) ** 2, 0) / scores.length;
      consistency = Math.max(0, 100 - Math.sqrt(variance) * 100);
    }
  }

  return {
    arm_power: clamp(armPower),
    footwork: clamp(footworkScore),
    core_stability: clamp(coreScore),
    strike_precision: clamp(precision),
    coordination: clamp(coordination),
    consistency: clamp(consistency),
  };
}

export function PerformanceRadar({ technical, footwork, fitness, symmetryScore, playerLabel }: Props) {
  const cardRef = useRef<HTMLDivElement>(null);
  const dims = useMemo(() => computeDimensions(technical, footwork, fitness, symmetryScore), [technical, footwork, fitness, symmetryScore]);

  const chartData = useMemo(() =>
    Object.entries(DIM_LABELS).map(([key, label]) => ({
      dimension: label,
      score: dims[key] ?? 50,
      fill: '#f59e0b',
    })),
  [dims]);

  const overallScore = useMemo(() =>
    Math.round(Object.values(dims).reduce((a, b) => a + b, 0) / Object.keys(dims).length),
  [dims]);

  const saveCard = useCallback(() => {
    const svgEl = cardRef.current?.querySelector('svg.recharts-surface');
    if (!svgEl) return;

    const svgClone = svgEl.cloneNode(true) as SVGSVGElement;
    const rect = svgEl.getBoundingClientRect();
    const W = rect.width * 2;
    const H = rect.height * 2;

    // Wrap in a card background + labels
    const canvas = document.createElement('canvas');
    canvas.width = W;
    canvas.height = H + 160;
    const ctx = canvas.getContext('2d')!;

    // Background
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Score
    ctx.fillStyle = '#f59e0b';
    ctx.font = 'bold 64px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(String(overallScore), canvas.width / 2, 72);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '14px sans-serif';
    ctx.fillText('OVERALL SCORE', canvas.width / 2, 94);

    // Draw radar SVG onto canvas
    const serializer = new XMLSerializer();
    const svgStr = serializer.serializeToString(svgClone);
    const img = new Image();
    const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
    const url = URL.createObjectURL(blob);

    img.onload = () => {
      ctx.drawImage(img, 0, 110, W, H);
      URL.revokeObjectURL(url);

      // Dimension labels below radar
      ctx.fillStyle = '#94a3b8';
      ctx.font = 'bold 12px sans-serif';
      let x = 40;
      const y = H + 130;
      for (const [key, label] of Object.entries(DIM_LABELS)) {
        ctx.textAlign = 'left';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(label, x, y - 4);
        ctx.fillStyle = '#818cf8';
        ctx.font = 'bold 16px sans-serif';
        ctx.fillText(String(dims[key] ?? 50), x + 90, y - 4);
        ctx.font = 'bold 12px sans-serif';
        x += 160;
      }

      // Footer
      ctx.fillStyle = '#475569';
      ctx.font = '10px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(`Badminton Coach AI · ${playerLabel}`, canvas.width / 2, H + 152);

      // Download
      const link = document.createElement('a');
      link.download = `badminton-radar-${playerLabel.toLowerCase()}.png`;
      link.href = canvas.toDataURL('image/png');
      link.click();
    };
    img.src = url;
  }, [dims, overallScore, playerLabel]);

  return (
    <div className="bg-court-mid/60 backdrop-blur-sm rounded-2xl border border-court-line/15 overflow-hidden">
      <div className="px-5 py-3 border-b border-court-line/10 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="font-display text-lg text-text-primary tracking-wide">PERFORMANCE RADAR</h2>
          <div className="flex items-center gap-1.5 bg-court-surface/50 px-3 py-1 rounded-lg">
            <span className="font-mono text-2xl font-bold text-shuttle-lime">{overallScore}</span>
            <span className="font-mono text-[10px] text-text-muted">/100</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10px] text-shuttle-lime">{playerLabel}</span>
          <button
            onClick={saveCard}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-shuttle-lime/20 hover:bg-shuttle-lime/30 text-shuttle-lime transition-colors"
          >
            <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
            </svg>
            <span className="font-mono text-[10px] tracking-wider">SAVE CARD</span>
          </button>
        </div>
      </div>
      <div ref={cardRef} className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          <div>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={chartData} cx="50%" cy="50%" outerRadius="80%">
                <PolarGrid stroke="#334155" />
                <PolarAngleAxis dataKey="dimension" tick={{ fill: '#94a3b8', fontSize: 11, fontWeight: 500 }} />
                <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: '#475569', fontSize: 10 }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
                  itemStyle={{ color: '#f59e0b' }}
                />
                <Radar
                  name="Score"
                  dataKey="score"
                  stroke="#818cf8"
                  fill="#6366f1"
                  fillOpacity={0.25}
                  strokeWidth={2}
                  dot={{ fill: '#f59e0b', r: 4 }}
                  activeDot={{ r: 6 }}
                />
              </RadarChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-3">
            {Object.entries(DIM_LABELS).map(([key, label]) => {
              const val = dims[key] ?? 50;
              const pct = (val / 100) * 100;
              return (
                <div key={key}>
                  <div className="flex justify-between items-center mb-1">
                    <span className="font-mono text-xs text-text-muted">{label}</span>
                    <span className="font-mono text-xs text-shuttle-lime font-bold">{val}</span>
                  </div>
                  <div className="w-full h-1.5 bg-court-surface rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-500 to-shuttle-lime rounded-full transition-all duration-700"
                      style={{ width: `${pct}%` }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}
