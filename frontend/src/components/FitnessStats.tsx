interface FitnessStatsProps {
  fitness: {
    rally_intensity?: number;
    fatigue_trend?: string;
    total_distance?: number;
    avg_recovery?: number;
  };
  footwork: {
    distance_covered?: number;
    avg_recovery?: number;
    recovery_times?: number[];
  };
}

const TREND_COLORS: Record<string, string> = {
  improving: '#00e676',
  stable: '#c8ff00',
  declining: '#ff5252',
  insufficient_data: '#556677',
  unknown: '#556677',
};

const TREND_LABELS: Record<string, string> = {
  improving: 'IMPROVING',
  stable: 'STABLE',
  declining: 'DECLINING',
  insufficient_data: 'INSUFFICIENT DATA',
  unknown: 'UNKNOWN',
};

export function FitnessStats({ fitness, footwork }: FitnessStatsProps) {
  const trend = fitness.fatigue_trend || 'unknown';
  const trendColor = TREND_COLORS[trend] || '#556677';
  const trendLabel = TREND_LABELS[trend] || trend.toUpperCase();

  const stats = [
    {
      label: 'AVG RALLY INTENSITY',
      value: fitness.rally_intensity != null ? `${fitness.rally_intensity.toFixed(2)} s/s` : '—',
    },
    {
      label: 'TOTAL DISTANCE',
      value: (fitness.total_distance || footwork.distance_covered) != null
        ? `${(fitness.total_distance || footwork.distance_covered || 0).toFixed(0)} px`
        : '—',
    },
    {
      label: 'AVG RECOVERY',
      value: (fitness.avg_recovery || footwork.avg_recovery) != null
        ? `${(fitness.avg_recovery || footwork.avg_recovery || 0).toFixed(2)} s`
        : '—',
    },
    {
      label: 'FATIGUE TREND',
      value: trendLabel,
      color: trendColor,
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      {stats.map(s => (
        <div key={s.label} className="px-4 py-3 rounded-xl bg-court-dark/40 border border-court-line/10">
          <p className="font-mono text-[10px] text-text-muted tracking-widest mb-1">{s.label}</p>
          <p className="font-mono text-sm" style={{ color: s.color || '#e8edf2' }}>{s.value}</p>
        </div>
      ))}
    </div>
  );
}
