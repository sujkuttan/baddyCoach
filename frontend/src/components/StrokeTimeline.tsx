import { useRef, useEffect, useCallback } from 'react';

interface Stroke {
  frame: number;
  timestamp: number;
  stroke_type: string;
  confidence: number;
  player_id: string;
  rally_id: number | null;
}

interface Rally {
  rally_id: number;
  start_frame: number;
  end_frame: number;
  shot_count: number;
}

interface StrokeTimelineProps {
  strokes: Stroke[];
  rallies: Rally[];
  fps: number;
  duration: number;
  currentTime: number;
  onSeek: (time: number) => void;
}

const STROKE_COLORS: Record<string, string> = {
  smash: '#ff5252',
  clear: '#448aff',
  drop: '#e040fb',
  net_shot: '#00e676',
  lift: '#ffab00',
  block: '#69f0ae',
  drive: '#ea80fc',
  rush: '#c8ff00',
  short_serve: '#82b1ff',
  long_serve: '#82b1ff',
  serve: '#82b1ff',
  unknown: '#666',
};

const RALLY_BG_COLORS = [
  'rgba(200,255,0,0.08)',
  'rgba(0,230,118,0.08)',
  'rgba(255,171,0,0.08)',
  'rgba(224,64,251,0.08)',
  'rgba(68,138,255,0.08)',
  'rgba(255,82,82,0.08)',
  'rgba(105,240,174,0.08)',
  'rgba(255,215,64,0.08)',
];

export function StrokeTimeline({ strokes, rallies, fps, duration, currentTime, onSeek }: StrokeTimelineProps) {
  const barRef = useRef<HTMLDivElement>(null);

  const handleClick = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!barRef.current) return;
    const rect = barRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const pct = x / rect.width;
    onSeek(pct * duration);
  }, [duration, onSeek]);

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <div className="stroke-timeline" style={{ position: 'relative', width: '100%', height: 64, marginTop: 8, cursor: 'pointer' }} onClick={handleClick} ref={barRef}>
      {/* Rally background bands */}
      {rallies.map((rally) => {
        const left = (rally.start_frame / fps / duration) * 100;
        const width = ((rally.end_frame - rally.start_frame) / fps / duration) * 100;
        return (
          <div
            key={`rally-${rally.rally_id}`}
            style={{
              position: 'absolute',
              left: `${left}%`,
              width: `${width}%`,
              top: 0,
              height: '100%',
              backgroundColor: RALLY_BG_COLORS[(rally.rally_id - 1) % RALLY_BG_COLORS.length],
              borderRadius: 4,
            }}
          />
        );
      })}

      {/* Stroke markers */}
      {strokes.map((stroke, i) => {
        const left = (stroke.timestamp / duration) * 100;
        const color = STROKE_COLORS[stroke.stroke_type] || '#666';
        return (
          <div
            key={`stroke-${i}`}
            title={`${stroke.stroke_type} (${(stroke.confidence * 100).toFixed(0)}%) @ ${formatTime(stroke.timestamp)}`}
            style={{
              position: 'absolute',
              left: `${left}%`,
              top: 8,
              width: 3,
              height: 48,
              backgroundColor: color,
              borderRadius: 2,
              transform: 'translateX(-50%)',
              transition: 'opacity 0.15s',
            }}
            onMouseEnter={(e) => { (e.target as HTMLElement).style.opacity = '1'; (e.target as HTMLElement).style.height = '56px'; }}
            onMouseLeave={(e) => { (e.target as HTMLElement).style.opacity = '0.85'; (e.target as HTMLElement).style.height = '48px'; }}
          />
        );
      })}

      {/* Playhead */}
      <div
        style={{
          position: 'absolute',
          left: `${progressPct}%`,
          top: 0,
          width: 2,
          height: '100%',
          backgroundColor: '#fff',
          transform: 'translateX(-50%)',
          pointerEvents: 'none',
        }}
      />
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
