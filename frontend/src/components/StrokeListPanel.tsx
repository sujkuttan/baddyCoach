import { useState } from 'react';

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

interface StrokeListPanelProps {
  strokes: Stroke[];
  rallies: Rally[];
  fps: number;
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

const STROKE_LABELS: Record<string, string> = {
  smash: 'Smash',
  clear: 'Clear',
  drop: 'Drop',
  net_shot: 'Net Shot',
  lift: 'Lift',
  block: 'Block',
  drive: 'Drive',
  rush: 'Rush',
  short_serve: 'Short Serve',
  long_serve: 'Long Serve',
  serve: 'Serve',
  unknown: 'Unknown',
};

export function StrokeListPanel({ strokes, rallies: _rallies, fps: _fps, onSeek }: StrokeListPanelProps) {
  const [filter, setFilter] = useState<string>('all');

  const uniqueTypes = Array.from(new Set(strokes.map(s => s.stroke_type))).sort();
  const filtered = filter === 'all' ? strokes : strokes.filter(s => s.stroke_type === filter);

  // Group by rally
  const byRally = new Map<number | null, Stroke[]>();
  for (const s of filtered) {
    const key = s.rally_id;
    if (!byRally.has(key)) byRally.set(key, []);
    byRally.get(key)!.push(s);
  }

  return (
    <div style={{ width: 280, maxHeight: '100%', overflow: 'auto', borderLeft: '1px solid #333', padding: 12, fontSize: 13 }}>
      <div style={{ fontWeight: 600, marginBottom: 8, color: '#c8ff00' }}>Stroke Map ({filtered.length})</div>

      {/* Filter */}
      <select
        value={filter}
        onChange={(e) => setFilter(e.target.value)}
        style={{ width: '100%', marginBottom: 10, background: '#1a1a2e', color: '#fff', border: '1px solid #333', borderRadius: 4, padding: '4px 8px' }}
      >
        <option value="all">All strokes ({strokes.length})</option>
        {uniqueTypes.map(t => (
          <option key={t} value={t}>{STROKE_LABELS[t] || t} ({strokes.filter(s => s.stroke_type === t).length})</option>
        ))}
      </select>

      {/* Grouped by rally */}
      {Array.from(byRally.entries()).map(([rallyId, rallyStrokes]) => (
        <div key={rallyId ?? 'none'} style={{ marginBottom: 10 }}>
          <div style={{ color: '#888', fontSize: 11, marginBottom: 4 }}>
            {rallyId ? `Rally ${rallyId}` : 'Between rallies'}
          </div>
          {rallyStrokes.map((stroke, i) => (
            <div
              key={i}
              onClick={() => onSeek(stroke.timestamp)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                padding: '4px 6px',
                marginBottom: 2,
                borderRadius: 4,
                cursor: 'pointer',
                background: 'rgba(255,255,255,0.03)',
                transition: 'background 0.15s',
              }}
              onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.08)'; }}
              onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.03)'; }}
            >
              <div style={{ width: 4, height: 20, borderRadius: 2, backgroundColor: STROKE_COLORS[stroke.stroke_type] || '#666', flexShrink: 0 }} />
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontSize: 12, fontWeight: 500 }}>{STROKE_LABELS[stroke.stroke_type] || stroke.stroke_type}</div>
                <div style={{ fontSize: 10, color: '#888' }}>
                  {formatTime(stroke.timestamp)} · {(stroke.confidence * 100).toFixed(0)}% · {stroke.player_id === 'player_1' ? 'Near' : 'Far'}
                </div>
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
