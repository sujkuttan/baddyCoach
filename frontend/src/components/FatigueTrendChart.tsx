import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

interface FatigueTrendChartProps {
  rallyIntensities: number[];
}

export function FatigueTrendChart({ rallyIntensities }: FatigueTrendChartProps) {
  const data = rallyIntensities.map((intensity, i) => ({
    rally: i + 1,
    intensity: parseFloat(intensity.toFixed(2)),
  }));

  const avg = rallyIntensities.length > 0
    ? rallyIntensities.reduce((a, b) => a + b, 0) / rallyIntensities.length
    : 0;

  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48">
        <p className="font-mono text-xs text-text-muted">No rally intensity data available</p>
      </div>
    );
  }

  return (
    <div className="w-full h-64">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#3d4f6340" />
          <XAxis
            dataKey="rally"
            tick={{ fill: '#556677', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={{ stroke: '#3d4f63' }}
            tickLine={false}
          />
          <YAxis
            tick={{ fill: '#556677', fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}
            axisLine={{ stroke: '#3d4f63' }}
            tickLine={false}
            label={{ value: 'shots/sec', angle: -90, position: 'insideLeft', fill: '#556677', fontSize: 9 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a2028',
              border: '1px solid #3d4f63',
              borderRadius: '8px',
              fontFamily: 'JetBrains Mono, monospace',
              fontSize: '11px',
            }}
            labelFormatter={(v) => `Rally ${v}`}
            formatter={(v: number) => [v.toFixed(2), 'Intensity']}
          />
          <ReferenceLine
            y={avg}
            stroke="#c8ff0060"
            strokeDasharray="4 4"
            label={{ value: `avg: ${avg.toFixed(2)}`, fill: '#c8ff00', fontSize: 9, fontFamily: 'JetBrains Mono, monospace' }}
          />
          <Line
            type="monotone"
            dataKey="intensity"
            stroke="#c8ff00"
            strokeWidth={2}
            dot={{ fill: '#c8ff00', r: 3 }}
            activeDot={{ fill: '#c8ff00', r: 5, stroke: '#0f1419', strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
