import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';

interface ShotChartProps {
  distribution: Record<string, number>;
}

const COLORS = [
  '#c8ff00', '#00e676', '#ffab00', '#e040fb',
  '#40c4ff', '#ff5252', '#69f0ae', '#ffd740',
];

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null;
  const { name, value } = payload[0];
  return (
    <div className="bg-court-dark/95 backdrop-blur-sm border border-court-line/30 rounded-lg px-4 py-2.5 shadow-xl">
      <p className="font-body text-sm text-text-primary">{name}</p>
      <p className="font-mono text-lg text-shuttle-lime font-semibold">{value}%</p>
    </div>
  );
};

export function ShotChart({ distribution }: ShotChartProps) {
  const data = Object.entries(distribution)
    .map(([name, value]) => ({
      name: name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()),
      value: Math.round(value * 100),
    }))
    .filter(d => d.value > 0)
    .sort((a, b) => b.value - a.value);

  if (data.length === 0) {
    return (
      <div className="h-[300px] flex items-center justify-center text-text-muted font-mono text-sm">
        No shot data available
      </div>
    );
  }

  return (
    <div className="relative">
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            cx="50%"
            cy="50%"
            innerRadius={60}
            outerRadius={100}
            paddingAngle={3}
            dataKey="value"
            stroke="none"
          >
            {data.map((_, i) => (
              <Cell key={i} fill={COLORS[i % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
        </PieChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-4 grid grid-cols-2 gap-2">
        {data.map((entry, i) => (
          <div key={entry.name} className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-court-dark/40">
            <div className="w-3 h-3 rounded-sm flex-shrink-0" style={{ backgroundColor: COLORS[i % COLORS.length] }} />
            <span className="font-body text-xs text-text-secondary truncate">{entry.name}</span>
            <span className="font-mono text-xs text-text-primary ml-auto">{entry.value}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
