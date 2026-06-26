import { useRef, useEffect } from 'react';

interface ZoneTransition {
  frame: number;
  zone: string;
  player_id: string;
}

interface CourtDimensions {
  length: number;
  width: number;
}

interface CourtHeatmapProps {
  zoneTransitions: ZoneTransition[];
  courtDimensions: CourtDimensions;
  selectedPlayer?: string;
}

const ZONE_POSITIONS: Record<string, [number, number]> = {
  front_left: [0.17, 0.17],
  front_center: [0.5, 0.17],
  front_right: [0.83, 0.17],
  mid_left: [0.17, 0.5],
  mid_center: [0.5, 0.5],
  mid_right: [0.83, 0.5],
  rear_left: [0.17, 0.83],
  rear_center: [0.5, 0.83],
  rear_right: [0.83, 0.83],
};

const PLAYER_COLORS: Record<string, string> = {
  player_1: '#c8ff00',
  player_2: '#e040fb',
};

export function CourtHeatmap({ zoneTransitions, courtDimensions, selectedPlayer }: CourtHeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    const pad = 30;

    ctx.clearRect(0, 0, w, h);

    // Court outline
    const courtX = pad;
    const courtY = pad;
    const courtW = w - pad * 2;
    const courtH = h - pad * 2;

    ctx.strokeStyle = '#3d4f63';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(courtX, courtY, courtW, courtH);

    // Net line (horizontal center)
    ctx.beginPath();
    ctx.moveTo(courtX, courtY + courtH / 2);
    ctx.lineTo(courtX + courtW, courtY + courtH / 2);
    ctx.strokeStyle = '#556677';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.stroke();
    ctx.setLineDash([]);

    // 3x3 grid lines
    ctx.strokeStyle = '#3d4f6380';
    ctx.lineWidth = 0.5;
    for (let i = 1; i < 3; i++) {
      // Vertical
      ctx.beginPath();
      ctx.moveTo(courtX + (courtW * i) / 3, courtY);
      ctx.lineTo(courtX + (courtW * i) / 3, courtY + courtH);
      ctx.stroke();
      // Horizontal
      ctx.beginPath();
      ctx.moveTo(courtX, courtY + (courtH * i) / 3);
      ctx.lineTo(courtX + courtW, courtY + (courtH * i) / 3);
      ctx.stroke();
    }

    // Zone labels
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.fillStyle = '#556677';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (const [zone, [nx, ny]] of Object.entries(ZONE_POSITIONS)) {
      const label = zone.replace(/_/g, ' ').replace('front', 'F').replace('mid', 'M').replace('rear', 'R').replace('left', 'L').replace('center', 'C').replace('right', 'R');
      ctx.fillText(label, courtX + courtW * nx, courtY + courtH * ny);
    }

    // Count zone visits per player
    const zoneCounts: Record<string, Record<string, number>> = {};
    for (const t of zoneTransitions) {
      if (selectedPlayer && t.player_id !== selectedPlayer) continue;
      if (!zoneCounts[t.player_id]) zoneCounts[t.player_id] = {};
      zoneCounts[t.player_id][t.zone] = (zoneCounts[t.player_id][t.zone] || 0) + 1;
    }

    const maxCount = Math.max(1, ...Object.values(zoneCounts).flatMap(pc => Object.values(pc)));

    // Draw heatmap dots
    for (const [playerId, zones] of Object.entries(zoneCounts)) {
      const color = PLAYER_COLORS[playerId] || '#c8ff00';
      for (const [zone, count] of Object.entries(zones)) {
        const pos = ZONE_POSITIONS[zone];
        if (!pos) continue;
        const [nx, ny] = pos;
        const radius = 8 + (count / maxCount) * 20;
        const alpha = 0.2 + (count / maxCount) * 0.6;

        ctx.beginPath();
        ctx.arc(courtX + courtW * nx, courtY + courtH * ny, radius, 0, Math.PI * 2);
        ctx.fillStyle = color + Math.round(alpha * 255).toString(16).padStart(2, '0');
        ctx.fill();

        // Count label
        if (count > 0) {
          ctx.fillStyle = '#e8edf2';
          ctx.font = 'bold 10px "JetBrains Mono", monospace';
          ctx.fillText(String(count), courtX + courtW * nx, courtY + courtH * ny);
        }
      }
    }

    // Legend
    const legendY = h - 16;
    let legendX = pad;
    ctx.font = '10px "JetBrains Mono", monospace';
    const playersToShow = selectedPlayer ? [selectedPlayer] : Object.keys(zoneCounts);
    for (const pid of playersToShow) {
      const color = PLAYER_COLORS[pid] || '#c8ff00';
      ctx.fillStyle = color;
      ctx.fillRect(legendX, legendY - 5, 10, 10);
      ctx.fillStyle = '#8899aa';
      ctx.textAlign = 'left';
      ctx.fillText(pid === 'player_1' ? 'Near' : pid === 'player_2' ? 'Far' : pid, legendX + 14, legendY + 1);
      legendX += 80;
    }
  }, [zoneTransitions, courtDimensions, selectedPlayer]);

  return (
    <canvas
      ref={canvasRef}
      className="w-full"
      style={{ height: '320px' }}
    />
  );
}
