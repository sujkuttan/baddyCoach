import { useEffect, useRef, useState, useCallback } from 'react';
import videojs from 'video.js';
import 'video.js/dist/video-js.css';

interface Rally {
  rally_id: number;
  start_frame: number;
  end_frame: number;
  shot_count: number;
}

interface VideoPlayerProps {
  jobId: string;
  rallies?: Rally[];
  fps?: number;
}

const RALLY_COLORS = [
  '#c8ff00', '#00e676', '#ffab00', '#e040fb', '#448aff',
  '#ff5252', '#69f0ae', '#ffd740', '#ea80fc', '#82b1ff',
];

export function VideoPlayer({ jobId, rallies = [], fps = 30 }: VideoPlayerProps) {
  const videoRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<any>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);

  useEffect(() => {
    if (!videoRef.current) return;

    const videoElement = document.createElement('video-js');
    videoElement.classList.add('vjs-big-play-centered', 'vjs-theme-city');
    videoRef.current.appendChild(videoElement);

    const player = videojs(videoElement, {
      controls: true,
      fluid: true,
      responsive: true,
      sources: [{ src: `/api/jobs/${jobId}/video`, type: 'video/mp4' }],
      playbackRates: [0.25, 0.5, 1, 1.5, 2],
      controlBar: {
        volumePanel: { inline: true },
        pictureInPictureToggle: true,
      },
    });

    playerRef.current = player;

    player.on('loadedmetadata', () => {
      setDuration(player.duration());
    });

    player.on('timeupdate', () => {
      setCurrentTime(player.currentTime());
    });

    return () => {
      player.dispose();
    };
  }, [jobId]);

  const seekToFrame = useCallback((frame: number) => {
    const player = playerRef.current;
    if (!player) return;
    player.currentTime(frame / fps);
  }, [fps]);

  if (rallies.length === 0 || duration === 0) {
    return (
      <div data-vjs-player ref={videoRef} className="rounded-xl overflow-hidden border border-court-line/20" />
    );
  }

  return (
    <div className="space-y-3">
      <div data-vjs-player ref={videoRef} className="rounded-xl overflow-hidden border border-court-line/20" />

      {/* Rally Timeline */}
      <div className="bg-court-dark/60 rounded-xl border border-court-line/15 p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="font-mono text-[10px] text-text-muted tracking-widest">RALLY TIMELINE</span>
          <span className="font-mono text-[10px] text-text-muted">
            {Math.floor(currentTime)}s / {Math.floor(duration)}s
          </span>
        </div>

        {/* Timeline bar */}
        <div className="relative h-8 bg-court-surface/30 rounded-lg overflow-hidden cursor-pointer">
          {/* Rally segments */}
          {rallies.map((rally, i) => {
            const start = (rally.start_frame / fps) / duration;
            const end = (rally.end_frame / fps) / duration;
            const width = Math.max(end - start, 0.005);
            return (
              <div
                key={rally.rally_id}
                className="absolute top-0 h-full opacity-60 hover:opacity-100 transition-opacity cursor-pointer group"
                style={{
                  left: `${start * 100}%`,
                  width: `${width * 100}%`,
                  backgroundColor: RALLY_COLORS[i % RALLY_COLORS.length],
                }}
                onClick={() => seekToFrame(rally.start_frame)}
              >
                {/* Tooltip */}
                <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block whitespace-nowrap bg-court-dark border border-court-line/30 rounded px-2 py-1 text-[9px] font-mono text-text-primary z-10">
                  Rally {rally.rally_id} · {rally.shot_count} shots
                </div>
              </div>
            );
          })}

          {/* Playhead */}
          <div
            className="absolute top-0 h-full w-0.5 bg-white/80 z-10 pointer-events-none"
            style={{ left: `${(currentTime / duration) * 100}%` }}
          />
        </div>

        {/* Rally legend */}
        <div className="flex flex-wrap gap-2 mt-2">
          {rallies.slice(0, 8).map((rally, i) => (
            <button
              key={rally.rally_id}
              onClick={() => seekToFrame(rally.start_frame)}
              className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-court-surface/30 hover:bg-court-surface/50 transition-colors"
            >
              <div
                className="w-2 h-2 rounded-full"
                style={{ backgroundColor: RALLY_COLORS[i % RALLY_COLORS.length] }}
              />
              <span className="font-mono text-[9px] text-text-muted">R{rally.rally_id}</span>
            </button>
          ))}
          {rallies.length > 8 && (
            <span className="font-mono text-[9px] text-text-muted self-center">+{rallies.length - 8} more</span>
          )}
        </div>
      </div>
    </div>
  );
}
