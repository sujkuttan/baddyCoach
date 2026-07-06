import { useEffect, useRef, useState, useCallback, forwardRef, useImperativeHandle } from 'react';
import { StrokeTimeline } from './StrokeTimeline';

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

interface VideoPlayerProps {
  jobId?: string | null;
  videoUrl?: string | null;
  rallies?: Rally[];
  strokes?: Stroke[];
  fps?: number;
}

const RALLY_COLORS = [
  '#c8ff00', '#00e676', '#ffab00', '#e040fb', '#448aff',
  '#ff5252', '#69f0ae', '#ffd740', '#ea80fc', '#82b1ff',
];

export interface VideoPlayerHandle {
  seekTo: (time: number) => void;
  playSegment: (start: number, end: number, loop?: boolean) => void;
  getCurrentTime: () => number;
}

export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(function VideoPlayer({ jobId, videoUrl, rallies = [], strokes = [], fps = 30 }, ref) {
  const videoElRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [videoError, setVideoError] = useState<string | null>(null);
  const src = videoUrl || (jobId ? `/api/jobs/${jobId}/video` : null);

  useEffect(() => {
    const el = videoElRef.current;
    if (!el || !src) return;

    setVideoError(null);

    const onLoaded = () => {
      setDuration(el.duration);
    };
    const onTimeUpdate = () => {
      setCurrentTime(el.currentTime);
    };
    const onError = () => {
      const err = el.error;
      const codes: Record<number, string> = {
        1: 'ABORTED',
        2: 'NETWORK',
        3: 'DECODE',
        4: 'SRC_NOT_SUPPORTED',
      };
      setVideoError(`Video error: ${codes[err?.code || 0] || 'UNKNOWN'} (code ${err?.code}), message: ${err?.message || 'none'}`);
    };

    el.addEventListener('loadedmetadata', onLoaded);
    el.addEventListener('timeupdate', onTimeUpdate);
    el.addEventListener('error', onError);

    return () => {
      el.removeEventListener('loadedmetadata', onLoaded);
      el.removeEventListener('timeupdate', onTimeUpdate);
      el.removeEventListener('error', onError);
    };
  }, [src]);

  const getCurrentTime = useCallback(() => {
    return videoElRef.current?.currentTime ?? 0;
  }, []);

  useImperativeHandle(ref, () => ({
    seekTo: (time: number) => {
      const el = videoElRef.current;
      if (el) el.currentTime = time;
    },
    playSegment: (start: number, end: number, loop: boolean = true) => {
      const el = videoElRef.current;
      if (!el) return;
      el.currentTime = start;
      el.play();
      const onTime = () => {
        if (el.currentTime >= end) {
          if (loop) {
            el.currentTime = start;
          } else {
            el.pause();
            el.removeEventListener('timeupdate', onTime);
          }
        }
      };
      el.addEventListener('timeupdate', onTime);
    },
    getCurrentTime,
  }), [getCurrentTime]);

  const seekToFrame = useCallback((frame: number) => {
    const el = videoElRef.current;
    if (!el) return;
    el.currentTime = frame / fps;
  }, [fps]);

  return (
    <div className="space-y-3">
      <div className="rounded-xl overflow-hidden border border-court-line/20 bg-black">
        {src ? (
          <video
            ref={videoElRef}
            src={src}
            controls
            preload="auto"
            className="w-full"
            style={{ maxHeight: '70vh', objectFit: 'contain' }}
          />
        ) : (
          <div className="aspect-video flex items-center justify-center bg-court-dark/40">
            <p className="font-mono text-sm text-text-muted">No video source</p>
          </div>
        )}
        {videoError && (
          <div className="px-3 py-2 bg-error-red/10 border-t border-error-red/20">
            <p className="font-mono text-[10px] text-error-red">{videoError}</p>
          </div>
        )}
      </div>

      {rallies.length > 0 && duration > 0 && (
        <>
          {/* Rally Timeline */}
          <div className="bg-court-dark/60 rounded-xl border border-court-line/15 p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[10px] text-text-muted tracking-widest">RALLY TIMELINE</span>
              <span className="font-mono text-[10px] text-text-muted">
                {Math.floor(currentTime)}s / {Math.floor(duration)}s
              </span>
            </div>

            <div className="relative h-8 bg-court-surface/30 rounded-lg overflow-hidden cursor-pointer">
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
                    <div className="absolute -top-8 left-1/2 -translate-x-1/2 hidden group-hover:block whitespace-nowrap bg-court-dark border border-court-line/30 rounded px-2 py-1 text-[9px] font-mono text-text-primary z-10">
                      Rally {rally.rally_id} · {rally.shot_count} shots
                    </div>
                  </div>
                );
              })}

              <div
                className="absolute top-0 h-full w-0.5 bg-white/80 z-10 pointer-events-none"
                style={{ left: `${(currentTime / duration) * 100}%` }}
              />
            </div>

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

          {/* Stroke Timeline */}
          {strokes.length > 0 && (
            <div className="bg-court-dark/60 rounded-xl border border-court-line/15 p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-[10px] text-text-muted tracking-widest">STROKE TIMELINE</span>
                <span className="font-mono text-[10px] text-text-muted">{strokes.length} strokes</span>
              </div>
              <StrokeTimeline
                strokes={strokes}
                rallies={rallies}
                fps={fps}
                duration={duration}
                currentTime={currentTime}
                onSeek={(time) => {
                  const el = videoElRef.current;
                  if (el) el.currentTime = time;
                }}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
});
