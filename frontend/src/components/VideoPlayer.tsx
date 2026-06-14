import { useEffect, useRef } from 'react';
import videojs from 'video.js';
import 'video.js/dist/video-js.css';

interface VideoPlayerProps {
  jobId: string;
}

export function VideoPlayer({ jobId }: VideoPlayerProps) {
  const videoRef = useRef<HTMLDivElement>(null);
  const playerRef = useRef<any>(null);

  useEffect(() => {
    if (!videoRef.current) return;

    const videoElement = document.createElement('video-js');
    videoElement.classList.add('vjs-big-play-centered', 'vjs-theme-city');
    videoRef.current.appendChild(videoElement);

    playerRef.current = videojs(videoElement, {
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

    return () => {
      playerRef.current?.dispose();
    };
  }, [jobId]);

  return (
    <div data-vjs-player ref={videoRef} className="rounded-xl overflow-hidden border border-court-line/20" />
  );
}
