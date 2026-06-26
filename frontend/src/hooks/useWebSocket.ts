import { useEffect, useRef, useState, useCallback } from 'react';

export interface ProgressEvent {
  stage: string;
  status: 'running' | 'complete' | 'failed';
  metadata?: Record<string, any>;
  error?: string;
}

export function useWebSocket(jobId: string | null) {
  const [events, setEvents] = useState<ProgressEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!jobId) return;

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${protocol}//${window.location.host}/api/jobs/${jobId}/progress`);
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => setConnected(false);
    ws.onmessage = (e) => {
      const data = JSON.parse(e.data);
      if (data.type !== 'pong') {
        setEvents(prev => [...prev, data]);
      }
    };

    return () => ws.close();
  }, [jobId]);

  const sendPing = useCallback(() => {
    wsRef.current?.send('ping');
  }, []);

  return { events, connected, sendPing };
}
