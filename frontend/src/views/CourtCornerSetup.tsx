import { useState, useRef, useEffect, useCallback } from 'react';
import { getFrame, setCourtCorners, getCourtCorners, processVideo } from '../utils/api';

interface Props {
  jobId: string;
  poseModel: string;
  sampleRate: number;
  onComplete: () => void;
  onBack: () => void;
  existingCorners?: number[][] | null;
  redoMode?: boolean;
}

const CORNER_LABELS = ['Bottom-Left (BL)', 'Bottom-Right (BR)', 'Top-Left (TL)', 'Top-Right (TR)'];

export function CourtCornerSetup({ jobId, poseModel, sampleRate, onComplete, onBack, existingCorners, redoMode }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [corners, setCorners] = useState<{ x: number; y: number }[]>(() => {
    if (existingCorners && existingCorners.length === 4) {
      return existingCorners.map(c => ({ x: c[0], y: c[1] }));
    }
    return [];
  });
  const [step, setStep] = useState(() => {
    if (existingCorners && existingCorners.length === 4) return 4;
    return 0;
  });
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    // If existingCorners were passed, load them; otherwise try fetching from API
    if (existingCorners && existingCorners.length === 4) {
      getFrame(jobId).then(blob => {
        const url = URL.createObjectURL(blob);
        const image = new Image();
        image.onload = () => {
          setImg(image);
          setLoading(false);
        };
        image.src = url;
      }).catch(() => setLoading(false));
    } else {
      // Try fetching saved corners from API, then load frame
      getCourtCorners(jobId).then(data => {
        if (data.corners && data.corners.length === 4 && corners.length === 0) {
          const saved = data.corners.map(c => ({ x: c[0], y: c[1] }));
          setCorners(saved);
          setStep(4);
        }
      }).catch(() => {});

      getFrame(jobId).then(blob => {
        const url = URL.createObjectURL(blob);
        const image = new Image();
        image.onload = () => {
          setImg(image);
          setLoading(false);
        };
        image.src = url;
      }).catch(() => setLoading(false));
    }
  }, [jobId, existingCorners]);

  const getCanvasCoords = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return null;
    const rect = canvas.getBoundingClientRect();
    const scaleX = img.naturalWidth / canvas.width;
    const scaleY = img.naturalHeight / canvas.height;
    return {
      x: Math.round((e.clientX - rect.left) * scaleX),
      y: Math.round((e.clientY - rect.top) * scaleY),
    };
  }, [img]);

  const handleCanvasClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    if (step >= 4) return;
    const pt = getCanvasCoords(e);
    if (!pt) return;
    setCorners(prev => [...prev, pt]);
    setStep(s => s + 1);
  }, [step, getCanvasCoords]);

  const handleUndo = () => {
    if (corners.length === 0) return;
    setCorners(prev => prev.slice(0, -1));
    setStep(s => s - 1);
  };

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = canvas.clientWidth;
    canvas.height = canvas.clientWidth * (img.naturalHeight / img.naturalWidth);

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

    const scaleX = canvas.width / img.naturalWidth;
    const scaleY = canvas.height / img.naturalHeight;

    for (let i = 0; i < corners.length; i++) {
      const cx = corners[i].x * scaleX;
      const cy = corners[i].y * scaleY;

      ctx.beginPath();
      ctx.arc(cx, cy, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#c8ff00';
      ctx.fill();
      ctx.strokeStyle = '#1a2028';
      ctx.lineWidth = 3;
      ctx.stroke();

      ctx.fillStyle = '#c8ff00';
      ctx.font = 'bold 14px DM Sans, sans-serif';
      ctx.fillText(`${i + 1}`, cx + 14, cy + 5);
    }

    if (corners.length > 1) {
      ctx.beginPath();
      ctx.moveTo(corners[0].x * scaleX, corners[0].y * scaleY);
      for (let i = 1; i < corners.length; i++) {
        ctx.lineTo(corners[i].x * scaleX, corners[i].y * scaleY);
      }
      ctx.strokeStyle = '#c8ff00';
      ctx.lineWidth = 2;
      ctx.setLineDash([6, 4]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [img, corners]);

  useEffect(() => { draw(); }, [draw]);

  const handleSkip = async () => {
    if (redoMode) {
      onBack();
      return;
    }
    setSaving(true);
    try {
      await processVideo(jobId, poseModel, sampleRate);
      onComplete();
    } catch (e) {
      setSaving(false);
    }
  };

  const handleSaveOnly = async () => {
    if (corners.length !== 4) return;
    setSaving(true);
    try {
      await setCourtCorners(jobId, corners.map(c => [c.x, c.y]));
      onBack();
    } catch (e) {
      setSaving(false);
    }
  };

  const handleConfirm = async () => {
    if (corners.length !== 4) return;
    setSaving(true);
    try {
      await setCourtCorners(jobId, corners.map(c => [c.x, c.y]));
      if (redoMode) {
        onComplete();
      } else {
        await processVideo(jobId, poseModel, sampleRate);
        onComplete();
      }
    } catch (e) {
      setSaving(false);
    }
  };

  return (
    <div className="min-h-screen court-pattern flex items-center justify-center p-6">
      <div className="w-full max-w-4xl animate-entrance">
        <div className="text-center mb-6">
          <h1 className="font-display text-4xl text-shuttle-lime tracking-wider mb-2">
            {redoMode ? 'REDO' : 'SET UP'} COURT CORNERS
          </h1>
          <p className="text-text-secondary text-sm">
            {redoMode
              ? 'Adjust the court corners below to fix auto-detection. Your existing corners are pre-loaded.'
              : 'For phone footage, click the 4 court corners in order: Bottom-Left → Bottom-Right → Top-Left → Top-Right'}
          </p>
        </div>

        {loading ? (
          <div className="flex items-center justify-center h-96">
            <div className="w-8 h-8 border-2 border-shuttle-lime/30 border-t-shuttle-lime rounded-full animate-spin" />
          </div>
        ) : (
          <>
            <div className="relative bg-court-surface rounded-xl overflow-hidden border border-court-line/30">
              <canvas
                ref={canvasRef}
                className="w-full cursor-crosshair"
                onClick={handleCanvasClick}
              />
            </div>

            <div className="flex items-center justify-between mt-4 px-1">
              <div className="text-text-secondary text-sm">
                {step < 4 ? (
                  <span>
                    Step {step + 1}/4: Click <span className="text-shuttle-lime font-bold">{CORNER_LABELS[step]}</span>
                  </span>
                ) : (
                  <span className="text-feather-green font-bold">All 4 corners placed. Confirm below.</span>
                )}
              </div>
              {corners.length > 0 && (
                <button
                  onClick={handleUndo}
                  className="text-text-muted hover:text-text-primary text-sm transition-colors"
                >
                  ← Undo
                </button>
              )}
            </div>

            <div className="flex gap-4 mt-8">
              <button
                onClick={handleSkip}
                disabled={saving}
                className="flex-1 py-3 rounded-xl border border-court-line/40 text-text-secondary
                           hover:bg-court-surface/50 transition-all font-body text-sm disabled:opacity-50"
              >
                {redoMode ? 'Cancel' : saving ? 'Starting Pipeline...' : 'Skip / Use Auto-Detection'}
              </button>
              {corners.length === 4 && (
                <button
                  onClick={handleSaveOnly}
                  disabled={saving}
                  className="flex-1 py-3 rounded-xl border border-shuttle-lime/40 text-shuttle-lime
                             hover:bg-shuttle-lime/10 transition-all font-body text-sm disabled:opacity-50"
                >
                  {saving ? 'Saving...' : 'Save Only'}
                </button>
              )}
              <button
                onClick={handleConfirm}
                disabled={corners.length !== 4 || saving}
                className="flex-1 py-3 rounded-xl bg-shuttle-lime text-court-dark font-bold
                           hover:bg-shuttle-lime-dim transition-all font-body text-sm
                           disabled:opacity-30 disabled:cursor-not-allowed"
              >
                {saving ? 'Saving...' : redoMode ? 'Save & Return' : 'Confirm & Process'}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
