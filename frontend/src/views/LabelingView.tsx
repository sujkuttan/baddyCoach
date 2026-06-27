import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { VideoPlayer, type VideoPlayerHandle } from '../components/VideoPlayer';

const COACH_CLASSES = [
  "net_shot", "block", "smash", "lift", "clear", "drive",
  "drop", "push", "rush", "cross_court", "short_serve", "long_serve",
];

const HOTKEYS: Record<string, string> = {
  "1": "net_shot", "2": "block", "3": "smash", "4": "lift",
  "5": "clear", "6": "drive", "7": "drop", "8": "push",
  "9": "rush", "0": "cross_court", "-": "short_serve", "=": "long_serve",
};

type LabelStatus = "labeled" | "unsure" | "not_a_shot" | "skipped";

interface ShotLabel {
  true_stroke: string;
  status: LabelStatus;
}

const STORAGE_PREFIX = "baddycoach_labels_";

function hash(s: string): string {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h) + s.charCodeAt(i);
    h |= 0;
  }
  return Math.abs(h).toString(36);
}

interface LabelingViewProps {
  shots: any[];
  videoUrl?: string | null;
  jobId?: string | null;
  fps?: number;
  labelPreRoll?: number;
  onBack: () => void;
}

export function LabelingView({ shots, videoUrl, jobId, fps = 30, labelPreRoll = 0.7, onBack }: LabelingViewProps) {
  const videoRef = useRef<VideoPlayerHandle>(null);
  const tableRef = useRef<HTMLDivElement>(null);

  const storageKey = useMemo(() => {
    const id = jobId || "import";
    return STORAGE_PREFIX + id + "_" + hash(JSON.stringify(shots.map(s => s.shot_id + ":" + s.frame)));
  }, [jobId, shots]);

  const [labels, setLabels] = useState<Record<number, ShotLabel>>(() => {
    try {
      const saved = localStorage.getItem(storageKey);
      if (saved) return JSON.parse(saved);
    } catch { /* ignore */ }
    return {};
  });

  const [selectedIdx, setSelectedIdx] = useState<number>(0);
  const [filterSource, setFilterSource] = useState<string>("all");
  const [showLabeled, setShowLabeled] = useState<boolean>(true);
  const [prefillPrediction, setPrefillPrediction] = useState<boolean>(true);

  useEffect(() => {
    localStorage.setItem(storageKey, JSON.stringify(labels));
  }, [labels, storageKey]);

  const filtered = useMemo(() => {
    return shots.filter((s, i) => {
      const lbl = labels[s.shot_id];
      const isLabeled = lbl && (lbl.status === "labeled" || lbl.status === "not_a_shot");
      if (!showLabeled && isLabeled) return false;
      if (filterSource !== "all" && s.stroke_source !== filterSource) return false;
      return true;
    });
  }, [shots, labels, filterSource, showLabeled]);

  const current = shots[selectedIdx];
  const filteredIdx = filtered.findIndex(s => s.shot_id === current?.shot_id);

  const labelCurrent = useCallback((stroke: string, status: LabelStatus) => {
    setLabels(prev => ({ ...prev, [current.shot_id]: { true_stroke: stroke, status } }));
    if (status === "labeled" || status === "not_a_shot") {
      setSelectedIdx(i => Math.min(i + 1, shots.length - 1));
    }
  }, [current, shots.length]);

  const playSegment = useCallback(() => {
    if (!current) return;
    const start = Math.max(0, current.start_ts - labelPreRoll);
    const end = current.ts_end || current.start_ts + 1.0;
    videoRef.current?.playSegment(start, end, true);
  }, [current, labelPreRoll]);

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (!current) return;
    if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
    const stroke = HOTKEYS[e.key];
    if (stroke) {
      e.preventDefault();
      labelCurrent(stroke, "labeled");
      return;
    }
    switch (e.key) {
      case "u":
        e.preventDefault();
        labelCurrent(current.stroke_type, "unsure");
        break;
      case "x":
        e.preventDefault();
        labelCurrent("", "not_a_shot");
        break;
      case " ":
        e.preventDefault();
        playSegment();
        break;
      case "Enter":
        e.preventDefault();
        if (!labels[current.shot_id]) {
          const prefill = prefillPrediction ? current.stroke_type : "unknown";
          labelCurrent(prefill, "labeled");
        } else {
          setSelectedIdx(i => Math.min(i + 1, shots.length - 1));
        }
        break;
      case "ArrowLeft":
        e.preventDefault();
        setSelectedIdx(i => Math.max(0, i - 1));
        break;
      case "ArrowRight":
        e.preventDefault();
        setSelectedIdx(i => Math.min(i + 1, shots.length - 1));
        break;
    }
  }, [current, labels, labelCurrent, playSegment, prefillPrediction]);

  useEffect(() => {
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleKeyDown]);

  useEffect(() => {
    if (current) {
      const t = Math.max(0, current.start_ts - labelPreRoll);
      videoRef.current?.seekTo(t);
    }
  }, [current, labelPreRoll]);

  const exportCsv = useCallback(() => {
    const rows = [["shot_id", "frame", "ts_start", "ts_end", "player_id", "side",
                   "predicted_stroke", "predicted_class_id", "true_stroke", "true_class_id", "label_status"].join(",")];
    for (const s of shots) {
      const lbl = labels[s.shot_id];
      const status = lbl?.status || "skipped";
      const true_stroke = lbl?.true_stroke || "";
      const true_class_id = true_stroke && s.side
        ? mapToClassId(true_stroke, s.side) : "";
      rows.push([
        s.shot_id, s.frame,
        (Math.max(0, s.start_ts - labelPreRoll)).toFixed(3),
        (s.ts_end || s.start_ts + 1.0).toFixed(3),
        s.player_id || "", s.side || "",
        s.stroke_type, s.shuttleset_class_id || 0,
        true_stroke, true_class_id, status,
      ].join(","));
    }
    const blob = new Blob([rows.join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `labels_${jobId || "import"}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [shots, labels, jobId]);

  const labeledCount = Object.values(labels).filter(l => l.status === "labeled" || l.status === "not_a_shot").length;

  if (!shots.length) {
    return (
      <div className="min-h-screen court-pattern p-6">
        <div className="max-w-4xl mx-auto text-center py-20">
          <p className="font-mono text-sm text-text-muted">No shots to label.</p>
          <button onClick={onBack} className="mt-4 px-4 py-2 rounded-lg bg-court-surface/30 hover:bg-court-surface/50 font-mono text-xs text-text-primary transition-colors">Back</button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen court-pattern p-4">
      <div className="max-w-7xl mx-auto space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button onClick={onBack} className="font-mono text-[10px] text-text-muted hover:text-text-primary transition-colors">&larr; Back</button>
            <h1 className="font-mono text-sm text-text-primary tracking-widest">SHOT LABELER</h1>
          </div>
          <div className="flex items-center gap-3">
            <span className="font-mono text-[10px] text-text-muted">
              {labeledCount} / {shots.length} labeled
            </span>
            <button onClick={exportCsv} className="px-3 py-1.5 rounded-lg bg-shuttle-lime/20 hover:bg-shuttle-lime/30 font-mono text-[10px] text-shuttle-lime transition-colors">
              EXPORT CSV
            </button>
            <button onClick={() => { setLabels({}); localStorage.removeItem(storageKey); }}
                    className="px-3 py-1.5 rounded-lg bg-error-red/10 hover:bg-error-red/20 font-mono text-[10px] text-error-red transition-colors">
              RESET
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Video + controls */}
          <div className="lg:col-span-2 space-y-3">
            <VideoPlayer ref={videoRef} videoUrl={videoUrl} jobId={jobId} fps={fps} />

            {current && (
              <div className="bg-court-dark/60 rounded-xl border border-court-line/15 p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] text-text-muted tracking-widest">
                    SHOT #{current.shot_id} &middot; t={current.start_ts}s &middot; {current.player_id || "?"} ({current.side || "?"})
                  </span>
                  <span className="font-mono text-[9px] text-text-muted">
                    {filteredIdx >= 0 ? `${filteredIdx + 1}/${filtered.length}` : `${selectedIdx + 1}/${shots.length}`}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className="font-mono text-[10px] text-text-muted">Predicted:</span>
                  <span className={`px-2 py-0.5 rounded font-mono text-[10px] ${
                    current.stroke_confidence >= 0.4 ? "text-shuttle-lime bg-shuttle-lime/10" :
                    current.stroke_confidence >= 0.2 ? "text-warning-yellow bg-warning-yellow/10" :
                    "text-error-red bg-error-red/10"
                  }`}>
                    {current.stroke_type} ({Math.round(current.stroke_confidence * 100)}%)
                  </span>
                  <span className="font-mono text-[9px] text-text-muted">{current.stroke_source || "bst"}</span>
                </div>

                {/* Class buttons */}
                <div className="grid grid-cols-6 gap-1.5">
                  {COACH_CLASSES.map((cls) => {
                    const isActive = labels[current.shot_id]?.true_stroke === cls;
                    return (
                      <button
                        key={cls}
                        onClick={() => labelCurrent(cls, "labeled")}
                        className={`px-2 py-1.5 rounded font-mono text-[9px] transition-colors ${
                          isActive
                            ? "bg-shuttle-lime/30 text-shuttle-lime border border-shuttle-lime/40"
                            : "bg-court-surface/20 hover:bg-court-surface/40 text-text-muted border border-transparent hover:border-court-line/20"
                        }`}
                      >
                        {cls.replace("_", " ")}
                      </button>
                    );
                  })}
                </div>

                {/* Action buttons */}
                <div className="flex items-center gap-2">
                  <button onClick={() => labelCurrent(current.stroke_type, "unsure")}
                          className="px-3 py-1 rounded-lg bg-warning-yellow/10 hover:bg-warning-yellow/20 font-mono text-[9px] text-warning-yellow transition-colors">
                    UNSURE [u]
                  </button>
                  <button onClick={() => labelCurrent("", "not_a_shot")}
                          className="px-3 py-1 rounded-lg bg-error-red/10 hover:bg-error-red/20 font-mono text-[9px] text-error-red transition-colors">
                    NOT A SHOT [x]
                  </button>
                  <button onClick={() => setSelectedIdx(i => Math.max(0, i - 1))}
                          className="px-3 py-1 rounded-lg bg-court-surface/20 hover:bg-court-surface/40 font-mono text-[9px] text-text-muted transition-colors">
                    &larr; PREV
                  </button>
                  <button onClick={() => setSelectedIdx(i => Math.min(i + 1, shots.length - 1))}
                          className="px-3 py-1 rounded-lg bg-court-surface/20 hover:bg-court-surface/40 font-mono text-[9px] text-text-muted transition-colors">
                    NEXT &rarr;
                  </button>
                </div>

                {/* Hotkey help */}
                <div className="font-mono text-[8px] text-text-muted/40">
                  Keys: 1-9,0,-,= stroke &middot; u=unsure &middot; x=not a shot &middot; Space=replay &middot; Enter=confirm+next &middot; &larr;&rarr;=prev/next
                </div>
              </div>
            )}
          </div>

          {/* Shot table */}
          <div className="space-y-2">
            {/* Filters */}
            <div className="bg-court-dark/60 rounded-xl border border-court-line/15 p-3 space-y-2">
              <div className="flex items-center gap-2">
                <label className="font-mono text-[9px] text-text-muted">Source:</label>
                <select value={filterSource} onChange={e => setFilterSource(e.target.value)}
                        className="bg-court-surface/30 border border-court-line/20 rounded px-1.5 py-0.5 font-mono text-[9px] text-text-primary">
                  <option value="all">All</option>
                  <option value="bst">BST</option>
                  <option value="agree">Agree</option>
                  <option value="physics_override">Veto</option>
                  <option value="physics_fallback">Fallback</option>
                  <option value="bst_no_physics">No physics</option>
                </select>
              </div>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showLabeled} onChange={e => setShowLabeled(e.target.checked)}
                       className="rounded border-court-line/30" />
                <span className="font-mono text-[9px] text-text-muted">Show labeled</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={prefillPrediction} onChange={e => setPrefillPrediction(e.target.checked)}
                       className="rounded border-court-line/30" />
                <span className="font-mono text-[9px] text-text-muted">Prefill with prediction</span>
              </label>
            </div>

            {/* Shot list */}
            <div ref={tableRef} className="bg-court-dark/60 rounded-xl border border-court-line/15 max-h-[60vh] overflow-y-auto">
              {filtered.map((s) => {
                const lbl = labels[s.shot_id];
                const isSelected = s.shot_id === current?.shot_id;
                const isLabeled = lbl && (lbl.status === "labeled" || lbl.status === "not_a_shot");
                return (
                  <div
                    key={s.shot_id}
                    onClick={() => {
                      const idx = shots.findIndex(sh => sh.shot_id === s.shot_id);
                      if (idx >= 0) setSelectedIdx(idx);
                    }}
                    className={`flex items-center gap-2 px-3 py-1.5 cursor-pointer transition-colors border-b border-court-line/5 ${
                      isSelected ? "bg-shuttle-lime/10 border-l-2 border-l-shuttle-lime" : "hover:bg-court-surface/20"
                    } ${isLabeled ? "opacity-40" : ""}`}
                  >
                    <span className="font-mono text-[9px] text-text-muted w-6 shrink-0">#{s.shot_id}</span>
                    <span className="font-mono text-[9px] text-text-primary w-14 shrink-0">
                      {s.stroke_type?.substring(0, 8)}
                    </span>
                    {lbl?.status === "labeled" && (
                      <span className="font-mono text-[9px] text-shuttle-lime shrink-0">
                        &rarr; {lbl.true_stroke.substring(0, 8)}
                      </span>
                    )}
                    {lbl?.status === "not_a_shot" && (
                      <span className="font-mono text-[9px] text-error-red shrink-0">[X]</span>
                    )}
                    {lbl?.status === "unsure" && (
                      <span className="font-mono text-[9px] text-warning-yellow shrink-0">[?]</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function mapToClassId(stroke: string, side: string): number {
  if (side === "far") {
    const idx = COACH_CLASSES.indexOf(stroke);
    return idx >= 0 ? idx + 1 : 0;
  }
  const idx = COACH_CLASSES.indexOf(stroke);
  return idx >= 0 ? idx + 13 : 0;
}
