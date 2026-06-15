# Report Dashboard Enhancement — Design Spec

**Date:** 2026-06-15
**Status:** Approved — Ready for Implementation

---

## 1. Overview

Enhance the existing ReportView with 5 missing features from the original spec: player selection toggle, court heatmap overlay, footwork/fitness charts, synchronized video timeline, and PDF export.

**Approach:** Incremental enhancement of the existing 3-tab ReportView (Approach A). No new routing, no layout redesign.

**Current state:** ReportView has video player, shot chart donut, coach panel, rally table, tactical tab (shot distribution + patterns), and technical tab (stroke assessments).

---

## 2. Features

### 2.1 Player Selection Toggle

**What:** A dropdown/toggle in the report header to switch between "Near" and "Far" player views. All charts, coach recommendations, and stats filter to the selected player.

**Location:** Report header, next to the existing tab switcher.

**Data source:** The report data is keyed by player_id (`report.footwork.player_1`, `report.fitness.player_1`, etc.). The toggle maps near/far to the actual player_id from the players artifact.

**Behavior:**
- Default: first player (near side)
- Changing the toggle re-renders all panels with that player's data
- Shot distribution chart shows only that player's shots
- Coach panel shows recommendations for that player
- Technical tab shows that player's stroke assessments

**UI:**
```
[NEW] | MATCH REPORT | [Near ▾] | [overview] [tactical] [technical] [fitness]
```

### 2.2 Court Heatmap Overlay

**What:** A canvas-based court visualization showing player position data from `court_analytics.zone_transitions`.

**Location:** Overview tab, below the video player (full width, 12 columns).

**Data source:** `report.court_analytics.zone_transitions` (array of `{frame, zone, player_id}`) and `report.court_analytics.court_dimensions` ({length, width}).

**Visualization:**
- Semi-transparent badminton court outline (scaled to component width)
- 3x3 grid lines showing the 9 zones
- Colored dots for each zone transition (color = player_id)
- Optional: intensity overlay showing visit frequency per zone

**Implementation:** HTML5 Canvas rendered in a React component. No external charting library needed — simple 2D drawing.

### 2.3 Footwork & Fitness Charts

**What:** Two new chart components for physical performance analysis.

**Location:** New 4th tab called "Fitness" in the header tab bar.

**Components:**

1. **FatigueTrendChart** (Recharts LineChart)
   - X-axis: Rally index (1, 2, 3, ...)
   - Y-axis: Shots per second (rally intensity)
   - Data: `report.fitness[selectedPlayer].rally_intensities`
   - Shows fatigue pattern over the match

2. **FitnessStatsCards** (stat display)
   - Total distance covered
   - Average recovery time
   - Fatigue trend indicator (improving/stable/declining with color)
   - Data: `report.fitness[selectedPlayer]` and `report.footwork[selectedPlayer]`

### 2.4 Synchronized Video Timeline

**What:** A timeline bar below the video player showing rally boundaries and shot markers.

**Location:** Overview tab, integrated into the VideoPlayer component area.

**Data source:** `report.rallies` (array of `{rally_id, start_frame, end_frame, shot_count}`)

**Visualization:**
- Horizontal bar representing total video duration
- Colored segments for each rally (alternating colors)
- Shot markers as small dots within segments
- Click on a segment → seek video to that rally's start frame
- Hover tooltip showing rally number and shot count

**Implementation:** Custom React component below the video.js player. Uses video.js `currentTime` to sync position indicator.

### 2.5 PDF Export

**What:** An "Export PDF" button that triggers browser print with print-optimized CSS.

**Location:** Report header, right side.

**Implementation:**
- Button calls `window.print()`
- Print CSS (`@media print`) in `index.css`:
  - Hides: video player, buttons, WebSocket indicators, interactive elements
  - Shows: all charts, stats, coach recommendations, rally table
  - Formats for A4 paper (proper margins, page breaks between sections)
  - Ensures charts render at print quality

---

## 3. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `frontend/src/views/ReportView.tsx` | Modify | Add player toggle, fitness tab, timeline, heatmap |
| `frontend/src/components/VideoPlayer.tsx` | Modify | Add timeline bar with rally segments |
| `frontend/src/components/CourtHeatmap.tsx` | Create | Canvas-based court visualization |
| `frontend/src/components/FatigueTrendChart.tsx` | Create | Recharts line chart for rally intensities |
| `frontend/src/components/FitnessStats.tsx` | Create | Stat cards for fitness/work metrics |
| `frontend/src/index.css` | Modify | Add print CSS media query |

---

## 4. Data Dependencies

All data comes from the existing `/api/jobs/{job_id}/report` endpoint. No backend changes needed.

| Feature | Report field | Type |
|---------|-------------|------|
| Player toggle | `report.footwork`, `report.fitness`, `report.tactical`, `report.technical` | per-player dict |
| Court heatmap | `report.court_analytics.zone_transitions`, `report.court_analytics.court_dimensions` | array, dict |
| Fatigue chart | `report.fitness[pid].rally_intensities` | float[] |
| Fitness stats | `report.fitness[pid]`, `report.footwork[pid]` | dict |
| Timeline | `report.rallies` | array of {start_frame, end_frame} |

---

## 5. Non-Goals

- Per-frame position data (not stored in backend)
- Video annotation/drawing tools
- Real-time video overlay synced to playback
- Server-side PDF generation
- Doubles player support (single player toggle only)
