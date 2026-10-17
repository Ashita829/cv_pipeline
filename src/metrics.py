import os

import pandas as pd


class MetricsLogger:
    """Stores one row per frame: stage timings, total latency, FPS, counts."""

    def __init__(self):
        self.rows = []

    def log(self, frame_idx, stage_times, total_ms, n_shapes, n_dets):
        row = {"frame": frame_idx, "total_ms": total_ms,
               "fps": 1000 / total_ms if total_ms > 0 else 0,
               "n_shapes": n_shapes, "n_detections": n_dets}
        row.update({f"{k}_ms": v for k, v in stage_times.items()})
        self.rows.append(row)

    def dataframe(self):
        return pd.DataFrame(self.rows)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.dataframe().to_csv(path, index=False)

    def summary(self):
        df = self.dataframe()
        return {
            "frames": len(df),
            "mean_fps": df["fps"].mean(),
            "mean_latency_ms": df["total_ms"].mean(),
            "p95_latency_ms": df["total_ms"].quantile(0.95),
        }


# ---------- Detection accuracy ----------
def iou(a, b):
    """Intersection over Union of two boxes (x1, y1, x2, y2)."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0


def match_frame(preds, gts, iou_thr=0.5):
    """Greedy matching of predictions to ground truth of the same class.
    preds/gts: lists of dicts with 'name' and 'box'. Returns (tp, fp, fn)."""
    used = set()
    tp = 0
    for p in sorted(preds, key=lambda d: -d.get("conf", 1)):
        best, best_j = 0, None
        for j, g in enumerate(gts):
            if j in used or g["name"] != p["name"]:
                continue
            score = iou(p["box"], g["box"])
            if score > best:
                best, best_j = score, j
        if best >= iou_thr:
            tp += 1
            used.add(best_j)
    return tp, len(preds) - tp, len(gts) - tp