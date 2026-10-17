from collections import Counter

import cv2


def draw_detections(frame, detections):
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f'{d["name"]} {d["conf"]:.2f}'
        if d["id"] is not None:
            label = f'#{d["id"]} ' + label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(frame, label, (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)


def draw_counts(frame, detections, unique_counts):
    """Dynamic text counts: current frame count + unique total per class."""
    current = Counter(d["name"] for d in detections)
    y = 25
    cv2.putText(frame, "Class: now / total", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    for name in sorted(set(current) | set(unique_counts)):
        y += 22
        text = f"{name}: {current.get(name, 0)} / {unique_counts.get(name, 0)}"
        cv2.putText(frame, text, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)


def draw_fps(frame, fps):
    h = frame.shape[0]
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)