import os

import cv2

from src import classical, config, overlay
from src.detector import YoloDetector
from src.metrics import MetricsLogger
from src.utils import StageTimer, resize_width


def process_video(video_path=config.VIDEO_PATH, use_classical=True, use_yolo=True,
                  model_name=config.YOLO_MODEL, width=config.FRAME_WIDTH,
                  blur_k=config.BLUR_KERNEL, show=True, save_path=None,
                  max_frames=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {video_path}")

    detector = YoloDetector(model_name) if use_yolo else None
    logger = MetricsLogger()
    writer = None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or (max_frames and idx >= max_frames):
            break

        timer = StageTimer()

        with timer.stage("resize"):
            frame = resize_width(frame, width)

        shapes, detections = [], []
        if use_classical:
            shapes, mask = classical.run(frame, timer, blur_k=blur_k)
        if use_yolo:
            with timer.stage("yolo"):
                detections = detector.detect(frame)

        with timer.stage("overlay"):
            out = frame.copy()
            if use_classical:
                classical.draw(out, shapes)
            if use_yolo:
                overlay.draw_detections(out, detections)
                overlay.draw_counts(out, detections, detector.unique_counts())

        # Timing stops BEFORE display/saving so it isn't counted in FPS
        total_ms = timer.total()
        logger.log(idx, timer.times, total_ms, len(shapes), len(detections))
        overlay.draw_fps(out, 1000 / total_ms)

        if save_path:
            if writer is None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(
                    save_path, cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (w, h))
            writer.write(out)

        if show:
            cv2.imshow("Pipeline (q to quit)", out)
            if use_classical:
                cv2.imshow("Mask", mask)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    return logger