from collections import defaultdict

from ultralytics import YOLO

from src import config


class YoloDetector:
    def __init__(self, model_name=None):
        self.model = YOLO(model_name or config.YOLO_MODEL)
        self.seen_ids = defaultdict(set)   # class name -> set of track IDs ever seen

    def detect(self, frame):
        """Run YOLO with tracking on one frame.
        Returns a list of dicts: box, name, conf, id."""
        result = self.model.track(
            frame, persist=True, conf=config.YOLO_CONF,
            classes=config.TARGET_CLASSES, verbose=False,
        )[0]

        detections = []
        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            name = result.names[cls_id]
            track_id = int(boxes.id[i]) if boxes.id is not None else None
            if track_id is not None:
                self.seen_ids[name].add(track_id)
            detections.append({
                "box": tuple(map(int, boxes.xyxy[i].tolist())),  # x1,y1,x2,y2
                "name": name,
                "conf": float(boxes.conf[i]),
                "id": track_id,
            })
        return detections

    def unique_counts(self):
        """Total unique objects per class across the whole video."""
        return {name: len(ids) for name, ids in self.seen_ids.items()}