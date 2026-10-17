# Computer Vision Object Detector & Frame-by-Frame Segmenter

Real-time pipeline combining classical OpenCV segmentation
(Gaussian blur, adaptive threshold, contours) with YOLOv8 detection
and tracking, plus per-stage latency metrics.

## Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

## Run
python run_pipeline.py --mode both --save
python run_experiments.py
python analyze_results.py

## Structure
- src/: classical.py, detector.py, overlay.py, metrics.py, pipeline.py
- outputs/plots/: performance graphs
- report/: whitepaper

## Note
Use Python 3.11 or 3.12 (PyTorch does not support 3.15 yet).