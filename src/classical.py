import cv2
import numpy as np

from src import config


def to_blurred_gray(frame, kernel, sigma):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, kernel, sigma)


def adaptive_mask(blurred, block, c):
    """Local threshold: each pixel is compared to the mean of its neighborhood.
    THRESH_BINARY_INV makes dark objects white in the mask."""
    return cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block, c,
    )


def clean_mask(mask, kernel_size):
    """Open removes small white specks, close fills small holes."""
    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def classify_shape(contour):
    """Guess a shape name from the number of polygon corners."""
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    n = len(approx)
    if n == 3:
        return "triangle"
    if n == 4:
        x, y, w, h = cv2.boundingRect(approx)
        return "square" if 0.9 <= w / h <= 1.1 else "rectangle"
    if n > 6:
        return "circle"
    return "polygon"


def find_shapes(mask, min_area, max_area):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:      # pixel-bounds filter
            shapes.append({
                "contour": c,
                "bbox": cv2.boundingRect(c),
                "area": area,
                "name": classify_shape(c),
            })
    return shapes


def run(frame, timer, blur_k=None, block=None, c=None):
    """Full classical chain. Returns (shapes, mask)."""
    blur_k = blur_k or config.BLUR_KERNEL
    block = block or config.ADAPT_BLOCK
    c = c if c is not None else config.ADAPT_C

    with timer.stage("blur"):
        blurred = to_blurred_gray(frame, blur_k, config.BLUR_SIGMA)
    with timer.stage("threshold"):
        mask = adaptive_mask(blurred, block, c)
    with timer.stage("morph"):
        mask = clean_mask(mask, config.MORPH_KERNEL)
    with timer.stage("contours"):
        shapes = find_shapes(mask, config.MIN_AREA, config.MAX_AREA)
    return shapes, mask


def draw(frame, shapes):
    for s in shapes:
        x, y, w, h = s["bbox"]
        cv2.drawContours(frame, [s["contour"]], -1, (0, 255, 0), 1)
        cv2.putText(frame, s["name"], (x, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)