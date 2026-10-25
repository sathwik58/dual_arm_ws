"""Shared code for the drawing recogniser: shape classes, synthetic training images,
preprocessing, the CNN, and geometric measurements (vertices / lines).

Used by train_shape_cnn.py (training) and shape_recognizer.py (ROS node).
"""
import math
import random

import cv2
import numpy as np
import torch
from torch import nn

# ------------------------------------------------------------------ classes
CLASSES = ['circle', 'triangle', 'square', 'pentagon', 'hexagon', 'star', 'spiral',
           'square_circle']

# Expected geometry per class: straight lines, corner vertices, curved strokes, separate parts
GEOMETRY = {
    'circle':        {'lines': 0,  'vertices': 0,  'curves': 1, 'parts': 1},
    'triangle':      {'lines': 3,  'vertices': 3,  'curves': 0, 'parts': 1},
    'square':        {'lines': 4,  'vertices': 4,  'curves': 0, 'parts': 1},
    'pentagon':      {'lines': 5,  'vertices': 5,  'curves': 0, 'parts': 1},
    'hexagon':       {'lines': 6,  'vertices': 6,  'curves': 0, 'parts': 1},
    'star':          {'lines': 10, 'vertices': 10, 'curves': 0, 'parts': 1},
    'spiral':        {'lines': 0,  'vertices': 0,  'curves': 1, 'parts': 1},
    'square_circle': {'lines': 4,  'vertices': 4,  'curves': 1, 'parts': 2},
}

INPUT = 96          # network input size (pixels)


# ------------------------------------------------------------------ synthetic images
def _polygon(n, cx, cy, r, rot):
    return [(cx + r * math.cos(rot + 2 * math.pi * i / n),
             cy + r * math.sin(rot + 2 * math.pi * i / n)) for i in range(n)]


def _star(cx, cy, r, rot, spikes=5):
    return [(cx + (r if i % 2 == 0 else 0.4 * r) * math.cos(rot + i * math.pi / spikes),
             cy + (r if i % 2 == 0 else 0.4 * r) * math.sin(rot + i * math.pi / spikes))
            for i in range(2 * spikes)]


def _spiral(cx, cy, r, rot, turns=3, n=300):
    return [(cx + r * i / n * math.cos(rot + 2 * math.pi * turns * i / n),
             cy + r * i / n * math.sin(rot + 2 * math.pi * turns * i / n)) for i in range(n + 1)]


def render_shape(name, rng=random):
    """A synthetic 'scan_topdown' image: tan table, black pen lines, 1 px ~ 1 mm."""
    size = rng.randint(220, 420)
    bg = np.array([120, 155, 185]) + rng.randint(-25, 25)        # BGR, table colour
    img = np.empty((size, size, 3), np.uint8)
    img[:] = np.clip(bg, 0, 255)
    cx = size / 2 + rng.uniform(-0.08, 0.08) * size
    cy = size / 2 + rng.uniform(-0.08, 0.08) * size
    r = rng.uniform(0.22, 0.40) * size
    rot = rng.uniform(0, 2 * math.pi)
    thick = rng.randint(2, 5)
    ink = (rng.randint(0, 40),) * 3

    def poly(pts, closed=True):
        p = np.round(np.array(pts) * 16).astype(np.int32)
        cv2.polylines(img, [p], closed, ink, thick, cv2.LINE_AA, shift=4)

    if name == 'circle':
        aspect = rng.uniform(0.93, 1.07)
        poly([(cx + r * math.cos(a), cy + aspect * r * math.sin(a))
              for a in np.linspace(0, 2 * math.pi, 180, endpoint=False)])
    elif name in ('triangle', 'square', 'pentagon', 'hexagon'):
        n = {'triangle': 3, 'square': 4, 'pentagon': 5, 'hexagon': 6}[name]
        poly(_polygon(n, cx, cy, r, rot))
    elif name == 'star':
        poly(_star(cx, cy, r, rot))
    elif name == 'spiral':
        poly(_spiral(cx, cy, r, rot), closed=False)
    elif name == 'square_circle':
        poly(_polygon(4, cx, cy, r, rot))
        rc = r * math.cos(math.pi / 4) * rng.uniform(0.45, 0.9)   # circle inside the square
        poly([(cx + rc * math.cos(a), cy + rc * math.sin(a))
              for a in np.linspace(0, 2 * math.pi, 120, endpoint=False)])

    # small perspective error (projection), blur and noise, like the real scan
    d = 0.03 * size
    src = np.float32([[0, 0], [size, 0], [size, size], [0, size]])
    dst = src + np.float32([[rng.uniform(-d, d), rng.uniform(-d, d)] for _ in range(4)])
    img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (size, size),
                              borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.7:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    noise = np.random.normal(0, rng.uniform(2, 8), img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------ preprocessing
def ink_mask(img_bgr):
    """Binary mask of the pen lines (ink = 255): pixels clearly darker than the table."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    seen = gray < 245                       # scan_drawing paints unseen areas white
    table = np.median(gray[seen]) if seen.any() else np.median(gray)
    mask = (gray < table - 40).astype(np.uint8) * 255
    # keep only pen strokes: drop specks (< 30 px) and solid dark blobs such as a
    # shadow or a robot part (a pen line is ~3 mm wide, so its centre is well under 10 px from an edge)
    thickness = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        part = labels == i
        if stats[i, cv2.CC_STAT_AREA] >= 30 and thickness[part].max() <= 10:
            keep[part] = 255
    return keep


def crop_square(mask, pad=0.12):
    """Crop to the ink, pad, make square. Returns (crop, (x0, y0, side)) or (None, None)."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None, None
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    side = int(max(x1 - x0, y1 - y0) * (1 + 2 * pad)) + 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sx, sy = int(round(cx - side / 2)), int(round(cy - side / 2))
    canvas = np.zeros((side, side), np.uint8)
    h, w = mask.shape
    ax0, ay0 = max(sx, 0), max(sy, 0)
    ax1, ay1 = min(sx + side, w), min(sy + side, h)
    canvas[ay0 - sy:ay1 - sy, ax0 - sx:ax1 - sx] = mask[ay0:ay1, ax0:ax1]
    return canvas, (sx, sy, side)


def to_tensor(mask):
    """Mask -> 1 x INPUT x INPUT float tensor (ink = 1). Line width is normalised first."""
    crop, _ = crop_square(mask)
    if crop is None:
        return None
    small = cv2.resize(crop, (INPUT, INPUT), interpolation=cv2.INTER_AREA)
    small = cv2.dilate((small > 40).astype(np.uint8), np.ones((2, 2), np.uint8))
    return torch.from_numpy(small.astype(np.float32))[None]


# ------------------------------------------------------------------ the CNN
class ShapeCNN(nn.Module):
    """4 conv blocks (16-32-64-128 channels) -> global average pool -> 2 fully connected."""

    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(1, 16), block(16, 32), block(32, 64), block(64, 128))
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, 64), nn.ReLU(inplace=True), nn.Dropout(0.3),
                                  nn.Linear(64, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


# ------------------------------------------------------------------ geometric measurement
def measure(mask):
    """Measure the drawing with OpenCV: separate parts, and for the largest part the
    number of corner vertices (polygon approximation) and how round it is."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.05 * mask.sum() / 255]
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    outer = max(contours, key=lambda c: cv2.contourArea(cv2.convexHull(c)))
    perim = cv2.arcLength(outer, True)
    area = cv2.contourArea(outer)
    circularity = 4 * math.pi * area / (perim * perim) if perim > 0 else 0.0
    approx = cv2.approxPolyDP(outer, 0.03 * perim, True)
    x, y, w, h = cv2.boundingRect(outer)
    return {'parts': len(parts), 'outer_vertices': len(approx),
            'circularity': round(circularity, 3), 'bbox_px': (x, y, w, h),
            'outer_polygon': approx.reshape(-1, 2).tolist()}