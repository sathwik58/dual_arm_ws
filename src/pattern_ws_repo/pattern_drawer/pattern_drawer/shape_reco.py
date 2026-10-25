"""Recognise the drawing in the UR5 top-down scan.

  ros2 run pattern_drawer shape_recognizer
  ros2 run pattern_drawer shape_recognizer --ros-args -p image:=/path/to/scan_topdown.png

What it does
  1. Load runs/latest/scan_topdown.png (made by scan_drawing, 1 px = 1 mm).
  2. Find the ink (pixels clearly darker than the table).
  3. CNN (trained by train_shape_cnn) -> shape class + confidence.
  4. OpenCV contour + approxPolyDP -> measured corner count, roundness, size, position.
  5. Report shape, number of vertices and lines, centre (world metres) and size;
     cross-check the CNN against the measurement.
  6. Save scan_result.png (annotated) + scan_result.json, publish the JSON on /drawing/shape.
"""
import json
import math
import os

import cv2
import numpy as np
import rclpy
import torch
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String

from pattern_drawer.shape_cnn import CLASSES, GEOMETRY, ShapeCNN, ink_mask, measure, to_tensor

SCAN_MARGIN = 0.05            # must match scan_drawing (area = last_drawing + 5 cm)
RUNS = os.path.expanduser('~/dual_arm_ws/src/pattern_ws_repo/pattern_drawer/runs')


class ShapeRecognizer(Node):
    def __init__(self):
        super().__init__('shape_recognizer')
        self.declare_parameter('image', os.path.join(RUNS, 'latest', 'scan_topdown.png'))
        self.declare_parameter('model', os.path.expanduser('~/pattern_models/shape_cnn.pt'))
        self.declare_parameter('workspace_file', os.path.expanduser('~/panda_workspace.yaml'))
        self.declare_parameter('mm_per_pixel', 1.0)
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)  # late joiners get it
        self.pub = self.create_publisher(String, '/drawing/shape', qos)

    # ---------------------------------------------------------- helpers
    def load_model(self):
        path = self.get_parameter('model').value
        if not os.path.exists(path):
            self.get_logger().error(f'{path} not found - train first: '
                                    'ros2 run pattern_drawer train_shape_cnn')
            return None, None
        ckpt = torch.load(path, map_location='cpu')
        classes = ckpt.get('classes', CLASSES)
        model = ShapeCNN(len(classes))
        model.load_state_dict(ckpt['state_dict'])
        model.eval()
        return model, classes

    def scan_origin(self):
        """World x_max and y_min of the scan image (row 0 = far x, col 0 = min y)."""
        folder = os.path.dirname(os.path.realpath(self.get_parameter('image').value))
        try:
            with open(os.path.join(folder, 'scan_area.yaml')) as f:
                a = yaml.safe_load(f)
            return a['x_max'], a['y_min']
        except (OSError, KeyError, TypeError):
            return None

    def pixel_to_world(self, col, row, origin):
        mm = self.get_parameter('mm_per_pixel').value / 1000.0
        return origin[0] - row * mm, origin[1] + col * mm

    # ---------------------------------------------------------- main
    def run(self):
        path = self.get_parameter('image').value
        img = cv2.imread(path)
        if img is None:
            self.get_logger().error(f'Cannot read {path} - run scan_drawing first')
            return
        model, classes = self.load_model()
        if model is None:
            return

        mask = ink_mask(img)
        if mask.sum() == 0:
            self.get_logger().error('No ink found in the image - is the drawing visible to '
                                    'the camera? (ink strips spawned, shadows off)')
            return
        # keep only the drawing: drop dark blobs much smaller than the main stroke
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        biggest = stats[1:, cv2.CC_STAT_AREA].max()
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < 0.03 * biggest:
                mask[labels == i] = 0

        # 1) CNN
        with torch.no_grad():
            prob = torch.softmax(model(to_tensor(mask)[None]), 1)[0]
        top = prob.argsort(descending=True)[:3].tolist()
        shape = classes[top[0]]
        conf = float(prob[top[0]])

        # 2) OpenCV measurement
        m = measure(mask)
        expected = GEOMETRY[shape]
        if expected['lines'] > 0:            # polygon: corners measured on the outer contour
            check = 'OK' if m['outer_vertices'] == expected['vertices'] else \
                f"MISMATCH (contour has {m['outer_vertices']} corners)"
        else:                                # curve: must be round-ish, no real corners
            check = 'OK' if shape == 'spiral' or m['circularity'] > 0.75 else \
                f"MISMATCH (roundness {m['circularity']:.2f})"

        x, y, w, h = m['bbox_px']
        mm = self.get_parameter('mm_per_pixel').value
        result = {
            'shape': shape, 'confidence': round(conf, 3),
            'alternatives': {classes[i]: round(float(prob[i]), 3) for i in top[1:]},
            'vertices': expected['vertices'], 'lines': expected['lines'],
            'curves': expected['curves'], 'parts': m['parts'],
            'measured_outer_vertices': m['outer_vertices'],
            'roundness': m['circularity'], 'check': check,
            'size_m': [round(h * mm / 1000, 3), round(w * mm / 1000, 3)],   # along x, along y
        }
        origin = self.scan_origin()
        if origin:
            cx, cy = self.pixel_to_world(x + w / 2, y + h / 2, origin)
            result['center_world'] = [round(cx, 3), round(cy, 3)]

        # 3) annotated image
        out = img.copy()
        out[mask > 0] = (0, 0, 255)                                  # ink in red
        poly = np.array(m['outer_polygon'], np.int32)
        if expected['lines'] > 0:
            cv2.polylines(out, [poly], True, (0, 200, 0), 2)
            for px, py in poly:
                cv2.circle(out, (int(px), int(py)), 6, (255, 0, 0), -1)
        cv2.rectangle(out, (x, y), (x + w, y + h), (200, 200, 0), 1)
        label = f"{shape} {conf * 100:.0f}%  V={result['vertices']} L={result['lines']}"
        out = cv2.copyMakeBorder(out, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        cv2.putText(out, label, (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        folder = os.path.dirname(path)
        cv2.imwrite(os.path.join(folder, 'scan_result.png'), out)
        with open(os.path.join(folder, 'scan_result.json'), 'w') as f:
            json.dump(result, f, indent=2)

        self.pub.publish(String(data=json.dumps(result)))
        log = self.get_logger().info
        log(f"Shape: {shape}  (CNN confidence {conf * 100:.1f}%)")
        log(f"Vertices: {result['vertices']}   Lines: {result['lines']}   "
            f"Curves: {result['curves']}   Separate parts: {result['parts']}")
        log(f"Check against contour: {check}")
        log(f"Size: {result['size_m'][0]:.3f} x {result['size_m'][1]:.3f} m" +
            (f"   Centre (world): ({result['center_world'][0]:.3f}, "
             f"{result['center_world'][1]:.3f})" if origin else ''))
        log(f"Saved {os.path.join(folder, 'scan_result.png')} and scan_result.json")


def main():
    rclpy.init()
    node = ShapeRecognizer()
    try:
        node.run()
        rclpy.spin_once(node, timeout_sec=0.5)     # let the message go out
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()