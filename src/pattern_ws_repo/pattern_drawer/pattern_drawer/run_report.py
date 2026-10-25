"""Text report for one run: what was asked vs what was drawn."""
import csv
import json
import os
import sys

import cv2
import numpy as np
import yaml

from pattern_drawer.shape_cnn import GEOMETRY, ink_mask

RUNS = os.path.expanduser('~/dual_arm_ws/src/pattern_ws_repo/pattern_drawer/runs')
TOL_MM = 5          # ink within 5 mm of the asked line counts as correct
PASS_ACC = 80.0     # accuracy % needed to PASS


def load(run):
    with open(os.path.join(run, 'meta.yaml')) as f:
        meta = yaml.safe_load(f)
    with open(os.path.join(run, 'scan_area.yaml')) as f:
        area = yaml.safe_load(f)
    with open(os.path.join(run, 'scan_result.json')) as f:
        result = json.load(f)
    scan = cv2.imread(os.path.join(run, 'scan_topdown.png'))
    return meta, area, result, scan


def commanded_mask(meta, area, shape_hw):
    """Draw the asked path on the same pixel grid as the scan (1 px = 1 mm)."""
    mm = area['mm_per_pixel'] / 1000.0
    mask = np.zeros(shape_hw, np.uint8)
    for stroke in meta['strokes']:
        pts = np.array([[(y - area['y_min']) / mm, (area['x_max'] - x) / mm]
                        for x, y in stroke], np.int32)          # (column, row)
        cv2.polylines(mask, [pts], False, 255, 3)
    return mask


def metrics(cmd, ink):
    d_to_cmd = cv2.distanceTransform(255 - cmd, cv2.DIST_L2, 3)
    d_to_ink = cv2.distanceTransform(255 - ink, cv2.DIST_L2, 3)
    if not ink.any():
        return 0.0, 0.0, float('nan'), 0.0
    precision = 100 * (d_to_cmd[ink > 0] <= TOL_MM).mean()
    coverage = 100 * (d_to_ink[cmd > 0] <= TOL_MM).mean()
    mean_err = float(d_to_cmd[ink > 0].mean())
    acc = 2 * precision * coverage / (precision + coverage) if precision + coverage else 0.0
    return coverage, precision, mean_err, acc


def main():
    run = os.path.realpath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(RUNS, 'latest'))
    try:
        meta, area, result, scan = load(run)
    except OSError as e:
        print(f'Missing file: {e.filename}\n'
              'Order: draw_node -> scan_drawing -> shape_recognizer -> run_report')
        return

    ink = ink_mask(scan)
    cmd = commanded_mask(meta, area, ink.shape)
    coverage, precision, mean_err, acc = metrics(cmd, ink)

    asked, found = meta['expected_class'], result['shape']
    exp = GEOMETRY[asked]
    passed = asked == found and acc >= PASS_ACC
    v_found = result['measured_outer_vertices'] if exp['lines'] else result['vertices']
    rows = [('Shape asked', asked),
            ('Shape detected', f"{found} ({result['confidence'] * 100:.0f}%)"),
            ('Vertices asked / found', f"{exp['vertices']} / {v_found}"),
            ('Lines asked / found', f"{exp['lines']} / {result['lines']}"),
            ('Coverage', f'{coverage:.1f} %'),
            ('Precision', f'{precision:.1f} %'),
            ('Mean error', f'{mean_err:.1f} mm'),
            ('Accuracy', f'{acc:.1f} %'),
            ('Result', 'PASS' if passed else 'FAIL')]

    # ---- report.txt
    p = meta['params']
    lines = ['=' * 50,
             f'RUN REPORT  {os.path.basename(run)}',
             '=' * 50,
             f"Time      : {meta['time']}",
             f"Asked     : {meta['shape']}  centre ({p['center_x']:.3f}, {p['center_y']:.3f}) m",
             '-' * 50]
    lines += [f'{k:<24}: {v}' for k, v in rows]
    lines += ['-' * 50, 'Files: scan_topdown.png, scan_result.png, views/', '']
    text = '\n'.join(lines)
    with open(os.path.join(run, 'report.txt'), 'w') as f:
        f.write(text)
    print(text)

    # ---- runs/summary.csv: one line per run
    summary = os.path.join(os.path.dirname(run), 'summary.csv')
    new = not os.path.exists(summary)
    with open(summary, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['run', 'shape_asked', 'shape_detected', 'confidence', 'vertices_asked',
                        'vertices_found', 'coverage', 'precision', 'mean_err_mm', 'accuracy',
                        'pass'])
        w.writerow([os.path.basename(run), asked, found, result['confidence'], exp['vertices'],
                    v_found, round(coverage, 1), round(precision, 1), round(mean_err, 2),
                    round(acc, 1), passed])


if __name__ == '__main__':
    main()