import argparse
import json
import os
import pickle
from glob import glob

import numpy as np
from tqdm import tqdm

COCO17_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8),
                (9, 10), (11, 12), (13, 14), (15, 16)]


def load_pose_sequence(sequence_dir):
    frames = []
    for path in sorted(glob(os.path.join(sequence_dir, '*.json'))):
        with open(path) as stream:
            record = json.load(stream)
        if not record.get('people'):
            continue
        values = record['people'][0].get('pose_keypoints_2d', [])
        if len(values) == 51:
            frames.append(np.asarray(values, dtype=np.float32).reshape(17, 3))
    return np.asarray(frames, dtype=np.float32)


def frame_metrics(pose):
    valid = pose[:, 2] > 0
    if valid.sum() < 2:
        return np.full((8, 3), np.nan, dtype=np.float32)
    xy = pose[:, :2].copy()
    height = np.ptp(xy[valid, 1])
    if height <= 1e-6:
        return np.full((8, 3), np.nan, dtype=np.float32)
    hip_valid = valid[11] and valid[12]
    center = xy[[11, 12]].mean(0) if hip_valid else xy[valid].mean(0)
    xy = (xy - center) * (128.0 / height)
    metrics = np.full((8, 3), np.nan, dtype=np.float32)
    for index, (left, right) in enumerate(COCO17_PAIRS):
        if not (valid[left] and valid[right]):
            continue
        lx, ly = xy[left]
        rx, ry = xy[right]
        metrics[index] = (
            abs(ly - ry),
            abs((lx + rx) / 2.0),
            abs(np.arctan2(ly - ry, lx - rx)),
        )
    return metrics


def robust_sequence_pav(poses):
    values = np.asarray([frame_metrics(frame) for frame in poses])
    result = np.zeros((8, 3), dtype=np.float32)
    for pair in range(8):
        for metric in range(3):
            column = values[:, pair, metric]
            column = column[np.isfinite(column)]
            if column.size == 0:
                continue
            q1, q3 = np.percentile(column, [25, 75])
            iqr = q3 - q1
            kept = column[(column >= q1 - 1.5 * iqr) &
                          (column <= q3 + 1.5 * iqr)]
            result[pair, metric] = (kept if kept.size else column).mean()
    return result


def discover(pose_root, heatmap_root):
    records = []
    for sequence_dir in sorted(glob(os.path.join(pose_root, '*', '*', '*'))):
        if not os.path.isdir(sequence_dir):
            continue
        rel = os.path.relpath(sequence_dir, pose_root)
        view = os.path.basename(sequence_dir)
        heatmap = os.path.join(heatmap_root, rel, view + '.pkl')
        if os.path.isfile(heatmap):
            records.append((rel, sequence_dir, heatmap))
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pose_root', required=True)
    parser.add_argument('--heatmap_root', required=True)
    parser.add_argument('--partition', required=True)
    parser.add_argument('--output_root', required=True)
    args = parser.parse_args()

    with open(args.partition) as stream:
        train_ids = set(json.load(stream)['TRAIN_SET'])
    records = discover(args.pose_root, args.heatmap_root)
    computed = []
    for rel, pose_dir, heatmap in tqdm(records, desc='Computing PAV'):
        poses = load_pose_sequence(pose_dir)
        if len(poses):
            computed.append((rel, heatmap, len(poses),
                             robust_sequence_pav(poses)))

    train_pavs = np.stack([
        pav for rel, _, _, pav in computed
        if rel.split(os.sep)[0] in train_ids
    ])
    minimum = train_pavs.min(axis=0)
    scale = np.maximum(train_pavs.max(axis=0) - minimum, 1e-6)

    for rel, heatmap, length, pav in tqdm(computed, desc='Writing DRF data'):
        destination = os.path.join(args.output_root, rel)
        os.makedirs(destination, exist_ok=True)
        map_link = os.path.join(destination, '0_heatmap.pkl')
        if not os.path.lexists(map_link):
            os.symlink(os.path.abspath(heatmap), map_link)
        normalized = np.clip((pav - minimum) / scale, 0.0, 1.0)
        repeated = np.repeat(normalized[None], length, axis=0)
        with open(os.path.join(destination, '1_pav.pkl'), 'wb') as stream:
            pickle.dump(repeated.astype(np.float32), stream)
    print('Wrote {} Skeleton Map + PAV sequences to {}'.format(
        len(computed), args.output_root))


if __name__ == '__main__':
    main()
