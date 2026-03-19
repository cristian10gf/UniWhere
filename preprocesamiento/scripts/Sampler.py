#!/usr/bin/env python3
"""
Keyframe selector for COLMAP.
Two strategies:
  diverse  (default): greedy max visual diversity + sharpness bonus.
             Good for scenes with very different areas.
  uniform:  temporally uniform sampling after sharpness filter.
             Guarantees overlap between consecutive frames — much better
             for COLMAP when frames come from a video sequence.
"""

import os
import sys
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path

def compute_sharpness(img_gray):
    """Laplacian variance — higher = sharper."""
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()

def compute_histogram(img_gray, bins=64):
    """Normalized grayscale histogram."""
    hist = cv2.calcHist([img_gray], [0], None, [bins], [0, 256])
    hist = hist.flatten()
    hist /= hist.sum() + 1e-8
    return hist

def histogram_distance(h1, h2):
    """Chi-squared histogram distance."""
    diff = h1 - h2
    denom = h1 + h2 + 1e-8
    return float(np.sum(diff * diff / denom))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir",  help="Directory with input frames")
    parser.add_argument("output_dir", help="Directory to copy selected keyframes")
    parser.add_argument("--num-frames", type=int, default=300)
    parser.add_argument("--min-sharpness", type=float, default=50.0,
                        help="Minimum Laplacian variance to consider a frame (filters blur)")
    parser.add_argument("--resize", type=int, default=320,
                        help="Resize width for fast processing (0 = no resize)")
    parser.add_argument("--strategy", choices=["diverse", "uniform"], default="diverse",
                        help="diverse: max visual diversity (default). uniform: even temporal spacing (better for video COLMAP).")
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load and sort all frames ---
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    frames = sorted([f for f in input_dir.iterdir() if f.suffix.lower() in extensions])
    total = len(frames)
    print(f"Found {total} frames in {input_dir}")

    # --- Pass 1: compute sharpness for all frames (fast, resized) ---
    print("Computing sharpness scores...")
    sharpness = []
    histograms = []
    valid_frames = []

    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        if args.resize > 0:
            h, w = img.shape
            scale = args.resize / max(w, h)
            if scale < 1.0:
                img = cv2.resize(img, (int(w * scale), int(h * scale)))

        s = compute_sharpness(img)
        if s < args.min_sharpness:
            continue  # skip blurry frames

        sharpness.append(s)
        histograms.append(compute_histogram(img))
        valid_frames.append(fpath)

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{total}] sharp frames so far: {len(valid_frames)}")

    print(f"Sharp frames (sharpness >= {args.min_sharpness}): {len(valid_frames)}")

    sharpness = np.array(sharpness)
    histograms = np.array(histograms)

    if len(valid_frames) <= args.num_frames:
        print(f"Not enough frames after filtering — using all {len(valid_frames)} sharp frames.")
        selected_idx = list(range(len(valid_frames)))
    elif args.strategy == "uniform":
        # --- Uniform temporal sampling ---
        # Pick N frames at evenly-spaced indices from the sharp frame pool.
        # This maximises temporal overlap between consecutive frames,
        # which is what COLMAP needs to chain a long video sequence.
        print(f"Selecting {args.num_frames} keyframes via uniform temporal sampling...")
        indices = np.linspace(0, len(valid_frames) - 1, args.num_frames, dtype=int)
        selected_idx = sorted(set(indices.tolist()))
        print(f"  Temporal step between frames: ~{len(valid_frames)/args.num_frames:.1f} sharp frames")
    else:
        # --- Greedy max-diversity selection ---
        print(f"Selecting {args.num_frames} keyframes via greedy max-diversity...")
        sharp_norm = (sharpness - sharpness.min()) / (sharpness.max() - sharpness.min() + 1e-8)

        selected_idx = [int(np.argmax(sharp_norm))]
        min_dist = np.zeros(len(valid_frames))
        for i in range(len(valid_frames)):
            min_dist[i] = histogram_distance(histograms[i], histograms[selected_idx[0]])

        for step in range(1, args.num_frames):
            scores = min_dist * (0.7 + 0.3 * sharp_norm)
            scores[selected_idx] = -1
            best = int(np.argmax(scores))
            selected_idx.append(best)
            for i in range(len(valid_frames)):
                d = histogram_distance(histograms[i], histograms[best])
                if d < min_dist[i]:
                    min_dist[i] = d
            if (step + 1) % 50 == 0:
                print(f"  Selected {step+1}/{args.num_frames}...")

        selected_idx.sort()

    selected = [valid_frames[i] for i in selected_idx]

    # --- Copy selected frames ---
    print(f"\nCopying {len(selected)} keyframes to {output_dir}...")
    for i, fpath in enumerate(selected):
        dst = output_dir / fpath.name
        shutil.copy2(str(fpath), str(dst))

    print(f"\n✅ Done! Selected {len(selected)} keyframes.")
    print(f"   Sharpness range: {sharpness[selected_idx].min():.1f} – {sharpness[selected_idx].max():.1f}")
    print(f"   Output: {output_dir}")

if __name__ == "__main__":
    main()
