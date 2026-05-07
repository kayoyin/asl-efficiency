"""Compute joint angles and effort scores for one or more handshape splits.

Pipeline:
    1. Load a JSON file of MediaPipe hand landmarks (the output of
       ``extract_landmarks.py``).
    2. Encode each sample into a vector of joint angles using
       :class:`src.rotation.Rotation`. Write ``data/angles/<split>.json``.
    3. Compute speaker / thumb / finger-independence / listener scores using
       :mod:`src.effort`. Write ``data/scores/<split>/*.json``.

Usage::

    python -m src.compute_scores \\
        --in_path data/google/supp0_5.json \\
        --split_name supp0_5 \\
        --angles_dir data/angles \\
        --scores_dir data/scores/supp0_5
"""

import argparse
import json
import os

from jax import numpy as jnp

from src.effort import compute_scores, load_resting_hands
from src.rotation import Rotation


def encode_split(in_path, out_path):
    """Read landmarks, encode joint angles, write the augmented JSON."""
    with open(in_path, "r") as f:
        dataset = json.load(f)

    rot = Rotation()
    samples = []
    for sample in dataset["samples"]:
        angles = rot.encode(jnp.array(sample["hand_landmarks"]))
        sample["angles"] = [float(a) for a in angles]
        samples.append(sample)
    dataset["samples"] = samples

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dataset, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_path", required=True,
                        help="Landmarks JSON produced by extract_landmarks.py")
    parser.add_argument("--split_name", required=True)
    parser.add_argument("--angles_dir", default="data/angles")
    parser.add_argument("--scores_dir", default=None,
                        help="Defaults to data/scores/<split_name>")
    args = parser.parse_args()

    angles_path = os.path.join(args.angles_dir, f"{args.split_name}.json")
    scores_dir = args.scores_dir or os.path.join("data", "scores", args.split_name)

    encode_split(args.in_path, angles_path)
    resting = load_resting_hands(args.angles_dir)
    compute_scores(angles_path, resting, scores_dir)


if __name__ == "__main__":
    main()
