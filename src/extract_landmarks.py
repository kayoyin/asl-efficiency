"""Extract isolated fingerspelled letters from the Google ASL Fingerspelling
Recognition Corpus (Chow et al., 2023).

The corpus ships only landmark sequences (no raw video). For an English
phrase ``P`` of length ``n``, we look for the ``n`` frames whose hand-velocity
local minima are the sharpest -- intuitively the moments where the hand
pauses to form a letter -- and align them to ``P`` in order. Manual
post-correction is then applied (not automated here).

Outputs one JSON file per source split with shape::

    {"language": "ase", "samples": [{"letter": "a", "signer": <id>,
                                     "hand_landmarks": [[x,y,z], ...]},
                                    ...]}

After this, run ``compute_scores.py`` to derive the joint angles and effort
measures the analysis notebook consumes.
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# Indices of the dominant-hand landmarks in the Google corpus' landmark layout.
# The corpus stacks face / pose / left-hand / right-hand. We keep only the
# 21 right-hand landmarks (indices 522:543 in the original column ordering).
RIGHT_HAND_SLICE = slice(522, 543)


def _hand_velocity(landmarks):
    """Mean per-landmark Euclidean velocity between consecutive frames."""
    diffs = np.diff(landmarks, axis=0)
    return np.linalg.norm(diffs, axis=-1).mean(axis=-1)


def _local_minima(values, n):
    """Return indices of the ``n`` deepest local minima of a 1D array."""
    candidates = []
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            candidates.append((values[i], i))
    candidates.sort()
    indices = sorted(idx for _, idx in candidates[:n])
    return indices


def extract_letters(landmarks_dir, label_csv, out_path, split_name):
    """Heuristically segment fingerspelled letters from the corpus.

    Parameters
    ----------
    landmarks_dir : str
        Directory of per-sequence ``.npy`` landmark files.
    label_csv : str
        Path to the corpus' phrase label CSV (columns include ``sequence_id``,
        ``file_id``, ``phrase``).
    out_path : str
        Where to write the resulting JSON file.
    split_name : str
        Subset name to record in the output (e.g. ``"supp0_5"``).
    """
    df = pd.read_csv(label_csv)
    samples = []

    for _, row in df.iterrows():
        seq_id = row["sequence_id"]
        phrase = re.sub(r"[^a-zA-Z]", "", str(row["phrase"])).lower()
        if not phrase:
            continue
        path = os.path.join(landmarks_dir, str(row["file_id"]), f"{seq_id}.npy")
        if not os.path.exists(path):
            continue

        kp = np.load(path)
        kp = kp.reshape((kp.shape[0], 3, -1)).transpose((0, 2, 1))
        hand = kp[:, RIGHT_HAND_SLICE, :]
        hand = np.nan_to_num(hand)

        velocity = _hand_velocity(hand)
        if len(velocity) < len(phrase):
            continue
        frame_idxs = _local_minima(velocity, len(phrase))
        if len(frame_idxs) != len(phrase):
            continue

        for letter, idx in zip(phrase, frame_idxs):
            samples.append(
                {
                    "letter": letter,
                    "signer": str(row.get("participant_id", seq_id)),
                    "img_path": "",
                    "hand_landmarks": hand[idx].tolist(),
                }
            )

    with open(out_path, "w") as f:
        json.dump({"language": "ase", "split": split_name, "samples": samples}, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--landmarks_dir", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--split_name", required=True)
    args = parser.parse_args()
    extract_letters(args.landmarks_dir, args.label_csv, args.out_path, args.split_name)


if __name__ == "__main__":
    main()
