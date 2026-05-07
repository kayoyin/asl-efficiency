"""Articulatory and perceptual effort metrics for handshapes.

Implements the three measures defined in Section 4.2 of Yin et al. (2024):

- ``distance``: angular distance between two handshapes (used both for
  ``handshape distance`` between two letters and ``thumb effort`` against a
  resting hand).
- ``finger_independence``: per-handshape articulatory effort, with thumb
  effort weighted by 2x as in the paper.
- ``handshape_distance``: convenience wrapper for the perceptual measure.

Inputs are joint-angle vectors produced by :class:`src.rotation.Rotation`.
A handshape passed as a dict ``{"hand_landmarks": ...}`` is encoded on the fly.
"""

import json
import os
from collections import defaultdict

import numpy as np
from jax import numpy as jnp

from src.rotation import ONE_HAND_JOINTS, Rotation


# Joints grouped by type across the four non-thumb fingers (MCP / PIP / DIP).
# Pairwise angle differences within a group quantify finger independence.
_FINGER_JOINT_TRIPLES = [
    [(2, 7, 8), (2, 11, 12), (2, 15, 16), (2, 19, 20)],   # MCP
    [(7, 8, 9), (11, 12, 13), (15, 16, 17), (19, 20, 21)],  # PIP
    [(8, 9, 10), (12, 13, 14), (16, 17, 18), (20, 21, 22)],  # DIP
]
FINGER_JOINTS = [
    [ONE_HAND_JOINTS.index(joint) for joint in group]
    for group in _FINGER_JOINT_TRIPLES
]

# Indices of thumb joints in ONE_HAND_JOINTS, used to weight thumb-effort.
_THUMB_JOINT_TRIPLES = [(2, 3, 4), (3, 4, 5), (6, 5, 4)]
THUMB_INDICES = [ONE_HAND_JOINTS.index(joint) for joint in _THUMB_JOINT_TRIPLES]
# A boolean mask over ONE_HAND_JOINTS that selects only thumb angles.
THUMB_WEIGHTS = np.array(
    [1 if i in THUMB_INDICES else 0 for i in range(len(ONE_HAND_JOINTS))],
    dtype=np.float64,
)
THUMB_WEIGHTS = THUMB_WEIGHTS / THUMB_WEIGHTS.sum()


_rotation = Rotation()


def _as_angle_vector(hand):
    """Encode hand landmarks if needed, then return a numpy array of angles."""
    if isinstance(hand, dict):
        hand = _rotation.encode(jnp.array(hand["hand_landmarks"]))
    angles = np.array([float(a) for a in hand])
    return np.nan_to_num(angles)


def distance(hand1, hand2, weights=None):
    """Mean (optionally weighted) angular distance between two handshapes."""
    a = _as_angle_vector(hand1)
    b = _as_angle_vector(hand2)
    diffs = np.abs(a - b)
    diffs = np.where(diffs > np.pi, 2 * np.pi - diffs, diffs)
    if weights is not None:
        return np.average(diffs, weights=weights)
    return np.mean(diffs, dtype=np.float64)


def thumb_effort(hand, resting_hands):
    """Distance from this hand's thumb to its closest resting-hand thumb."""
    return min(distance(hand, r, weights=THUMB_WEIGHTS) for r in resting_hands)


def finger_independence(hand):
    """Articulatory effort score (FI) for a handshape.

    Sums (over joint groups MCP/PIP/DIP) the mean pairwise angular distance
    between fingers, then averages across groups. Thumb effort is added
    separately and weighted 2x, following the paper.
    """
    angles = _as_angle_vector(hand)
    total = 0.0
    for group in FINGER_JOINTS:
        scores = []
        for i, joint_i in enumerate(group):
            for joint_j in group[i + 1 :]:
                diff = abs(angles[joint_i] - angles[joint_j])
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                scores.append(diff)
        total += sum(scores) / len(scores)
    return total / len(FINGER_JOINTS)


def handshape_distance(hand1, hand2):
    """Perceptual effort: angular distance between two handshapes."""
    return distance(hand1, hand2)


def load_resting_hands(angles_dir):
    """Load the resting-hand pool used to compute thumb effort.

    Combines:
      * The mean handshape per Google FS letter for the natural unmarked
        handshapes A, C, S (which look close to a relaxed fist).
      * Manually annotated resting-hand photos in ``resting.json``.
      * The grand mean Google handshape in ``mean.json``.
    """
    resting = []

    with open(os.path.join(angles_dir, "google_mean.json"), "r") as f:
        for sample in json.load(f)["samples"]:
            if sample["letter"] in "acs":
                resting.append(sample["angles"])

    with open(os.path.join(angles_dir, "resting.json"), "r") as f:
        for sample in json.load(f)["samples"]:
            resting.append(sample["angles"])

    with open(os.path.join(angles_dir, "mean.json"), "r") as f:
        resting.append(_rotation.encode(jnp.array(json.load(f))))

    return resting


def compute_scores(angles_path, resting_hands, scores_dir):
    """Compute and dump articulatory + perceptual scores for one split.

    The split is the JSON file under ``data/angles/`` (e.g. ``supp0_5.json``).
    For every signed letter we record per-sample finger independence,
    speaker (resting-hand) distance, thumb-only distance, and pairwise
    listener (handshape) distances.
    """
    with open(angles_path, "r") as f:
        dataset = json.load(f)

    speaker = defaultdict(list)
    thumb = defaultdict(list)
    finger_ind = defaultdict(list)
    listener = defaultdict(list)

    samples = dataset["samples"]
    for sample in samples:
        letter = sample["letter"]
        angles = sample["angles"]
        speaker[letter].append(min(distance(angles, r) for r in resting_hands))
        thumb[letter].append(thumb_effort(angles, resting_hands))
        finger_ind[letter].append(finger_independence(angles))

    for i, s1 in enumerate(samples):
        for s2 in samples[i + 1 :]:
            c1, c2 = s1["letter"], s2["letter"]
            if c1 == c2:
                continue
            key = c1 + c2 if c1 <= c2 else c2 + c1
            listener[key].append(distance(s1["angles"], s2["angles"]))

    os.makedirs(scores_dir, exist_ok=True)
    with open(os.path.join(scores_dir, "speaker_all.json"), "w") as f:
        json.dump(speaker, f)
    with open(os.path.join(scores_dir, "thumb.json"), "w") as f:
        json.dump(thumb, f)
    with open(os.path.join(scores_dir, "finger_ind_all.json"), "w") as f:
        json.dump(finger_ind, f)
    with open(os.path.join(scores_dir, "listener_all.json"), "w") as f:
        json.dump(listener, f)
