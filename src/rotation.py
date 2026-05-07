"""Convert MediaPipe hand landmarks to a translation/scale-invariant
joint-angle representation.

For each adjacent triple of MediaPipe hand landmarks (a, b, c) listed in
``ONE_HAND_JOINTS``, we compute the 3D angle between vectors b->a and b->c.
The resulting vector of joint angles is what the paper uses as the canonical
"angular representation" of a handshape.
"""

import jax
from jax import numpy as jnp


# Triples (a, b, c) of MediaPipe landmark indices (offset by +2 to leave room
# for the two dummy points used by the encoder) defining the joints whose
# angles characterize a handshape. Indices 0/1 are placeholders for the wrist
# orientation reference frame.
ONE_HAND_JOINTS = [
    (1, 0, 2), (0, 2, 7), (7, 2, 19), (19, 2, 3), (2, 3, 4),
    (3, 4, 5), (6, 5, 4), (2, 7, 8), (8, 7, 11), (7, 8, 9),
    (8, 9, 10), (7, 11, 12), (12, 11, 15), (11, 12, 13),
    (12, 13, 14), (16, 15, 19), (19, 15, 11), (15, 16, 17),
    (16, 17, 18), (20, 19, 2), (2, 19, 15), (19, 20, 21),
    (20, 21, 22), (2, 11, 12), (2, 15, 16), (2, 19, 20),
]


HAND_PALM_CONNECTIONS = [(0, 1), (0, 5), (9, 13), (13, 17), (5, 9), (0, 17)]
HAND_THUMB_CONNECTIONS = [(1, 2), (2, 3), (3, 4)]
HAND_INDEX_CONNECTIONS = [(5, 6), (6, 7), (7, 8)]
HAND_MIDDLE_CONNECTIONS = [(9, 10), (10, 11), (11, 12)]
HAND_RING_CONNECTIONS = [(13, 14), (14, 15), (15, 16)]
HAND_PINKY_CONNECTIONS = [(17, 18), (18, 19), (19, 20)]
HAND_CONNECTIONS = (
    HAND_PALM_CONNECTIONS
    + HAND_THUMB_CONNECTIONS
    + HAND_INDEX_CONNECTIONS
    + HAND_MIDDLE_CONNECTIONS
    + HAND_RING_CONNECTIONS
    + HAND_PINKY_CONNECTIONS
)


def get_lengths(frame):
    """Return per-bone lengths for a frame of (22, 3) hand landmarks."""
    hand_lengths = [1.0, jnp.linalg.norm(jnp.array([0.0, 0.0, 0.0]) - frame[2])]
    for (a, b) in HAND_CONNECTIONS:
        hand_lengths.append(jnp.linalg.norm(frame[a + 2] - frame[b + 2]))
    return jnp.array(hand_lengths)


v_get_lengths = jax.vmap(jax.vmap(get_lengths, 0, 0), 0, 0)


def se3_to_so3(data, joints, angle_only=True):
    """Compute the angle (or rotation) between vectors b->a and b->c."""
    data = jnp.array(data)
    a = data[joints[0]]
    b = data[joints[1]]
    c = data[joints[2]]

    ba = a - b
    bc = c - b

    ba_mag = jnp.linalg.norm(ba)
    bc_mag = jnp.linalg.norm(bc)

    ba_norm = jax.lax.cond(
        ba_mag > 0,
        lambda x: x / ba_mag,
        lambda x: jnp.zeros_like(x).astype(jnp.float32),
        ba,
    )
    bc_norm = jax.lax.cond(
        bc_mag > 0,
        lambda x: x / bc_mag,
        lambda x: jnp.zeros_like(x).astype(jnp.float32),
        bc,
    )

    v = jnp.cross(ba_norm, bc_norm)
    cos_theta = jnp.dot(ba_norm, bc_norm)
    vx = jnp.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])

    def _check_colinear(c):
        return jax.lax.cond(
            c == -1,
            lambda x: jnp.array([[-1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]]),
            lambda x: jnp.eye(3) + x[0] + jnp.dot(x[0], x[0]) * (1 / (1 + x[1])),
            [vx, c],
        )

    r = jax.lax.cond(
        jnp.all(v == 0),
        lambda x: jnp.eye(3).astype(jnp.float32),
        _check_colinear,
        cos_theta,
    )

    if angle_only:
        return jnp.arccos((jnp.trace(r) - 1) / 2)
    return r


_to_r6 = jax.vmap(
    jax.vmap(jax.vmap(se3_to_so3, (None, 0), 0), (0, None), 0),
    (0, None),
    0,
)


class Rotation(object):
    """Encode a 3D hand pose as a vector of joint angles."""

    def __init__(self, angle_only=True):
        self.angle_only = angle_only

    def encode(self, x):
        """Convert (21, 3) MediaPipe landmarks to a flat vector of joint angles."""
        x = jnp.array([[x]])
        batch_size, seq_len = x.shape[:2]
        dummy_points = jnp.array([[0, 0, 0], [1, 0, 0]])
        dummy_points = jnp.repeat(
            jnp.repeat(
                dummy_points[jnp.newaxis, jnp.newaxis, :], batch_size, axis=0
            ),
            seq_len,
            axis=1,
        )
        x = jnp.concatenate([dummy_points, x], axis=2)
        joints = jnp.array(ONE_HAND_JOINTS)
        if self.angle_only:
            return _to_r6(x, joints)[0][0]
        hand_lengths = v_get_lengths(x)
        return _to_r6(x, joints), jnp.array(hand_lengths)
