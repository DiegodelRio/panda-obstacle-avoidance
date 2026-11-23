"""
kinematics.py - Forward kinematics and Jacobian of the Panda.

Uses the product of exponentials (PoE) with screw axes in the base frame:

    T(q) = e^[S1]q1 * e^[S2]q2 * ... * e^[S7]q7 * M

This file uses only NumPy and has no Gazebo code, so test_kinematics.py can check
it without the simulator running.
"""

import numpy as np

# Screw axes with the arm at zero pose, one row per joint.
# Each row is (wx, wy, wz, vx, vy, vz), where v = -w x p.
SCREW_AXES = np.array([
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Joint 1
    [0.0, 1.0, 0.0, -0.333, 0.0, 0.0],  # Joint 2
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Joint 3
    [0.0, -1.0, 0.0, 0.649, 0.0, -0.0825],  # Joint 4
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Joint 5
    [0.0, -1.0, 0.0, 1.033, 0.0, 0.0],  # Joint 6
    [0.0, 0.0, -1.0, 0.0, 0.088, 0.0],  # Joint 7
])

# End-effector pose at zero pose: panda_hand_tcp in the base frame.
c = np.cos(np.pi / 4)  # √2/2

M = np.array([
    [c, c, 0.0, 0.088],
    [c, -c, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.8226],
    [0.0, 0.0, 0.0, 1.0],
])


def skew(w):
    """Return the 3x3 skew-symmetric matrix [w] of a 3-vector w."""
    return np.array([
        [0.0, -w[2], w[1]],
        [w[2], 0.0, -w[0]],
        [-w[1], w[0], 0.0],
    ])


def exp_screw(S, theta):
    """Return the 4x4 matrix e^[S]theta for a screw axis S and an angle theta."""
    w = S[:3]
    v = S[3:]
    W = skew(w)

    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * (W @ W)

    G = np.eye(3) * theta + (1 - np.cos(theta)) * W + (theta - np.sin(theta)) * (W @ W)
    p = G @ v

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def fk(q):
    """Return the 4x4 end-effector pose T(q) for seven joint angles q."""
    T = np.eye(4)
    for i in range(7):
        T = T @ exp_screw(SCREW_AXES[i], q[i])
    return T @ M


def adjoint(T):
    """Return the 6x6 adjoint matrix Ad_T of a 4x4 transform T."""
    R = T[:3, :3]
    p = T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = skew(p) @ R
    return A


def space_jacobian(q):
    """Return the 6x7 space Jacobian at q.

    Row order is (w, v), matching SCREW_AXES. Column i is screw axis i moved to
    the current pose by joints 1 to i-1.
    """
    J = np.zeros((6, 7))
    T = np.eye(4)
    for i in range(7):
        J[:, i] = adjoint(T) @ SCREW_AXES[i]
        T = T @ exp_screw(SCREW_AXES[i], q[i])
    return J
