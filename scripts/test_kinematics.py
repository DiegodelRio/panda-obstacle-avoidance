#!/usr/bin/python3
"""
test_kinematics.py - Checks your kinematics.py without running Gazebo.

Run it after each function you finish:

    /usr/bin/python3 scripts/test_kinematics.py

It stops at the first check that fails or is not written yet.
"""

import numpy as np

import kinematics as kin

# Two known poses. The expected positions come from the Panda's model files.
HOME = np.array([0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4])
START_POSE = np.array(
    [-0.303918, -0.404112, -0.104118, -2.705802, -0.054971, 2.302898, 0.422461]
)
HOME_POSITION = np.array([0.3069, 0.0, 0.4869])
POINT_A = np.array([0.35, -0.15, 0.25])

# Hand pointing straight down: base x stays, base y and z flip.
HAND_DOWN = np.diag([1.0, -1.0, -1.0])

POSITION_TOLERANCE = 0.001  # m
ROTATION_TOLERANCE = 0.001  # difference in any rotation matrix entry
JACOBIAN_TOLERANCE = 1e-5


def check_screw_axes():
    """Every joint needs a filled-in axis, and each w must be a unit vector."""
    for number, axis in enumerate(kin.SCREW_AXES, start=1):
        length = np.linalg.norm(axis[:3])
        assert abs(length - 1.0) < 1e-9, f"Joint {number}: |w| is {length:.4f}, not 1"
    m_rotation = kin.M[:3, :3]
    assert np.allclose(m_rotation @ m_rotation.T, np.eye(3), atol=1e-6), (
        "The rotation part of M is not a valid rotation matrix"
    )
    assert not np.allclose(kin.M, np.eye(4)), "M is still the identity matrix"


def check_skew():
    """[w] must match the cross product w x b."""
    w = np.array([1.0, 2.0, 3.0])
    b = np.array([-2.0, 0.5, 4.0])
    assert np.allclose(kin.skew(w) @ b, np.cross(w, b)), "[w] b is not equal to w x b"


def check_exp_screw():
    """Test a zero angle, then a half turn about a line through (1, 0, 0)."""
    axis = np.array([0.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    assert np.allclose(kin.exp_screw(axis, 0.0), np.eye(4)), (
        "A zero angle must give the identity matrix"
    )
    # A half turn about that line carries the origin to (2, 0, 0).
    half_turn = kin.exp_screw(axis, np.pi)
    assert np.allclose(half_turn[:3, 3], [2.0, 0.0, 0.0], atol=1e-9), (
        f"Half turn moved the origin to {np.round(half_turn[:3, 3], 4)}"
    )


def check_pose(name, q, expected_position):
    """Compare FK with a known end-effector position and the hand-down rotation."""
    T = kin.fk(q)
    gap = np.linalg.norm(T[:3, 3] - expected_position)
    assert gap < POSITION_TOLERANCE, (
        f"{name}: FK gives {np.round(T[:3, 3], 4)}, expected {expected_position}"
    )
    assert np.allclose(T[:3, :3], HAND_DOWN, atol=ROTATION_TOLERANCE), (
        f"{name}: the hand should point straight down, FK gives\n"
        f"{np.round(T[:3, :3], 4)}"
    )


def check_fk():
    check_pose("Home pose", HOME, HOME_POSITION)
    check_pose("Start pose", START_POSE, POINT_A)


def numerical_jacobian(q, step=1e-5):
    """Estimate the space Jacobian by nudging each joint in FK.

    (dT/dq) * T^-1 is a 4x4 matrix holding the twist of that joint.
    """
    columns = []
    T_inverse = np.linalg.inv(kin.fk(q))
    for i in range(7):
        nudge = np.zeros(7)
        nudge[i] = step
        dT = (kin.fk(q + nudge) - kin.fk(q - nudge)) / (2.0 * step)
        twist = dT @ T_inverse
        w = [twist[2, 1], twist[0, 2], twist[1, 0]]
        v = twist[:3, 3]
        columns.append(np.r_[w, v])
    return np.array(columns).T


def check_jacobian():
    """Compare your Jacobian with the numerical one at random poses."""
    random = np.random.default_rng(0)
    for _ in range(20):
        q = random.uniform(-2.0, 2.0, 7)
        J = kin.space_jacobian(q)
        assert J.shape == (6, 7), f"The Jacobian must be 6x7, got {J.shape}"
        worst = np.max(np.abs(J - numerical_jacobian(q)))
        assert worst < JACOBIAN_TOLERANCE, (
            f"The Jacobian differs from the numerical one by {worst:.2e}"
        )


CHECKS = [
    ("Screw axes and M", check_screw_axes),
    ("skew()", check_skew),
    ("exp_screw()", check_exp_screw),
    ("fk()", check_fk),
    ("space_jacobian()", check_jacobian),
]


def main():
    for name, check in CHECKS:
        try:
            check()
        except NotImplementedError as error:
            print(f"STOPPED at {name}: {error}")
            return 1
        except AssertionError as error:
            print(f"FAIL {name}: {error}")
            return 1
        print(f"PASS {name}")
    print("All kinematics checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
