#!/usr/bin/python3
"""
test_avoid.py - Checks avoidance.py and run_avoid.py without running Gazebo.

Run it after each function you finish:

    /usr/bin/python3 scripts/test_avoid.py

It stops at the first check that fails or is not written yet. The last check
runs the whole A-to-B move in Python, using your kinematics and your designed
controller, and checks that the arm stays clear of the ball.
"""

import numpy as np

import avoidance as av
from kinematics import fk, skew, space_jacobian

START_POSE = np.array(
    [-0.303918, -0.404112, -0.104118, -2.705802, -0.054971, 2.302898, 0.422461]
)
TOLERANCE = 1e-6


def check_point_now():
    """At zero pose a point must stay put; at other poses it must agree with fk()."""
    p = np.array([0.1, -0.2, 0.3])
    assert np.allclose(av.point_now(p, np.eye(4)), p), (
        "With T = identity, point_now() must return the point unchanged"
    )
    # The hand capsule's end point is 0.0574 m above the end-effector along the
    # hand's own z axis, which points down. fk() gives the same point.
    random = np.random.default_rng(1)
    for _ in range(10):
        q = random.uniform(-2.0, 2.0, 7)
        mine = av.point_now(np.array([0.088, 0.0, 0.88]), av.link_transforms(q)[7])
        expected = (fk(q) @ np.array([0.0, 0.0, -0.0574, 1.0]))[:3]
        assert np.allclose(mine, expected, atol=TOLERANCE), (
            f"point_now() gives {np.round(mine, 4)}, fk() gives {np.round(expected, 4)}"
        )


def check_closest_point():
    """Test a point beside the middle of a segment, and points past each end."""
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    cases = [
        (np.array([0.3, 2.0, 0.0]), np.array([0.3, 0.0, 0.0]), "beside the middle"),
        (np.array([-0.5, 1.0, 0.0]), a, "past the start"),
        (np.array([1.7, 0.0, 1.0]), b, "past the end"),
    ]
    for o, expected, where in cases:
        got = av.closest_point_on_segment(a, b, o)
        assert np.allclose(got, expected), (
            f"For a point {where}, expected {expected}, got {np.round(got, 4)}"
        )


def check_point_jacobian():
    """Compare your point Jacobian with one estimated by nudging each joint."""
    random = np.random.default_rng(2)
    for _ in range(10):
        q = random.uniform(-2.0, 2.0, 7)
        _name, start_zero, end_zero, k, _radius = av.CAPSULES[random.integers(len(av.CAPSULES))]
        p_zero = 0.5 * (start_zero + end_zero)
        c = av.point_now(p_zero, av.link_transforms(q)[k])
        J_c = av.point_jacobian(space_jacobian(q), c, k)
        assert J_c.shape == (3, 7), f"The point Jacobian must be 3x7, got {J_c.shape}"

        step = 1e-6
        numerical = np.zeros((3, 7))
        for i in range(7):
            nudge = np.zeros(7)
            nudge[i] = step
            plus = av.point_now(p_zero, av.link_transforms(q + nudge)[k])
            minus = av.point_now(p_zero, av.link_transforms(q - nudge)[k])
            numerical[:, i] = (plus - minus) / (2.0 * step)
        worst = np.max(np.abs(J_c - numerical))
        assert worst < 1e-5, f"The point Jacobian differs from the numerical one by {worst:.2e}"


def check_avoidance_velocity():
    """No push when far away; half strength halfway inside D_SAFE."""
    random = np.random.default_rng(3)
    J_c = random.normal(size=(3, 7))
    n = np.array([0.0, 0.6, 0.8])
    far = av.avoidance_velocity(av.D_SAFE + 0.01, J_c, n)
    assert np.allclose(far, 0.0), "Beyond D_SAFE the push must be 0"
    half = av.avoidance_velocity(av.D_SAFE / 2, J_c, n)
    expected = av.K_AVOID * 0.5 * J_c.T @ n
    assert np.allclose(half, expected), "At D_SAFE / 2 the push must be K_AVOID * 0.5 * J_c^T n"


def check_designed_controller():
    """The push must not move the hand, and the whole move must avoid the ball."""
    import run_avoid as ra
    from run_basic import basic_controller

    # 1. Same hand motion as the basic controller, because the extra term is in
    #    the null space. Checked at the start pose, where the push is active.
    p_desired, v_desired = ra.desired_motion(1.0)
    q = START_POSE
    p = fk(q)[:3, 3]
    J_s = space_jacobian(q)
    J = np.vstack([J_s[:3], J_s[3:] - skew(p) @ J_s[:3]])
    designed = ra.designed_controller(q, p_desired, v_desired)
    basic = basic_controller(q, p_desired, v_desired)
    assert designed.shape == (7,), f"The controller must return 7 joint speeds, got {designed.shape}"
    assert not np.allclose(designed, basic), (
        "The designed controller gives the same joint speeds as the basic one: the push is missing"
    )
    assert np.allclose(J @ designed, J @ basic, atol=1e-6), (
        "The push changes the hand's motion: it must go through N = I - J+ J"
    )

    # 2. The whole move, simulated in Python.
    q = START_POSE.copy()
    worst_error = 0.0
    worst_clearance = np.inf
    steps = int((ra.MOVE_TIME + ra.HOLD_TIME) / ra.DT)
    for step in range(steps + 1):
        t = step * ra.DT
        p_desired, v_desired = ra.desired_motion(t)
        worst_error = max(worst_error, np.linalg.norm(fk(q)[:3, 3] - p_desired))
        worst_clearance = min(worst_clearance, av.arm_clearance(q)[0])
        qdot = np.clip(ra.designed_controller(q, p_desired, v_desired), -ra.SPEED_LIMIT, ra.SPEED_LIMIT)
        q = np.clip(q + qdot * ra.DT, ra.Q_MIN, ra.Q_MAX)
    print(f"     simulated move: largest tracking error {1000 * worst_error:.2f} mm, "
          f"closest approach {1000 * worst_clearance:.1f} mm")
    assert worst_clearance > 0.05, (
        f"The arm came within {1000 * worst_clearance:.1f} mm of the ball (capsule model)"
    )
    assert worst_error < 0.002, f"The largest tracking error was {1000 * worst_error:.1f} mm"


CHECKS = [
    ("point_now()", check_point_now),
    ("closest_point_on_segment()", check_closest_point),
    ("point_jacobian()", check_point_jacobian),
    ("avoidance_velocity()", check_avoidance_velocity),
    ("designed_controller()", check_designed_controller),
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
    print("All avoidance checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
