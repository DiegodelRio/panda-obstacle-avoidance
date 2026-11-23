"""
avoidance.py - Distance from the arm to the ball, and the push that moves the
arm away from it.

The arm is modeled as 5 capsules. A capsule is a line segment with a radius,
so its surface is every point within that radius of the segment. Each capsule's
end points are fixed to one link, so they move with the product of
exponentials up to that link, the same math as fk().

This file uses only NumPy and kinematics.py, so test_avoid.py can check it
without the simulator running.
"""

import numpy as np

from kinematics import SCREW_AXES, exp_screw, skew

# The obstacle. These match the red ball in panda_world.sdf.
BALL_CENTER = np.array([0.26, 0.12, 0.66])  # m
BALL_RADIUS = 0.08  # m

# The push starts when the arm is closer than D_SAFE to the ball surface.
D_SAFE = 0.30  # m
# Strength of the push [rad/s per m].
K_AVOID = 20.0

# The arm as 5 capsules:
#   (name, start point at zero pose [m], end point at zero pose [m],
#    number of joints that move it, radius [m])
# Both end points of a capsule move with the same joints, because each end point
# lies on the axis of any joint between its own link and the capsule's link.
CAPSULES = [
    ("upper arm", np.array([0.0, 0.0, 0.333]), np.array([0.0825, 0.0, 0.649]), 3, 0.075),
    ("elbow", np.array([0.0825, 0.0, 0.649]), np.array([0.0, 0.0, 0.774]), 4, 0.075),
    ("forearm", np.array([0.0, 0.0, 0.774]), np.array([0.0, 0.0, 1.033]), 5, 0.07),
    ("wrist", np.array([0.0, 0.0, 1.033]), np.array([0.088, 0.0, 1.033]), 6, 0.07),
    ("hand", np.array([0.088, 0.0, 1.033]), np.array([0.088, 0.0, 0.88]), 7, 0.07),
]


def link_transforms(q):
    """Return the 8 partial products of exponentials at q.

    transforms[k] = e^[S1]q1 * ... * e^[Sk]qk, the motion of everything moved by
    joints 1 to k. transforms[0] is the identity, because nothing has moved yet.
    """
    transforms = [np.eye(4)]
    for i in range(7):
        transforms.append(transforms[-1] @ exp_screw(SCREW_AXES[i], q[i]))
    return transforms


def point_now(p_zero, T):
    """Return where a point fixed to a link is now.

    p_zero is the point's position at zero pose, and T is that link's partial
    product from link_transforms(). Write the point in homogeneous coordinates
    (x, y, z, 1), multiply by T, and keep the first 3 numbers.
    """
    p_h = np.append(p_zero, 1.0)    # (x, y, z) becomes (x, y, z, 1)
    return (T @ p_h)[:3]            # move the point, then keep x, y, z


def closest_point_on_segment(a, b, o):
    """Return the point on the segment from a to b that is closest to point o.

    Project o onto the line through a and b:
        t = ((o - a) . (b - a)) / |b - a|^2
    Clip t to the range 0 to 1 so the point stays on the segment, then return
        c = a + t (b - a)
    """
    d = b - a                               # direction along the segment
    t = np.dot(o - a, d) / np.dot(d, d)     # how far along the segment o projects
    t = np.clip(t, 0.0, 1.0)                # stay between a and b
    return a + t * d                        # the closest point on the segment


def arm_clearance(q):
    """Return (clearance, c, n, k) for the capsule closest to the ball.

    clearance: gap between the arm surface and the ball surface [m]. Negative
               means they overlap.
    c:         the closest point on that capsule's segment, in the base frame.
    n:         unit vector from the ball center toward c.
    k:         number of joints that move that capsule.
    """
    transforms = link_transforms(q)
    best = None

    for _name, start_zero, end_zero, k, radius in CAPSULES:
        # Move the capsule's end points to where they are now.
        a = point_now(start_zero, transforms[k])
        b = point_now(end_zero, transforms[k])

        # Closest point on the segment, and the gap between the two surfaces.
        c = closest_point_on_segment(a, b, BALL_CENTER)
        center_gap = np.linalg.norm(c - BALL_CENTER)
        clearance = center_gap - radius - BALL_RADIUS

        # Keep the capsule that is closest to the ball.
        if best is None or clearance < best[0]:
            n = (c - BALL_CENTER) / center_gap
            best = (clearance, c, n, k)

    return best


def point_jacobian(J_s, c, k):
    """Return the 3x7 Jacobian of point c, fixed to a link moved by joints 1 to k.

    It works like the hand's Jacobian in the basic controller: c_dot = v_s - [c] w.
    Use the first k columns of the space Jacobian J_s, and leave the other
    columns as 0, because joints after k do not move this link.
    """
    J_c = np.zeros((3, 7))              # 3 rows (x, y, z speed of c), 7 joints, all 0 to start
    J_w = J_s[:3, :k]                   # angular part (w), joints 1 to k
    J_v = J_s[3:, :k]                   # linear part (v_s), joints 1 to k
    J_c[:, :k] = J_v - skew(c) @ J_w    # c_dot = v_s - [c] w
    return J_c


def avoidance_velocity(clearance, J_c, n):
    """Return the 7 joint speeds that push point c away from the ball.

        qdot_avoid = K_AVOID * max(0, D_SAFE - clearance) / D_SAFE * J_c^T n

    It is 0 when the arm is farther than D_SAFE from the ball, and grows as the
    arm gets closer.
    """
    push = K_AVOID * max(0.0, D_SAFE - clearance) / D_SAFE   # 0 when far, grows when close
    return push * (J_c.T @ n)                                # joint speeds that move c along n
