#!/usr/bin/python3
"""
check_motion.py - Step 2 check for the Gazebo Panda setup.

Tests the two connections every later controller needs:
    READ:  joint angles arrive from Gazebo in Python
    WRITE: targets sent from Python move the arm

The test moves the arm to its start pose at point A, swings joint 1 back and
forth, returns to the start pose, and prints PASS or FAIL. Start bash run_sim.sh first, then run this in a
second terminal:

    /usr/bin/python3 scripts/check_motion.py
"""

import math
import sys
import time

from gz_panda import GzPanda

# Start pose [rad]. It matches initial_position in model.sdf and puts the
# end-effector at point A with the hand pointing straight down.
START_POSE = [-0.303918, -0.404112, -0.104118, -2.705802, -0.054971, 2.302898, 0.422461]

# How often this script reads the arm and sends targets.
LOOP_HZ = 100.0
DT = 1.0 / LOOP_HZ

# Joint 1 swing: target = start + AMPLITUDE * sin(2 * pi * FREQUENCY * t)
# The amplitude is small enough to keep the arm clear of the ball.
AMPLITUDE = 0.3  # rad
FREQUENCY = 0.25  # Hz
SWING_TIME = 8.0  # s, which covers two full swings

# Pass limits.
SETTLED_TOLERANCE = 0.01  # rad, how close every joint must get to the start pose
SETTLE_TIMEOUT = 8.0  # s, the longest allowed time to reach the start pose
MAX_TRACKING_ERROR = 0.05  # rad, the largest allowed target-to-angle gap
MIN_SWING_RANGE = 0.5  # rad, the smallest joint 1 motion that counts as moving


def go_to_start(panda):
    """Hold the start pose target until all joints arrive. Return True on success."""
    deadline = time.monotonic() + SETTLE_TIMEOUT
    while time.monotonic() < deadline:
        panda.send(START_POSE)
        q, _ = panda.read()
        worst = max(abs(a - b) for a, b in zip(q, START_POSE))
        if worst < SETTLED_TOLERANCE:
            return True
        time.sleep(DT)
    return False


def swing_joint_1(panda):
    """Swing joint 1 and return (largest tracking error, range of motion)."""
    worst_error = 0.0
    lowest = math.inf
    highest = -math.inf

    # The arm begins at the start pose, so that is the first target to compare.
    target = list(START_POSE)

    start = time.monotonic()
    while True:
        t = time.monotonic() - start
        if t >= SWING_TIME:
            break

        # Compare the angle with the target sent on the previous pass. That
        # target is the one the arm has had time to act on.
        q, _ = panda.read()
        worst_error = max(worst_error, abs(q[0] - target[0]))
        lowest = min(lowest, q[0])
        highest = max(highest, q[0])

        # Build this pass's target: the start pose on every joint except joint 1.
        target = list(START_POSE)
        target[0] = START_POSE[0] + AMPLITUDE * math.sin(2.0 * math.pi * FREQUENCY * t)
        panda.send(target)
        time.sleep(DT)

    return worst_error, highest - lowest


def main():
    panda = GzPanda()

    # READ test: fail early with a useful message if nothing arrives.
    print("Waiting for joint states from Gazebo...")
    if not panda.wait_for_state(timeout=5.0):
        print("FAIL: no joint state arrived after 5 s.")
        print("Check that bash run_sim.sh is running and the world is not paused.")
        print("List the live topics with: gz topic -l | grep panda")
        return 1

    q, _ = panda.read()
    print("READ OK. Joint angles [rad]:", [round(value, 3) for value in q])

    # gz-transport needs a moment to connect the publishers to Gazebo. go_to_start
    # keeps resending, so a target lost during this time does no harm.
    print("Moving to the start pose...")
    if not go_to_start(panda):
        q, _ = panda.read()
        print("FAIL: the arm did not reach the start pose.")
        print("Final angles [rad]:", [round(value, 3) for value in q])
        return 1
    print("Start pose reached.")

    # WRITE test: joint 1 must follow a moving target.
    print(f"Swinging joint 1 for {SWING_TIME:.0f} s...")
    worst_error, swing_range = swing_joint_1(panda)
    print(f"Largest tracking error: {worst_error:.4f} rad")
    print(f"Joint 1 range of motion: {swing_range:.3f} rad")

    returned = go_to_start(panda)

    passed = (
        returned
        and worst_error < MAX_TRACKING_ERROR
        and swing_range > MIN_SWING_RANGE
    )
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
