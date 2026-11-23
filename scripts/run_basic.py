#!/usr/bin/python3
"""
run_basic.py - Moves the end-effector from A to B with the basic controller.

Each pass of the loop does this:
    1. Read the joint angles from Gazebo.
    2. Work out where the end-effector should be right now.
    3. Turn that into joint speeds with your controller.
    4. Add those speeds to the joint targets and send the targets to Gazebo.

Restart the simulator before each run so the arm begins at point A:

    Terminal 1: bash run_sim.sh
    Terminal 2: /usr/bin/python3 scripts/run_basic.py

Each run saves its data to results/basic_run.csv for your plots.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from gz_panda import GzPanda
from kinematics import fk, skew, space_jacobian

# The end-effector path [m]. These match the dots in panda_world.sdf.
POINT_A = np.array([0.35, -0.15, 0.25])
POINT_B = np.array([0.60, 0.15, 0.35])

# Joint angles [rad] that put the end-effector at point A with the hand down.
START_POSE = np.array(
    [-0.303918, -0.404112, -0.104118, -2.705802, -0.054971, 2.302898, 0.422461]
)

# Timing.
MOVE_TIME = 8.0  # s, time to go from A to B
HOLD_TIME = 1.0  # s, time to stay at B after the move
LOOP_HZ = 100.0
DT = 1.0 / LOOP_HZ

# Hand orientation to hold during the whole move: pointing straight down.
# Columns: hand x along base x, hand y along base -y, hand z along base -z.
HAND_DOWN = np.diag([1.0, -1.0, -1.0])

# Feedback gains [1/s]. They pull the hand back onto the path if it drifts.
K_POSITION = 4.0
K_ORIENTATION = 4.0

# Real Panda joint limits [rad]. Joint 4 uses its real upper limit here, which is
# tighter than the one in model.sdf.
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Largest joint speed sent to Gazebo [rad/s]. model.sdf caps it at 1.5.
SPEED_LIMIT = 1.5

# How far the arm may be from START_POSE when a run begins [rad].
START_TOLERANCE = 0.02

RESULTS_FILE = Path(__file__).resolve().parent.parent / "results" / "basic_run.csv"


def desired_motion(t):
    """Return the desired end-effector position and velocity at time t."""
    # tau goes from 0 at the start to 1 at the end of the move, then stays at 1.
    tau = min(t / MOVE_TIME, 1.0)

    # Quintic time scaling: s goes from 0 to 1 with zero speed and zero
    # acceleration at both ends, so the arm starts and stops smoothly.
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / MOVE_TIME
    if t >= MOVE_TIME:
        s_dot = 0.0

    p_desired = POINT_A + s * (POINT_B - POINT_A)
    v_desired = s_dot * (POINT_B - POINT_A)
    return p_desired, v_desired


def orientation_error(R, R_desired):
    """Return the small rotation vector that turns R toward R_desired."""
    error = np.zeros(3)
    for i in range(3):
        error += np.cross(R[:, i], R_desired[:, i])
    return 0.5 * error


def basic_controller(q, p_desired, v_desired):
    """Return seven joint speeds [rad/s] that track the desired motion."""
    # Where the hand is now.
    T = fk(q)
    R = T[:3, :3]
    p = T[:3, 3]

    # Space Jacobian: twist = J_s @ qdot, with rows (w, v_s).
    J_s = space_jacobian(q)
    J_w = J_s[:3, :]
    J_vs = J_s[3:, :]

    # Change v_s to the velocity of the hand point itself: p_dot = v_s - [p] w.
    J = np.vstack([J_w, J_vs - skew(p) @ J_w])

    # Desired hand motion, plus feedback that corrects any drift.
    w_command = K_ORIENTATION * orientation_error(R, HAND_DOWN)
    v_command = v_desired + K_POSITION * (p_desired - p)
    twist = np.concatenate([w_command, v_command])

    # Pseudoinverse (Lecture 16): J+ = J^T (J J^T)^-1
    J_pinv = J.T @ np.linalg.inv(J @ J.T)
    return J_pinv @ twist


def save_log(rows):
    """Write one line per loop pass to RESULTS_FILE."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with RESULTS_FILE.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["t", "x_des", "y_des", "z_des", "x", "y", "z"]
            + [f"q{i}" for i in range(1, 8)]
        )
        writer.writerows(rows)


def main():
    panda = GzPanda()

    print("Waiting for joint states from Gazebo...")
    if not panda.wait_for_state(timeout=5.0):
        print("No joint state arrived. Check that bash run_sim.sh is running.")
        return 1

    # A joint-space move back to A could hit the ball, so ask for a restart.
    q, _ = panda.read()
    if np.max(np.abs(np.array(q) - START_POSE)) > START_TOLERANCE:
        print("The arm is not at point A. Restart the simulator and try again.")
        return 1

    # q_target is the pose sent to Gazebo. It starts where the arm already is.
    q_target = START_POSE.copy()

    # Sending the same pose for 1 s lets gz-transport connect before the arm moves.
    connect_until = time.monotonic() + 1.0
    while time.monotonic() < connect_until:
        panda.send(q_target)
        time.sleep(DT)

    log = []
    start = time.monotonic()
    next_pass = start

    while True:
        t = time.monotonic() - start
        if t > MOVE_TIME + HOLD_TIME:
            break

        # Read the measured joint angles.
        q, _ = panda.read()
        q = np.array(q)

        # Compare where the end-effector should be with where it is.
        p_desired, v_desired = desired_motion(t)
        p = fk(q)[:3, 3]

        # Ask the controller for joint speeds, within the speed limit.
        qdot = basic_controller(q, p_desired, v_desired)
        qdot = np.clip(qdot, -SPEED_LIMIT, SPEED_LIMIT)

        # Move the joint targets forward by one time step, within the joint limits.
        q_target = np.clip(q_target + qdot * DT, Q_MIN, Q_MAX)
        panda.send(q_target)

        log.append([t, *p_desired, *p, *q])

        # Wait for the next pass so the loop stays at LOOP_HZ.
        next_pass += DT
        time.sleep(max(0.0, next_pass - time.monotonic()))

    save_log(log)

    # Columns 1 to 3 hold the desired position and columns 4 to 6 the actual one.
    data = np.array(log)
    errors = np.linalg.norm(data[:, 1:4] - data[:, 4:7], axis=1)
    print(f"Largest tracking error: {1000.0 * errors.max():.1f} mm")
    print(f"Final position error: {1000.0 * errors[-1]:.1f} mm")
    print(f"Data saved to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
