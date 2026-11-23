#!/usr/bin/python3
"""
run_avoid.py - Moves the end-effector from A to B with the designed controller.

The designed controller is the basic controller plus a null-space term:

    qdot = J+ [w_cmd; v_cmd]  +  (I - J+ J) qdot_avoid

The first term tracks the A-to-B path exactly as in run_basic.py. The second
term pushes the arm away from the ball, and (I - J+ J) keeps that push from
moving the hand.

Restart the simulator before each run so the arm begins at point A:

    Terminal 1: bash run_sim.sh
    Terminal 2: /usr/bin/python3 scripts/run_avoid.py

Each run saves its data to results/avoid_run.csv for your plots.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

from avoidance import arm_clearance, avoidance_velocity, point_jacobian
from gz_panda import GzPanda
from kinematics import fk, skew, space_jacobian
from run_basic import (
    DT,
    HAND_DOWN,
    HOLD_TIME,
    K_ORIENTATION,
    K_POSITION,
    MOVE_TIME,
    Q_MAX,
    Q_MIN,
    SPEED_LIMIT,
    START_POSE,
    START_TOLERANCE,
    desired_motion,
    orientation_error,
)

RESULTS_FILE = Path(__file__).resolve().parent.parent / "results" / "avoid_run.csv"


def designed_controller(q, p_desired, v_desired):
    """Return seven joint speeds [rad/s] that track the path and avoid the ball."""
    # Part 1: the same tracking term as the basic controller.
    T = fk(q)
    R = T[:3, :3]
    p = T[:3, 3]

    J_s = space_jacobian(q)
    J_w = J_s[:3, :]
    J_vs = J_s[3:, :]
    J = np.vstack([J_w, J_vs - skew(p) @ J_w])

    w_command = K_ORIENTATION * orientation_error(R, HAND_DOWN)
    v_command = v_desired + K_POSITION * (p_desired - p)
    twist = np.concatenate([w_command, v_command])

    J_pinv = J.T @ np.linalg.inv(J @ J.T)
    qdot_track = J_pinv @ twist

    # Part 2: the push away from the ball.
    clearance, c, n, k = arm_clearance(q)
    J_c = point_jacobian(J_s, c, k)
    qdot_avoid = avoidance_velocity(clearance, J_c, n)

    # Part 3: keep only the part of the push that does not move the hand.
    N = np.eye(7) - J_pinv @ J              # null-space projector: removes any hand motion
    return qdot_track + N @ qdot_avoid      # follow the path, and move the elbow away from the ball


def save_log(rows):
    """Write one line per loop pass to RESULTS_FILE."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with RESULTS_FILE.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["t", "x_des", "y_des", "z_des", "x", "y", "z"]
            + [f"q{i}" for i in range(1, 8)]
            + ["clearance"]
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
        clearance = arm_clearance(q)[0]

        # Ask the controller for joint speeds, within the speed limit.
        qdot = designed_controller(q, p_desired, v_desired)
        qdot = np.clip(qdot, -SPEED_LIMIT, SPEED_LIMIT)

        # Move the joint targets forward by one time step, within the joint limits.
        q_target = np.clip(q_target + qdot * DT, Q_MIN, Q_MAX)
        panda.send(q_target)

        log.append([t, *p_desired, *p, *q, clearance])

        # Wait for the next pass so the loop stays at the loop rate.
        next_pass += DT
        time.sleep(max(0.0, next_pass - time.monotonic()))

    save_log(log)

    # Columns 1 to 3 hold the desired position, 4 to 6 the actual one, and the
    # last column the arm-to-ball clearance from the capsule model.
    data = np.array(log)
    errors = np.linalg.norm(data[:, 1:4] - data[:, 4:7], axis=1)
    print(f"Largest tracking error: {1000.0 * errors.max():.1f} mm")
    print(f"Final position error: {1000.0 * errors[-1]:.1f} mm")
    print(f"Closest approach to the ball (capsule model): {1000.0 * data[:, -1].min():.1f} mm")
    print(f"Data saved to {RESULTS_FILE}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
