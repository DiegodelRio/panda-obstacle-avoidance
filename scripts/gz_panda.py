"""
gz_panda.py - Connection between Python and the Panda in Gazebo.

This file only moves numbers in and out of the simulator. It has no kinematics
or control logic, so your controllers can stay in plain NumPy and be tested
without Gazebo.

Data flow:
    Gazebo -> JointStatePublisher plugin -> joint_state topic -> read()
    send() -> /panda/jointN/cmd_pos topics -> JointPositionController plugins

ROS is not involved. Everything goes through gz-transport, Gazebo's own
messaging library. Run scripts that import this file with /usr/bin/python3,
because the gz Python packages are installed for the system Python.
"""

import threading
import time

from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.transport13 import Node

# The seven arm joints, in the order used by every list in this project.
JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]

# Gazebo publishes all joint states here. The name is built from the world and
# model names in panda_world.sdf.
STATE_TOPIC = "/world/panda_world/model/panda/joint_state"

# Each joint has its own command topic, as set in model.sdf.
COMMAND_TOPICS = [f"/panda/joint{i}/cmd_pos" for i in range(1, 8)]


class GzPanda:
    """Reads joint states from Gazebo and sends joint position targets to it."""

    def __init__(self):
        # The Node owns every gz-transport connection, so it must stay alive
        # for as long as this object does.
        self._node = Node()

        # gz-transport calls _on_state from its own thread while the main
        # program calls read(). The lock stops read() from seeing a half-updated
        # state.
        self._lock = threading.Lock()
        self._q = None
        self._qd = None
        self._last_update = None

        # One publisher for each joint's command topic.
        self._publishers = [
            self._node.advertise(topic, Double) for topic in COMMAND_TOPICS
        ]

        # Run _on_state each time Gazebo publishes a joint state message.
        if not self._node.subscribe(Model, STATE_TOPIC, self._on_state):
            raise RuntimeError(f"Could not subscribe to {STATE_TOPIC}")

    def _on_state(self, msg):
        """Store the latest joint angles and speeds from Gazebo."""
        # Gazebo does not promise an order, so index the joints by name.
        positions = {}
        velocities = {}
        for joint in msg.joint:
            # Keep only the last part of a scoped name such as panda::panda_joint1.
            name = joint.name.split("::")[-1]
            positions[name] = joint.axis1.position
            velocities[name] = joint.axis1.velocity

        # Ignore a message that does not contain all seven arm joints.
        if not all(name in positions for name in JOINT_NAMES):
            return

        with self._lock:
            self._q = [positions[name] for name in JOINT_NAMES]
            self._qd = [velocities[name] for name in JOINT_NAMES]
            self._last_update = time.monotonic()

    def wait_for_state(self, timeout=5.0):
        """Return True once a joint state arrives, or False after timeout seconds."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if self._q is not None:
                    return True
            time.sleep(0.01)
        return False

    def read(self):
        """Return copies of the latest joint angles [rad] and speeds [rad/s]."""
        with self._lock:
            if self._q is None:
                raise RuntimeError("No joint state received yet")
            return list(self._q), list(self._qd)

    def send(self, q_target):
        """Send seven joint position targets [rad] to Gazebo."""
        if len(q_target) != len(self._publishers):
            raise ValueError(f"Expected 7 joint targets, got {len(q_target)}")

        for publisher, value in zip(self._publishers, q_target):
            msg = Double()
            msg.data = float(value)
            publisher.publish(msg)
