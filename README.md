# Panda Obstacle Avoidance

A Franka Emika Panda moves its end-effector in a straight line from point A to point B while a fixed ball sits in the way of its arm. Two controllers are compared:

- **Basic controller:** Jacobian pseudoinverse. It tracks the path but drives the forearm into the ball.
- **Designed controller:** the same pseudoinverse plus a null-space term that swings the elbow around the ball while the end-effector keeps tracking the path.

All kinematics and controllers are written from scratch with NumPy: product-of-exponentials forward kinematics, the space Jacobian, the pseudoinverse and the null-space projector. The arm is simulated in Gazebo Harmonic and controlled from Python through Gazebo's own messaging library, gz-transport. ROS is not used.

MECH 5390/6390 Special Topics: Modern Robotics, Fall 2026, The University of Texas at El Paso.

## Requirements

- Ubuntu 22.04 (tested)
- Gazebo Harmonic, with its Python bindings
- Python 3 with NumPy
- git

The project runs on a CPU and does not need a dedicated GPU.

## Install

### 1. Gazebo Harmonic and the Python packages

```bash
sudo apt-get update
sudo apt-get install curl lsb-release gnupg
sudo curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null
sudo apt-get update
sudo apt-get install gz-harmonic python3-gz-transport13 python3-numpy git
```

If ROS 2 Humble is installed on the same computer, run `apt-get install -s gz-harmonic` first and check that it does not remove any `ros-humble-*` packages.

### 2. Check the install

```bash
gz sim shapes.sdf
```

A window with six simple shapes should open. Close it, then run:

```bash
/usr/bin/python3 -c "from gz.transport13 import Node; import numpy; print('ok')"
```

### 3. Get the code

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
```

## Check the math without Gazebo

These tests use only NumPy, so they run without the simulator:

```bash
/usr/bin/python3 scripts/test_kinematics.py
/usr/bin/python3 scripts/test_avoid.py
```

Each test should end with `All ... checks passed.` The last avoidance check simulates the whole A-to-B move in Python.

## Run the simulation

Use two terminals, both in the project folder. Restart the simulator before each controller run, so the arm begins at point A.

**Terminal 1: start Gazebo**

```bash
bash run_sim.sh
```

The arm rises from straight up and stops with its end-effector at the green dot (point A). The red ball is the obstacle, and the magenta dot is point B.

**Terminal 2: run a controller**

```bash
/usr/bin/python3 scripts/run_basic.py     # basic controller
/usr/bin/python3 scripts/run_avoid.py     # designed controller
```

Each run prints its largest tracking error and saves its data to the `results` folder.

Optional connection check, which swings joint 1 and prints PASS or FAIL:

```bash
/usr/bin/python3 scripts/check_motion.py
```

Use `/usr/bin/python3` for every script. The Gazebo Python packages are installed for the system Python, so a virtual environment or conda will not find them.

## Results

| Controller | Largest tracking error | Final position error | Closest approach to the ball |
| --- | --- | --- | --- |
| Basic | 349.9 mm | 268.4 mm | Forearm blocked by the ball |
| Designed | To be added | To be added | To be added |

## How it works

### Forward kinematics (`scripts/kinematics.py`)

Product of exponentials, with the screw axes and the end-effector pose `M` found at zero pose:

$$T(q) = e^{[S_1]q_1} e^{[S_2]q_2} \cdots e^{[S_7]q_7} M$$

The space Jacobian's column *i* is screw axis *i* moved by joints 1 to *i* − 1:

$$J_{s,i} = \mathrm{Ad}_{e^{[S_1]q_1} \cdots e^{[S_{i-1}]q_{i-1}}} \, S_i$$

### Basic controller (`scripts/run_basic.py`)

The desired path is a straight line with a quintic time scaling, so the hand starts and stops at rest:

$$p_d(t) = A + s(t)(B - A), \qquad s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5, \quad \tau = t / T$$

The joint speeds come from the pseudoinverse, with feedback on the position and orientation errors:

$$\dot q = J^{+} \begin{bmatrix} \omega_{cmd} \\ v_{cmd} \end{bmatrix}, \qquad J^{+} = J^T (J J^T)^{-1}$$

### Designed controller (`scripts/avoidance.py`, `scripts/run_avoid.py`)

The arm has 7 joints and the task uses 6, so one joint motion is left over. The null-space term uses it to move the elbow without moving the hand:

$$\dot q = J^{+} \begin{bmatrix} \omega_{cmd} \\ v_{cmd} \end{bmatrix} + (I - J^{+} J)\, \dot q_{avoid}$$

$$\dot q_{avoid} = k \, \frac{\max(0,\ d_{safe} - d)}{d_{safe}} \, J_c^T n$$

Here `d` is the gap between the arm and the ball, `c` is the arm's closest point, `n` points from the ball toward `c`, and `J_c` is the Jacobian of `c`. The arm is modeled as 5 capsules to find `d`.

## Project layout

```
run_sim.sh                 Starts Gazebo with the Panda world
worlds/panda_world.sdf     Ground, light, Panda, the ball, and points A and B
models/panda/              Panda model: SDF, meshes, licence and notice
scripts/
  gz_panda.py              Reads joint angles from Gazebo and sends joint targets
  kinematics.py            Forward kinematics and space Jacobian
  test_kinematics.py       Checks kinematics.py without Gazebo
  run_basic.py             Trajectory and basic controller
  avoidance.py             Arm-to-ball distance and the avoidance push
  run_avoid.py             Designed controller
  test_avoid.py            Checks avoidance.py and run_avoid.py without Gazebo
  check_motion.py          Connection check between Python and Gazebo
```

## Credits

The Panda geometry, meshes and inertia values come from `franka_description`, using the copy in [example-robot-data](https://github.com/Gepetto/example-robot-data), under the Apache 2.0 licence. The changes made for this project are listed in `models/panda/NOTICE.md`.
