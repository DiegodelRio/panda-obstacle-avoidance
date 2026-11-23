# Panda model notice

The geometry, meshes, inertia values and joint data come from `franka_description`, using the `panda_description` copy in [example-robot-data](https://github.com/Gepetto/example-robot-data). They are licensed under Apache 2.0, and a copy of the licence is in `LICENSE`.

Changes made for this project:

- Converted the URDF to SDF.
- Removed the massless `panda_link8` and `panda_hand_tcp` links.
  - The hand now mounts directly on `panda_link7`.
  - `panda_hand_tcp` is an SDF frame.
- Fixed the fingers in the closed position.
- Bolted the base to the world.
- Raised the `panda_joint4` upper limit from -0.0698 to 0 rad so the arm can start at q = 0.
- Added Gazebo plugins for joint commands and joint state.
- Left out the URDF's joint damping, friction and safety controller values, which the position controllers bypass.
