## Review Notes

- Base file: `r2v2_with_shell.urdf`
- Intent: keep the robot at `28` active DoF without dexterous hands, and lock the `2` head DoF.
- Changed joints:
  - `head_yaw_joint`: `revolute` -> `fixed`
  - `head_pitch_joint`: `revolute` -> `fixed`
- Result:
  - Active DoF used by the body: `28`
  - Head DoF: locked
