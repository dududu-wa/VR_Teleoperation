## Review Notes

- Base file: `r2v2_with_shell.urdf`
- Intent: keep the training asset aligned with the 26-DoF LAFAN1 R2V2 MuJoCo
  motion exports used by AMP, including the two head DoF and excluding
  dexterous hand DoF.
- Current joints:
  - `head_yaw_joint`: `revolute`
  - `head_pitch_joint`: `revolute`
- Result:
  - Active DoF used by the body policy: `26`
  - Head DoF: unlocked
