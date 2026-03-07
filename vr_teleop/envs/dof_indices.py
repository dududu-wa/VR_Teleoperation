"""
DOF index definitions for split control: 13 locomotion DOFs + 16 VR/upper-body DOFs.

The G1 robot has 29 DOFs total:
  0-5:   left leg (6)
  6-11:  right leg (6)
  12:    waist_yaw
  13:    waist_roll
  14:    waist_pitch
  15-21: left arm (7)
  22-28: right arm (7)

Locomotion policy controls 13 DOFs: 12 leg joints + waist_pitch.
VR tracking (or default pose) controls 16 DOFs: waist_yaw, waist_roll + 14 arm joints.
"""

# 12 leg joints + waist_pitch (index 14)
LOCO_DOF_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14]

# waist_yaw (12), waist_roll (13) + 14 arm joints (15-28)
VR_DOF_INDICES = [12, 13, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]

NUM_LOCO_DOFS = len(LOCO_DOF_INDICES)   # 13
NUM_VR_DOFS = len(VR_DOF_INDICES)        # 16
