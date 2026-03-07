"""
G1 left-right symmetry utilities for PPO symmetry loss.
Builds permutation and sign-flip matrices for observations and actions.
"""

import torch
from vr_teleop.robot.g1_config import G1Config


def build_action_symmetry_matrix(cfg: G1Config = None) -> torch.Tensor:
    """Build (13, 13) symmetry permutation matrix for locomotion actions.

    For the symmetry loss, we need a matrix S_a such that:
        mirror(action) = S_a @ action
    where mirror swaps left/right legs and negates roll/yaw joints.

    Locomotion (13 DOFs):
      [0-5]  left leg:  pitch, roll, yaw, knee, ankle_pitch, ankle_roll
      [6-11] right leg: pitch, roll, yaw, knee, ankle_pitch, ankle_roll
      [12]   waist_pitch

    Mirror map:
      left_no <-> right_no (same sign): {0<->6, 3<->9, 4<->10}
      left_op <-> right_op (negate):    {1<->7, 2<->8, 5<->11}
      waist_pitch (same sign):  {12}
    """
    if cfg is None:
        cfg = G1Config()

    n = cfg.loco_dofs  # 13
    S = torch.zeros(n, n, dtype=torch.float32)

    # Left <-> Right, same sign
    left_no = cfg.loco_sym_left_no    # [0, 3, 4]
    right_no = cfg.loco_sym_right_no  # [6, 9, 10]
    for l, r in zip(left_no, right_no):
        S[l, r] = 1.0
        S[r, l] = 1.0

    # Left <-> Right, opposite sign
    left_op = cfg.loco_sym_left_op    # [1, 2, 5]
    right_op = cfg.loco_sym_right_op  # [7, 8, 11]
    for l, r in zip(left_op, right_op):
        S[l, r] = -1.0
        S[r, l] = -1.0

    # Waist pitch: same sign
    for idx in cfg.loco_sym_waist_no:
        S[idx, idx] = 1.0

    return S


def build_obs_symmetry_matrix(obs_dim: int, cfg: G1Config = None) -> torch.Tensor:
    """Build observation symmetry permutation matrix.

    Single-step actor observation layout (67-dim):
      [0:3]   base_ang_vel         -> negate x (roll_rate), keep y, negate z (yaw_rate)
      [3:6]   projected_gravity    -> negate y component
      [6:19]  dof_pos (loco 13)    -> action symmetry matrix
      [19:32] dof_vel (loco 13)    -> action symmetry matrix
      [32:45] last_actions (13)    -> action symmetry matrix
      [45:61] upper_body_pos (16)  -> swap left(8) <-> right(8), negate roll/yaw
      [61:63] commands vx, vy      -> keep vx, negate vy
      [63]    command wz           -> negate
      [64]    gait_id              -> keep
      [65:67] clock (sin, cos)     -> keep (symmetric gait phase)

    Returns:
        (obs_dim, obs_dim) matrix S_o such that mirror(obs) = S_o @ obs
    """
    if cfg is None:
        cfg = G1Config()

    S = torch.zeros(obs_dim, obs_dim, dtype=torch.float32)
    S_act = build_action_symmetry_matrix(cfg)
    n_loco = cfg.loco_dofs  # 13

    idx = 0

    # base_ang_vel (3): negate x (roll), keep y (pitch), negate z (yaw)
    S[idx, idx] = -1.0      # omega_x -> -omega_x
    S[idx+1, idx+1] = 1.0   # omega_y -> omega_y
    S[idx+2, idx+2] = -1.0  # omega_z -> -omega_z
    idx += 3

    # projected_gravity (3): negate y
    S[idx, idx] = 1.0        # gx -> gx
    S[idx+1, idx+1] = -1.0   # gy -> -gy
    S[idx+2, idx+2] = 1.0    # gz -> gz
    idx += 3

    # dof_pos loco (13): use action symmetry
    S[idx:idx+n_loco, idx:idx+n_loco] = S_act
    idx += n_loco

    # dof_vel loco (13): same symmetry
    S[idx:idx+n_loco, idx:idx+n_loco] = S_act
    idx += n_loco

    # last_actions (13): same symmetry
    S[idx:idx+n_loco, idx:idx+n_loco] = S_act
    idx += n_loco

    # upper_body_pos (16): waist_yaw(1) + waist_roll(1) + left_arm(7) + right_arm(7)
    # VR_DOF_INDICES = [12, 13, 15..21, 22..28]
    # Within the 16-dim block:
    #   [0]    waist_yaw  -> negate (left-right flip)
    #   [1]    waist_roll -> negate
    #   [2:9]  left arm:  shoulder_pitch(no), shoulder_roll(op), shoulder_yaw(op),
    #                     elbow(no), wrist_roll(op), wrist_pitch(no), wrist_yaw(op)
    #   [9:16] right arm: same structure
    # waist: negate on mirror
    S[idx, idx] = -1.0      # waist_yaw -> -waist_yaw
    S[idx+1, idx+1] = -1.0  # waist_roll -> -waist_roll
    idx += 2

    # Arms: swap left(7) <-> right(7) with sign flips
    n_arm = 7
    # same sign: shoulder_pitch(0), elbow(3), wrist_pitch(5)
    # opposite sign: shoulder_roll(1), shoulder_yaw(2), wrist_roll(4), wrist_yaw(6)
    arm_same = [0, 3, 5]
    arm_opp = [1, 2, 4, 6]
    for j in arm_same:
        S[idx + j, idx + n_arm + j] = 1.0        # left -> right (same sign)
        S[idx + n_arm + j, idx + j] = 1.0         # right -> left (same sign)
    for j in arm_opp:
        S[idx + j, idx + n_arm + j] = -1.0        # left -> right (negate)
        S[idx + n_arm + j, idx + j] = -1.0         # right -> left (negate)
    idx += 2 * n_arm  # 14

    # commands: vx (keep), vy (negate)
    if idx < obs_dim:
        S[idx, idx] = 1.0      # vx -> vx
        idx += 1
    if idx < obs_dim:
        S[idx, idx] = -1.0     # vy -> -vy
        idx += 1
    if idx < obs_dim:
        S[idx, idx] = -1.0     # wz -> -wz
        idx += 1

    # Remaining dims (gait_id, clock): identity
    while idx < obs_dim:
        S[idx, idx] = 1.0
        idx += 1

    return S
