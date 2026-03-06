"""
G1 left-right symmetry utilities for PPO symmetry loss.
Builds permutation and sign-flip matrices for observations and actions.
"""

import torch
from vr_teleop.robot.g1_config import G1Config


def build_action_symmetry_matrix(cfg: G1Config = None) -> torch.Tensor:
    """Build (15, 15) symmetry permutation matrix for lower-body actions.

    For the symmetry loss, we need a matrix S_a such that:
        mirror(action) = S_a @ action
    where mirror swaps left/right legs and negates roll/yaw joints.

    Lower body (15 DOFs):
      [0-5]  left leg:  pitch, roll, yaw, knee, ankle_pitch, ankle_roll
      [6-11] right leg: pitch, roll, yaw, knee, ankle_pitch, ankle_roll
      [12]   waist_yaw
      [13]   waist_roll
      [14]   waist_pitch

    Mirror map:
      left_no <-> right_no (same sign): {0<->6, 3<->9, 4<->10}
      left_op <-> right_op (negate):    {1<->7, 2<->8, 5<->11}
      waist_no (same sign):  {14}
      waist_op (negate):     {12, 13}
    """
    if cfg is None:
        cfg = G1Config.from_falcon_yaml_if_available()

    n = cfg.lower_body_dofs  # 15
    S = torch.zeros(n, n, dtype=torch.float32)

    # Left <-> Right, same sign ("no" type)
    left_no = cfg.lower_sym_left_no    # [0, 3, 4]
    right_no = cfg.lower_sym_right_no  # [6, 9, 10]
    for l, r in zip(left_no, right_no):
        S[l, r] = 1.0
        S[r, l] = 1.0

    # Left <-> Right, opposite sign ("op" type)
    left_op = cfg.lower_sym_left_op    # [1, 2, 5]
    right_op = cfg.lower_sym_right_op  # [7, 8, 11]
    for l, r in zip(left_op, right_op):
        S[l, r] = -1.0
        S[r, l] = -1.0

    # Waist: same sign
    for idx in cfg.symmetric_waist_no:
        S[idx, idx] = 1.0

    # Waist: opposite sign (negate)
    for idx in cfg.symmetric_waist_op:
        S[idx, idx] = -1.0

    return S


def build_obs_symmetry_matrix(obs_dim: int, cfg: G1Config = None) -> torch.Tensor:
    """Build observation symmetry permutation matrix.

    Single-step actor observation layout (58-dim):
      [0:3]   base_ang_vel         -> negate x (roll_rate), keep y, negate z (yaw_rate)
      [3:6]   projected_gravity    -> negate y component
      [6:21]  dof_pos (lower 15)   -> action symmetry matrix
      [21:36] dof_vel (lower 15)   -> action symmetry matrix
      [36:51] last_actions (15)    -> action symmetry matrix
      [51:53] commands vx, vy      -> keep vx, negate vy
      [53]    command wz           -> negate
      [54]    gait_id              -> keep
      [55]    intervention_flag    -> keep
      [56:58] clock (sin, cos)     -> keep (symmetric gait phase)

    Returns:
        (obs_dim, obs_dim) matrix S_o such that mirror(obs) = S_o @ obs
    """
    if cfg is None:
        cfg = G1Config.from_falcon_yaml_if_available()

    S = torch.zeros(obs_dim, obs_dim, dtype=torch.float32)
    S_act = build_action_symmetry_matrix(cfg)
    n_lower = cfg.lower_body_dofs  # 15

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

    # dof_pos lower (15): use action symmetry
    S[idx:idx+n_lower, idx:idx+n_lower] = S_act
    idx += n_lower

    # dof_vel lower (15): same symmetry
    S[idx:idx+n_lower, idx:idx+n_lower] = S_act
    idx += n_lower

    # last_actions (15): same symmetry
    S[idx:idx+n_lower, idx:idx+n_lower] = S_act
    idx += n_lower

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

    # Remaining dims (gait_id, intervention_flag, clock): identity
    while idx < obs_dim:
        S[idx, idx] = 1.0
        idx += 1

    return S
