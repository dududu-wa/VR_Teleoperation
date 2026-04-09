"""
Convert lafan1_pkl_r2v2 pkl files to npz format for AMP training.

pkl format (r2v2, 26 DOF):
  fps, root_pos (T,3), root_rot (T,4) xyzw, dof_pos (T,26)

  r2v2 26 DOF: 12 legs + 2 waist + 5 left arm + 5 right arm + 2 head
  Training env 24 DOF: 12 legs + 2 waist + 5 left arm + 5 right arm
    (drops head joints, drops hand joints)

Output npz (24 DOF, head joints dropped):
  dt, dof_names, body_names,
  dof_positions, dof_velocities,
  body_positions, body_rotations (wxyz),
  body_linear_velocities, body_angular_velocities

AMP obs dim: 24+24+1+6+3+3+12 = 73
"""

import argparse
import os
import glob
import pickle
import numpy as np

# r2v2 26-DOF order (from MuJoCo XML, pkl stores xyzw quaternion)
R2V2_26_DOF_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_pitch_joint",          # 13
    "left_shoulder_pitch_joint",  # 14
    "left_shoulder_roll_joint",   # 15
    "left_shoulder_yaw_joint",    # 16
    "left_arm_pitch_joint",       # 17
    "left_arm_yaw_joint",         # 18
    "right_shoulder_pitch_joint", # 19
    "right_shoulder_roll_joint",  # 20
    "right_shoulder_yaw_joint",   # 21
    "right_arm_pitch_joint",      # 22
    "right_arm_yaw_joint",        # 23
    "head_yaw_joint",             # 24  <- dropped
    "head_pitch_joint",           # 25  <- dropped
]

# Training env 24-DOF order (from r2_config.py default_joint_angles, NUM_ACTIONS=24)
R2_24_DOF_NAMES = [
    "left_hip_pitch_joint",       # 0
    "left_hip_roll_joint",        # 1
    "left_hip_yaw_joint",         # 2
    "left_knee_joint",            # 3
    "left_ankle_pitch_joint",     # 4
    "left_ankle_roll_joint",      # 5
    "right_hip_pitch_joint",      # 6
    "right_hip_roll_joint",       # 7
    "right_hip_yaw_joint",        # 8
    "right_knee_joint",           # 9
    "right_ankle_pitch_joint",    # 10
    "right_ankle_roll_joint",     # 11
    "waist_yaw_joint",            # 12
    "waist_pitch_joint",          # 13
    "left_shoulder_pitch_joint",  # 14
    "left_shoulder_roll_joint",   # 15
    "left_shoulder_yaw_joint",    # 16
    "left_arm_pitch_joint",       # 17
    "left_arm_yaw_joint",         # 18
    "right_shoulder_pitch_joint", # 19
    "right_shoulder_roll_joint",  # 20
    "right_shoulder_yaw_joint",   # 21
    "right_arm_pitch_joint",      # 22
    "right_arm_yaw_joint",        # 23
]

# Mapping: for each of the 24 training DOFs, which pkl index to use
R2V2_NAME_TO_IDX = {name: i for i, name in enumerate(R2V2_26_DOF_NAMES)}
DOF_24_FROM_PKL = [R2V2_NAME_TO_IDX[name] for name in R2_24_DOF_NAMES]

# Body names for AMP obs (base + 4 key bodies)
BODY_NAMES = [
    "base_link",
    "left_hand_roll_link",
    "right_hand_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
]

# Approximate body offsets in root frame (meters)
BODY_OFFSETS_LOCAL = {
    "base_link":            np.array([0.0,  0.0,  0.0],  dtype=np.float32),
    "left_hand_roll_link":  np.array([0.0,  0.3,  0.1],  dtype=np.float32),
    "right_hand_roll_link": np.array([0.0, -0.3,  0.1],  dtype=np.float32),
    "left_ankle_roll_link": np.array([0.0,  0.1, -0.9],  dtype=np.float32),
    "right_ankle_roll_link":np.array([0.0, -0.1, -0.9],  dtype=np.float32),
}


def finite_diff_vel(pos, dt):
    vel = np.zeros_like(pos)
    vel[1:-1] = (pos[2:] - pos[:-2]) / (2 * dt)
    vel[0]    = (pos[1]  - pos[0])  / dt
    vel[-1]   = (pos[-1] - pos[-2]) / dt
    return vel


def quat_angular_velocity(quats_wxyz, dt):
    """Compute angular velocity from wxyz quaternion sequence."""
    T = quats_wxyz.shape[0]
    ang_vel = np.zeros((T, 3), dtype=np.float32)
    for i in range(1, T - 1):
        qw, qx, qy, qz = quats_wxyz[i]
        dqw, dqx, dqy, dqz = (quats_wxyz[i + 1] - quats_wxyz[i - 1]) / (2 * dt)
        ang_vel[i, 0] = 2 * (qw * dqx - qx * dqw + qy * dqz - qz * dqy)
        ang_vel[i, 1] = 2 * (qw * dqy - qy * dqw - qx * dqz + qz * dqx)
        ang_vel[i, 2] = 2 * (qw * dqz - qz * dqw + qx * dqy - qy * dqx)
    ang_vel[0] = ang_vel[1]
    ang_vel[-1] = ang_vel[-2]
    return ang_vel


def quat_apply_np(q_wxyz, v):
    """Rotate vectors v by quaternions q. q: (T,4) wxyz, v: (T,3) -> (T,3)"""
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    t = 2 * np.stack([
        y * v[:, 2] - z * v[:, 1],
        z * v[:, 0] - x * v[:, 2],
        x * v[:, 1] - y * v[:, 0],
    ], axis=-1)
    return v + w[:, None] * t + np.cross(np.stack([x, y, z], axis=-1), t)


def process_one(data, target_base_height=0.92):
    fps = int(data["fps"])
    dt = 1.0 / fps
    root_pos      = data["root_pos"].astype(np.float32).copy()
    root_rot_xyzw = data["root_rot"].astype(np.float32)  # pkl stores xyzw
    # convert xyzw -> wxyz for internal processing
    root_rot_wxyz = np.concatenate([root_rot_xyzw[:, 3:], root_rot_xyzw[:, :3]], axis=-1)
    dof_pos_26    = data["dof_pos"].astype(np.float32)
    T = root_pos.shape[0]

    # height normalization
    mean_z = float(np.clip(root_pos[:, 2].mean(), 1e-4, None))
    root_pos[:, 2] *= target_base_height / mean_z

    # build 24-DOF arrays (head joints dropped)
    dof_pos = np.zeros((T, 24), dtype=np.float32)
    for dst_i, src_i in enumerate(DOF_24_FROM_PKL):
        dof_pos[:, dst_i] = dof_pos_26[:, src_i]
    dof_vel = finite_diff_vel(dof_pos, dt)

    # body states
    root_lin_vel = finite_diff_vel(root_pos, dt)
    root_ang_vel = quat_angular_velocity(root_rot_wxyz, dt)

    body_pos = np.zeros((T, len(BODY_NAMES), 3), dtype=np.float32)
    body_rot = np.zeros((T, len(BODY_NAMES), 4), dtype=np.float32)
    body_lv  = np.zeros((T, len(BODY_NAMES), 3), dtype=np.float32)
    body_av  = np.zeros((T, len(BODY_NAMES), 3), dtype=np.float32)

    for bi, bname in enumerate(BODY_NAMES):
        offset = np.tile(BODY_OFFSETS_LOCAL[bname], (T, 1))
        rotated = quat_apply_np(root_rot_wxyz, offset)
        body_pos[:, bi] = root_pos + rotated
        body_rot[:, bi] = root_rot_wxyz  # stored as wxyz; collect_reference_motions converts to xyzw
        body_lv[:, bi]  = root_lin_vel
        body_av[:, bi]  = root_ang_vel

    return dt, dof_pos, dof_vel, body_pos, body_rot, body_lv, body_av


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",  default="/home/ubuntu/lzxworkspace/codespace/retargeting/lafan1_pkl_r2v2")
    parser.add_argument("--output_dir", default="legged_gym/motions")
    parser.add_argument("--pattern",    default="walk*.pkl")
    parser.add_argument("--merge",      action="store_true")
    parser.add_argument("--output_name", default="r2_walk.npz")
    parser.add_argument("--target_base_height", type=float, default=0.92)
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
    if not pkl_files:
        raise FileNotFoundError(f"No pkl files matching {args.pattern} in {args.input_dir}")
    print(f"Found {len(pkl_files)} pkl files")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.merge:
        all_dp, all_dv, all_bp, all_br, all_lv, all_av = [], [], [], [], [], []
        dt_val = None
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            dt, dp, dv, bp, br, lv, av = process_one(data, args.target_base_height)
            if dt_val is None:
                dt_val = dt
            all_dp.append(dp); all_dv.append(dv)
            all_bp.append(bp); all_br.append(br)
            all_lv.append(lv); all_av.append(av)

        out = os.path.join(args.output_dir, args.output_name)
        T_total = sum(x.shape[0] for x in all_dp)
        np.savez(out,
            dt=np.array(dt_val, dtype=np.float32),
            dof_names=np.array(R2_24_DOF_NAMES),
            body_names=np.array(BODY_NAMES),
            dof_positions=np.concatenate(all_dp),
            dof_velocities=np.concatenate(all_dv),
            body_positions=np.concatenate(all_bp),
            body_rotations=np.concatenate(all_br),
            body_linear_velocities=np.concatenate(all_lv),
            body_angular_velocities=np.concatenate(all_av),
        )
        print(f"Merged {len(pkl_files)} files -> {out}  (T={T_total}, dof=24)")
        print(f"AMP obs dim: 24+24+1+6+3+3+12 = 73")
    else:
        for pkl_path in pkl_files:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            dt, dp, dv, bp, br, lv, av = process_one(data, args.target_base_height)
            name = os.path.splitext(os.path.basename(pkl_path))[0]
            out = os.path.join(args.output_dir, f"{name}.npz")
            np.savez(out,
                dt=np.array(dt, dtype=np.float32),
                dof_names=np.array(R2_24_DOF_NAMES),
                body_names=np.array(BODY_NAMES),
                dof_positions=dp, dof_velocities=dv,
                body_positions=bp, body_rotations=br,
                body_linear_velocities=lv, body_angular_velocities=av,
            )
            print(f"Saved {out}  (T={dp.shape[0]}, dof=24)")


if __name__ == "__main__":
    main()
