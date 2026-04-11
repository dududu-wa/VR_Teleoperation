"""
Convert lafan1_npz_r2v2 files to VR_Teleoperation AMP motion format.

Source format (lafan1_npz_r2v2):
  - dof_positions: (T, 26)  includes head_yaw, head_pitch at indices 24,25
  - body_positions: (T, 31, 3)  all R2V2 bodies
  - dt: shape (1,)

Target format (legged_gym/motions):
  - dof_positions: (T, 24)  head joints removed
  - body_positions: (T, 5, 3)  only AMP key bodies
  - dt: scalar float32
  - dof_names: exactly R2_DOF_NAMES (24 entries)
  - body_names: exactly AMP_BODY_NAMES (5 entries)
"""

import argparse
import glob
import os
import numpy as np

HEAD_JOINT_NAMES = {'head_yaw_joint', 'head_pitch_joint'}

AMP_BODY_NAMES = [
    'base_link',
    'left_arm_yaw_link',
    'right_arm_yaw_link',
    'left_ankle_roll_link',
    'right_ankle_roll_link',
]


def convert_file(src_path, dst_path, target_base_height=0.92):
    data = np.load(src_path, allow_pickle=True)

    # ---- DOF selection: drop head joints ----
    dof_names_raw = data['dof_names']
    all_dof_names = [str(n) for n in dof_names_raw.tolist()]
    active_dof_idx = [i for i, n in enumerate(all_dof_names) if n not in HEAD_JOINT_NAMES]
    active_dof_names = [all_dof_names[i] for i in active_dof_idx]
    assert len(active_dof_names) == 24, f"Expected 24 active DOFs, got {len(active_dof_names)}"

    dof_positions = data['dof_positions'][:, active_dof_idx].astype(np.float32)
    dof_velocities = data['dof_velocities'][:, active_dof_idx].astype(np.float32)

    # ---- Body selection: pick AMP key bodies ----
    body_names_raw = data['body_names']
    all_body_names = [str(n) for n in body_names_raw.tolist()]
    body_name_to_idx = {n: i for i, n in enumerate(all_body_names)}

    missing = [n for n in AMP_BODY_NAMES if n not in body_name_to_idx]
    if missing:
        raise ValueError(f"{src_path}: missing body names: {missing}")

    amp_body_idx = [body_name_to_idx[n] for n in AMP_BODY_NAMES]

    body_positions = data['body_positions'][:, amp_body_idx, :].astype(np.float32)
    body_rotations = data['body_rotations'][:, amp_body_idx, :].astype(np.float32)
    body_linear_velocities = data['body_linear_velocities'][:, amp_body_idx, :].astype(np.float32)
    body_angular_velocities = data['body_angular_velocities'][:, amp_body_idx, :].astype(np.float32)

    # ---- Height rescaling ----
    # base_link is index 0 in the selected body array
    pelvis_z = body_positions[:, 0, 2]
    mean_pelvis_z = float(np.clip(np.mean(pelvis_z), 1e-4, None))
    z_scale = target_base_height / mean_pelvis_z
    body_positions[:, :, 2] = (body_positions[:, :, 2] * z_scale).astype(np.float32)
    body_linear_velocities[:, :, 2] = (body_linear_velocities[:, :, 2] * z_scale).astype(np.float32)

    # ---- dt ----
    dt_raw = data['dt']
    dt = float(dt_raw.flat[0])

    # ---- Save ----
    os.makedirs(os.path.dirname(dst_path) if os.path.dirname(dst_path) else '.', exist_ok=True)
    np.savez(
        dst_path,
        dt=np.float32(dt),
        dof_names=np.array(active_dof_names, dtype='U30'),
        body_names=np.array(AMP_BODY_NAMES, dtype='U30'),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    T = dof_positions.shape[0]
    print(f"  {os.path.basename(src_path):40s}  T={T:6d}  z_scale={z_scale:.3f}  -> {dst_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert lafan1_npz_r2v2 → VR_Teleoperation AMP motion format")
    parser.add_argument('--input', required=True, help='Source directory (lafan1_npz_r2v2)')
    parser.add_argument('--output', required=True, help='Destination directory (legged_gym/motions)')
    parser.add_argument('--pattern', default='*.npz')
    parser.add_argument('--target-base-height', type=float, default=0.92)
    args = parser.parse_args()

    src_files = sorted(glob.glob(os.path.join(args.input, args.pattern)))
    if not src_files:
        raise FileNotFoundError(f"No npz files found in {args.input}")

    os.makedirs(args.output, exist_ok=True)
    print(f"Converting {len(src_files)} files from {args.input} -> {args.output}")

    for src in src_files:
        dst = os.path.join(args.output, os.path.basename(src))
        try:
            convert_file(src, dst, target_base_height=args.target_base_height)
        except Exception as e:
            print(f"  ERROR {src}: {e}")

    print(f"\nDone. {len(src_files)} files written to {args.output}")


if __name__ == '__main__':
    main()
