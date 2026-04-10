import argparse
import glob
import os

import numpy as np

from legged_gym import LEGGED_GYM_ROOT_DIR


R2_DOF_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_arm_pitch_joint",
    "left_arm_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_arm_pitch_joint",
    "right_arm_yaw_joint",
]

G1_TO_R2_DOF_MAP = {
    "left_hip_pitch_joint": "left_hip_pitch_joint",
    "left_hip_roll_joint": "left_hip_roll_joint",
    "left_hip_yaw_joint": "left_hip_yaw_joint",
    "left_knee_joint": "left_knee_joint",
    "left_ankle_pitch_joint": "left_ankle_pitch_joint",
    "left_ankle_roll_joint": "left_ankle_roll_joint",
    "right_hip_pitch_joint": "right_hip_pitch_joint",
    "right_hip_roll_joint": "right_hip_roll_joint",
    "right_hip_yaw_joint": "right_hip_yaw_joint",
    "right_knee_joint": "right_knee_joint",
    "right_ankle_pitch_joint": "right_ankle_pitch_joint",
    "right_ankle_roll_joint": "right_ankle_roll_joint",
    "waist_yaw_joint": "waist_yaw_joint",
    "waist_pitch_joint": "waist_pitch_joint",
    "left_shoulder_pitch_joint": "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint": "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint": "left_shoulder_yaw_joint",
    "left_elbow_joint": "left_arm_pitch_joint",
    "left_wrist_roll_joint": "left_arm_yaw_joint",
    "right_shoulder_pitch_joint": "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint": "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint": "right_shoulder_yaw_joint",
    "right_elbow_joint": "right_arm_pitch_joint",
    "right_wrist_roll_joint": "right_arm_yaw_joint",
}

BODY_MAP = {
    "pelvis": "base_link",
    "left_rubber_hand": "left_hand_roll_link",
    "right_rubber_hand": "right_hand_roll_link",
    "left_ankle_roll_link": "left_ankle_roll_link",
    "right_ankle_roll_link": "right_ankle_roll_link",
}


def _decode_name_array(values):
    names = []
    for value in values:
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def _find_key(npz_data, candidates):
    for key in candidates:
        if key in npz_data:
            return key
    raise KeyError(f"None of keys found: {candidates}")


def _retarget_single_motion(input_file, output_file, target_base_height=0.92):
    src = np.load(input_file, allow_pickle=True)

    dof_names_key = _find_key(src, ["dof_names"])
    body_names_key = _find_key(src, ["body_names"])
    dof_pos_key = _find_key(src, ["dof_positions", "dof_pos"])
    dof_vel_key = _find_key(src, ["dof_velocities", "dof_vel"])
    body_pos_key = _find_key(src, ["body_positions", "body_pos"])
    body_rot_key = _find_key(src, ["body_rotations", "body_rot"])
    body_lin_vel_key = _find_key(src, ["body_linear_velocities", "body_lin_vel"])
    body_ang_vel_key = _find_key(src, ["body_angular_velocities", "body_ang_vel"])

    src_dof_names = _decode_name_array(src[dof_names_key].tolist())
    src_body_names = _decode_name_array(src[body_names_key].tolist())

    src_dof_idx = {name: i for i, name in enumerate(src_dof_names)}
    src_body_idx = {name: i for i, name in enumerate(src_body_names)}

    reverse_map = {dst: src_name for src_name, dst in G1_TO_R2_DOF_MAP.items()}
    src_dof_order = []
    for r2_name in R2_DOF_NAMES:
        g1_name = reverse_map[r2_name]
        if g1_name not in src_dof_idx:
            raise KeyError(f"Source DOF name not found in npz: {g1_name}")
        src_dof_order.append(src_dof_idx[g1_name])

    dof_positions = src[dof_pos_key][:, src_dof_order]
    dof_velocities = src[dof_vel_key][:, src_dof_order]

    src_body_selected = []
    dst_body_names = []
    for src_name, dst_name in BODY_MAP.items():
        if src_name not in src_body_idx:
            raise KeyError(f"Source body name not found in npz: {src_name}")
        src_body_selected.append(src_body_idx[src_name])
        dst_body_names.append(dst_name)

    body_positions = src[body_pos_key][:, src_body_selected]
    body_rotations = src[body_rot_key][:, src_body_selected]
    body_linear_velocities = src[body_lin_vel_key][:, src_body_selected]
    body_angular_velocities = src[body_ang_vel_key][:, src_body_selected]

    pelvis_z = body_positions[:, 0, 2]
    mean_pelvis_z = float(np.clip(np.mean(pelvis_z), 1e-4, None))
    z_scale = target_base_height / mean_pelvis_z
    body_positions[:, :, 2] *= z_scale
    body_linear_velocities[:, :, 2] *= z_scale

    dt = float(src["dt"]) if "dt" in src else (1.0 / float(src["fps"]) if "fps" in src else 1.0 / 30.0)

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez(
        output_file,
        dt=np.array(dt, dtype=np.float32),
        dof_names=np.array(R2_DOF_NAMES),
        body_names=np.array(dst_body_names),
        dof_positions=dof_positions.astype(np.float32),
        dof_velocities=dof_velocities.astype(np.float32),
        body_positions=body_positions.astype(np.float32),
        body_rotations=body_rotations.astype(np.float32),
        body_linear_velocities=body_linear_velocities.astype(np.float32),
        body_angular_velocities=body_angular_velocities.astype(np.float32),
    )

    required_bodies = {
        "base_link",
        "left_hand_roll_link",
        "right_hand_roll_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    }
    missing_bodies = sorted(required_bodies - set(dst_body_names))
    if missing_bodies:
        raise ValueError(f"Retargeted output missing required bodies: {missing_bodies}")

    if dof_positions.shape[1] != len(R2_DOF_NAMES):
        raise ValueError(
            f"Retargeted DOF dimension mismatch: got {dof_positions.shape[1]}, expected {len(R2_DOF_NAMES)}"
        )

    print(f"Retargeted motion saved to: {output_file}")
    print(f"dof_positions shape: {dof_positions.shape}")
    print(f"body_positions shape: {body_positions.shape}")


def retarget_motion(input_path, output_path, target_base_height=0.92, pattern="*.npz"):
    if os.path.isdir(input_path):
        if os.path.splitext(output_path)[1]:
            raise ValueError("--output must be a directory path when --input is a directory.")
        input_files = sorted(glob.glob(os.path.join(input_path, pattern)))
        if not input_files:
            raise FileNotFoundError(f"No input npz files matching {pattern} found in directory: {input_path}")
        os.makedirs(output_path, exist_ok=True)
        output_files = []
        for input_file in input_files:
            output_file = os.path.join(output_path, os.path.basename(input_file))
            _retarget_single_motion(input_file, output_file, target_base_height=target_base_height)
            output_files.append(output_file)
        print(f"Retargeted {len(output_files)} motion files into directory: {output_path}")
        return output_files

    if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
        os.makedirs(output_path, exist_ok=True)
        output_file = os.path.join(output_path, os.path.basename(input_path))
    else:
        output_file = output_path

    _retarget_single_motion(input_path, output_file, target_base_height=target_base_height)
    return [output_file]


def main():
    parser = argparse.ArgumentParser(description="Retarget G1 motion npz file(s) to R2 format.")
    parser.add_argument("--input", required=True, help="Input G1 motion npz file path or directory")
    parser.add_argument(
        "--output",
        default=os.path.join(LEGGED_GYM_ROOT_DIR, "legged_gym", "motions"),
        help="Output R2 motion npz path or directory",
    )
    parser.add_argument("--pattern", default="*.npz", help="Glob pattern when --input is a directory")
    parser.add_argument("--target-base-height", type=float, default=0.92)
    args = parser.parse_args()

    retarget_motion(
        args.input,
        args.output,
        target_base_height=args.target_base_height,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
