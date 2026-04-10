# AMP Motion Directory

This directory is the canonical location for AMP reference motions in this project.

The AMP pipeline reads `.npz` files directly from this folder. We do not require an intermediate `.pkl` format in the training path.

## Directory Contract

- Store one or more motion clips as `.npz` files in this directory.
- The AMP config points to the directory itself, not to a single file.
- The motion loader scans `*.npz` in filename-sorted order and treats each file as one clip.

Current default AMP path:

```text
{LEGGED_GYM_ROOT_DIR}/legged_gym/motions
```

## Required NPZ Fields

Each motion file must contain the fields below:

- `dof_names`
- `body_names`
- `dof_positions`
- `dof_velocities`
- `body_positions`
- `body_rotations`
- `body_linear_velocities`
- `body_angular_velocities`
- `dt` or `fps`

Expected shapes:

- `dof_positions`: `(num_frames, num_dofs)`
- `dof_velocities`: `(num_frames, num_dofs)`
- `body_positions`: `(num_frames, num_bodies, 3)`
- `body_rotations`: `(num_frames, num_bodies, 4)`
- `body_linear_velocities`: `(num_frames, num_bodies, 3)`
- `body_angular_velocities`: `(num_frames, num_bodies, 3)`

## Multi-Clip Constraints

When multiple `.npz` files are placed in this directory, they must share the same:

- `dof_names`
- `body_names`
- `dt`

If any clip uses different names or a different timestep, the loader raises an error during initialization.

The loader samples clips with probability proportional to their valid duration, and it keeps AMP history inside a single clip so the history window never crosses clip boundaries.

## How To Generate Motions

Use `legged_gym/scripts/retarget_motion.py` to convert source motions into R2-compatible `.npz` files.

Single-file example:

```powershell
python legged_gym/scripts/retarget_motion.py `
  --input D:\codebase\vr_project\humanoid_amp\motions\G1_walk.npz `
  --output D:\codebase\vr_project\VR_Teleoperation\legged_gym\motions
```

Batch example:

```powershell
python legged_gym/scripts/retarget_motion.py `
  --input D:\codebase\vr_project\humanoid_amp\motions `
  --output D:\codebase\vr_project\VR_Teleoperation\legged_gym\motions `
  --pattern *.npz
```

Batch mode preserves the input basenames and writes all converted files into the output directory.

## What The Retarget Script Does

The retarget script converts source motions into the R2 layout used by AMP:

- remaps G1 joint names into the R2 DOF order
- selects the R2 bodies required by the AMP observation pipeline
- rescales body height to the target base height
- rescales vertical body linear velocity consistently with the height scaling

The script validates that the retargeted result still contains the required R2 bodies and the expected DOF count.

## Recommended Naming

Use stable, descriptive clip names so the directory stays readable, for example:

- `r2_walk_forward.npz`
- `r2_walk_turn_left.npz`
- `r2_idle_shift_weight.npz`

Filename order does not change clip contents, but keeping names consistent makes the dataset easier to inspect and maintain.

## Quick Checklist

Before training with AMP, make sure:

- all reference motions are stored under `legged_gym/motions`
- all clips are `.npz`
- all clips share identical `dof_names`, `body_names`, and `dt`
- the retargeted output contains valid R2 bodies
- the AMP config still points at the directory, not at one file
