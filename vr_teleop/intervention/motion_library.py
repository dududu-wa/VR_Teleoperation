"""
Motion library for storing and sampling feasible upper-body motion clips.

Used in Phase 4+ training to replay recorded VR-derived or pre-computed
feasible arm motions during intervention, providing more realistic
perturbation patterns than pure random noise.

Clips are stored as sequences of upper-body joint angles with metadata
(duration, arm used, motion type).
"""

import torch
import numpy as np
import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict


@dataclass
class MotionClip:
    """A single motion clip for upper body replay."""
    joint_angles: np.ndarray    # (T, 14) upper body joint angles
    dt: float                   # Timestep between frames
    duration: float             # Total duration = T * dt
    motion_type: str = 'unknown'  # Type tag: arm_swing, reach, wave, etc.
    metadata: dict = field(default_factory=dict)


class MotionLibrary:
    """Stores and samples feasible upper-body motion clips.

    Features:
      - Load clips from NPZ files or add programmatically
      - Random clip sampling with optional filtering by type
      - Clip playback with interpolation
      - Batch sampling for multiple environments
    """

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device('cpu')
        self.clips: List[MotionClip] = []
        self._type_indices: Dict[str, List[int]] = {}

    def add_clip(self, clip: MotionClip):
        """Add a motion clip to the library."""
        idx = len(self.clips)
        self.clips.append(clip)
        if clip.motion_type not in self._type_indices:
            self._type_indices[clip.motion_type] = []
        self._type_indices[clip.motion_type].append(idx)

    def add_from_array(
        self, joint_angles: np.ndarray, dt: float,
        motion_type: str = 'unknown',
    ):
        """Add a clip from a numpy array.

        Args:
            joint_angles: (T, 14) joint angle trajectory
            dt: Timestep between frames
            motion_type: String tag for the motion type
        """
        clip = MotionClip(
            joint_angles=joint_angles.astype(np.float32),
            dt=dt,
            duration=joint_angles.shape[0] * dt,
            motion_type=motion_type,
        )
        self.add_clip(clip)

    @property
    def num_clips(self) -> int:
        return len(self.clips)

    @property
    def motion_types(self) -> list:
        return list(self._type_indices.keys())

    def sample_clip(self, motion_type: str = None) -> Optional[MotionClip]:
        """Sample a random clip, optionally filtered by type.

        Args:
            motion_type: If specified, only sample from this type

        Returns:
            A random MotionClip or None if library is empty
        """
        if self.num_clips == 0:
            return None

        if motion_type is not None and motion_type in self._type_indices:
            indices = self._type_indices[motion_type]
            idx = indices[np.random.randint(len(indices))]
        else:
            idx = np.random.randint(self.num_clips)

        return self.clips[idx]

    def sample_batch(
        self, batch_size: int, motion_type: str = None
    ) -> Tuple[Optional[List[MotionClip]], Optional[List[int]]]:
        """Sample a batch of clips.

        Returns:
            (clips, indices) or (None, None) if library is empty
        """
        if self.num_clips == 0:
            return None, None

        if motion_type is not None and motion_type in self._type_indices:
            pool = self._type_indices[motion_type]
        else:
            pool = list(range(self.num_clips))

        indices = [pool[np.random.randint(len(pool))] for _ in range(batch_size)]
        clips = [self.clips[i] for i in indices]
        return clips, indices

    def get_frame(
        self, clip_idx: int, time: float, interpolate: bool = True
    ) -> np.ndarray:
        """Get a frame from a clip at a given time.

        Args:
            clip_idx: Index of the clip
            time: Time in seconds (clamped to clip duration)
            interpolate: Whether to linearly interpolate between frames

        Returns:
            (14,) joint angles at the given time
        """
        clip = self.clips[clip_idx]
        T = clip.joint_angles.shape[0]

        # Clamp time to clip duration
        time = np.clip(time, 0.0, clip.duration - clip.dt)

        frame_f = time / clip.dt
        frame_i = int(frame_f)
        frac = frame_f - frame_i

        if not interpolate or frame_i >= T - 1:
            return clip.joint_angles[min(frame_i, T - 1)]

        # Linear interpolation
        return (1.0 - frac) * clip.joint_angles[frame_i] + frac * clip.joint_angles[frame_i + 1]

    def get_frame_torch(
        self, clip_idx: int, time: float, interpolate: bool = True
    ) -> torch.Tensor:
        """Get frame as torch tensor."""
        frame = self.get_frame(clip_idx, time, interpolate)
        return torch.from_numpy(frame).float().to(self.device)

    # ---- Persistence ----

    def save(self, path: str):
        """Save library to a directory.

        Creates:
          - clips.npz: joint angle arrays
          - metadata.json: clip metadata
        """
        os.makedirs(path, exist_ok=True)

        # Save arrays
        arrays = {}
        metadata_list = []
        for i, clip in enumerate(self.clips):
            arrays[f'clip_{i}'] = clip.joint_angles
            metadata_list.append({
                'dt': clip.dt,
                'duration': clip.duration,
                'motion_type': clip.motion_type,
                'metadata': clip.metadata,
            })

        np.savez_compressed(os.path.join(path, 'clips.npz'), **arrays)

        with open(os.path.join(path, 'metadata.json'), 'w') as f:
            json.dump(metadata_list, f, indent=2)

    def load(self, path: str):
        """Load library from a directory."""
        clips_path = os.path.join(path, 'clips.npz')
        meta_path = os.path.join(path, 'metadata.json')

        if not os.path.exists(clips_path) or not os.path.exists(meta_path):
            raise FileNotFoundError(f"Motion library files not found in {path}")

        data = np.load(clips_path)
        with open(meta_path, 'r') as f:
            metadata_list = json.load(f)

        self.clips = []
        self._type_indices = {}

        for i, meta in enumerate(metadata_list):
            key = f'clip_{i}'
            if key not in data:
                continue
            clip = MotionClip(
                joint_angles=data[key].astype(np.float32),
                dt=meta['dt'],
                duration=meta['duration'],
                motion_type=meta.get('motion_type', 'unknown'),
                metadata=meta.get('metadata', {}),
            )
            self.add_clip(clip)

    # ---- Built-in motion generators ----

    @staticmethod
    def generate_arm_swing(
        duration: float = 2.0, dt: float = 0.02,
        amplitude: float = 0.3, frequency: float = 1.5,
    ) -> MotionClip:
        """Generate a synthetic arm swing motion clip.

        Opposing shoulder pitch oscillation mimicking natural walking.
        """
        T = int(duration / dt)
        t = np.linspace(0, duration, T)

        angles = np.zeros((T, 14), dtype=np.float32)
        # Left shoulder pitch (index 0)
        angles[:, 0] = amplitude * np.sin(2 * np.pi * frequency * t)
        # Right shoulder pitch (index 7)
        angles[:, 7] = -amplitude * np.sin(2 * np.pi * frequency * t)
        # Left elbow (index 3) - slight bend
        angles[:, 3] = 0.3 * amplitude * np.cos(2 * np.pi * frequency * t)
        # Right elbow (index 10)
        angles[:, 10] = -0.3 * amplitude * np.cos(2 * np.pi * frequency * t)

        return MotionClip(
            joint_angles=angles, dt=dt, duration=duration,
            motion_type='arm_swing',
        )

    @staticmethod
    def generate_reaching(
        duration: float = 1.5, dt: float = 0.02,
        target_shoulder_pitch: float = 0.8,
        target_elbow: float = 0.5,
        arm: str = 'left',
    ) -> MotionClip:
        """Generate a reaching motion clip for one arm."""
        T = int(duration / dt)
        t = np.linspace(0, 1, T)  # normalized time

        # Smooth S-curve trajectory
        s = 3 * t**2 - 2 * t**3  # cubic ease in-out

        angles = np.zeros((T, 14), dtype=np.float32)
        if arm == 'left':
            angles[:, 0] = target_shoulder_pitch * s
            angles[:, 3] = target_elbow * s
        else:
            angles[:, 7] = target_shoulder_pitch * s
            angles[:, 10] = target_elbow * s

        return MotionClip(
            joint_angles=angles, dt=dt, duration=duration,
            motion_type='reaching',
            metadata={'arm': arm},
        )

    @staticmethod
    def generate_wave(
        duration: float = 2.0, dt: float = 0.02,
        arm: str = 'right',
    ) -> MotionClip:
        """Generate a waving motion clip."""
        T = int(duration / dt)
        t = np.linspace(0, duration, T)

        angles = np.zeros((T, 14), dtype=np.float32)
        offset = 7 if arm == 'right' else 0

        # Raise arm (shoulder pitch + roll)
        angles[:, offset + 0] = 1.0  # shoulder pitch up
        angles[:, offset + 1] = 0.3 * (1 if arm == 'left' else -1)  # shoulder roll out
        # Elbow bent
        angles[:, offset + 3] = 0.8
        # Wrist wave
        angles[:, offset + 6] = 0.4 * np.sin(2 * np.pi * 2.0 * t)  # wrist yaw wave

        return MotionClip(
            joint_angles=angles, dt=dt, duration=duration,
            motion_type='wave',
            metadata={'arm': arm},
        )


class MotionReplayBuffer:
    """Per-environment motion replay state for batch training.

    Tracks which clip each environment is playing and at what time.
    """

    def __init__(self, num_envs: int, library: MotionLibrary,
                 device: torch.device = None):
        self.num_envs = num_envs
        self.library = library
        self.device = device or torch.device('cpu')

        self.active_clip_idx = np.full(num_envs, -1, dtype=np.int32)
        self.playback_time = np.zeros(num_envs, dtype=np.float32)
        self.active = np.zeros(num_envs, dtype=bool)

    def reset(self, env_ids: np.ndarray):
        """Reset replay state for specific environments."""
        self.active_clip_idx[env_ids] = -1
        self.playback_time[env_ids] = 0.0
        self.active[env_ids] = False

    def start_replay(self, env_ids: np.ndarray, motion_type: str = None):
        """Start replaying clips for specified environments."""
        if self.library.num_clips == 0:
            return

        for env_id in env_ids:
            clip = self.library.sample_clip(motion_type)
            if clip is not None:
                idx = self.library.clips.index(clip)
                self.active_clip_idx[env_id] = idx
                self.playback_time[env_id] = 0.0
                self.active[env_id] = True

    def step(self, dt: float) -> np.ndarray:
        """Advance all active replays and return current frames.

        Returns:
            (N, 14) joint angles for all envs (zeros for inactive)
        """
        frames = np.zeros((self.num_envs, 14), dtype=np.float32)

        for i in range(self.num_envs):
            if not self.active[i] or self.active_clip_idx[i] < 0:
                continue

            clip_idx = self.active_clip_idx[i]
            clip = self.library.clips[clip_idx]

            # Check if clip is done
            if self.playback_time[i] >= clip.duration:
                self.active[i] = False
                continue

            frames[i] = self.library.get_frame(
                clip_idx, self.playback_time[i])
            self.playback_time[i] += dt

        return frames

    def step_torch(self, dt: float) -> torch.Tensor:
        """Step and return as torch tensor."""
        frames = self.step(dt)
        return torch.from_numpy(frames).float().to(self.device)
