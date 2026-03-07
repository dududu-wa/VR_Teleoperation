"""
DDS bridge for Unitree G1 real robot communication.

Bridges the trained policy to the real G1 robot via Unitree SDK2
DDS topics (lowcmd / lowstate). Runs a control loop that:
  1. Subscribes to lowstate (joint pos, vel, IMU)
  2. Builds observations for the policy
  3. Runs inference to get target joint positions
  4. Publishes lowcmd with PD targets

Requires:
  - unitree_sdk2py installed
  - CycloneDDS environment configured
"""

import time
import struct
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional

from vr_teleop.robot.g1_config import G1Config
from vr_teleop.deploy.policy_wrapper import PolicyWrapper
from vr_teleop.envs.dof_indices import LOCO_DOF_INDICES, NUM_LOCO_DOFS


@dataclass
class DDSBridgeConfig:
    """Configuration for DDS bridge."""
    # Control loop
    control_dt: float = 0.02  # 50 Hz control loop (matches policy dt)
    action_scale: float = 0.25

    # Safety
    max_torque: float = 50.0  # Nm, per-joint torque clamp
    position_limit_margin: float = 0.1  # rad, margin from joint limits
    enable_safety_check: bool = True

    # Startup
    startup_duration: float = 3.0  # seconds to ramp from current to policy
    startup_ramp: bool = True


class DDSBridge:
    """DDS communication bridge for G1 real robot deployment.

    Connects trained policy to real robot via Unitree SDK2 DDS.
    """

    def __init__(self, policy: PolicyWrapper,
                 cfg: DDSBridgeConfig = None,
                 robot_cfg: G1Config = None):
        self.policy = policy
        self.cfg = cfg or DDSBridgeConfig()
        self.robot_cfg = robot_cfg or G1Config()

        # State from robot
        self.joint_pos = np.zeros(self.robot_cfg.num_dofs)
        self.joint_vel = np.zeros(self.robot_cfg.num_dofs)
        self.imu_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        self.imu_gyro = np.zeros(3)
        self.state_received = False

        # Default DOF positions
        self.default_dof_pos = self.robot_cfg.get_default_dof_pos().numpy()

        # Joint limits for safety
        self.pos_lower = np.array(self.robot_cfg.dof_pos_lower)
        self.pos_upper = np.array(self.robot_cfg.dof_pos_upper)

        # PD gains
        self.kp = self.robot_cfg.get_kp().numpy()
        self.kd = self.robot_cfg.get_kd().numpy()

        # DDS publishers/subscribers (lazy init)
        self._pub = None
        self._sub = None
        self._initialized = False

    def _init_dds(self):
        """Initialize DDS publishers and subscribers.

        This is separated from __init__ so the bridge can be
        created before the DDS infrastructure is ready.
        """
        try:
            from unitree_sdk2py.core.channel import (
                ChannelSubscriber, ChannelPublisher, ChannelFactoryInitialize
            )
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
            from unitree_sdk2py.idl.default import (
                unitree_hg_msg_dds__LowState_ as LowState_default
            )
        except ImportError:
            raise ImportError(
                "unitree_sdk2py is required for DDS bridge. "
                "Install it from the Unitree SDK2 Python package."
            )

        # Initialize DDS
        ChannelFactoryInitialize(0)

        # Publisher for low-level commands
        self._cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._cmd_pub.Init()

        # Subscriber for low-level state
        self._state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._state_sub.Init(self._lowstate_callback, 10)

        self._LowCmd_ = LowCmd_
        self._LowState_default = LowState_default
        self._initialized = True

        print("DDS bridge initialized")
        print(f"  Control dt: {self.cfg.control_dt}s")
        print(f"  Action scale: {self.cfg.action_scale}")

    def _lowstate_callback(self, msg):
        """Callback for lowstate DDS messages."""
        num_motor = self.robot_cfg.num_dofs

        for i in range(num_motor):
            self.joint_pos[i] = msg.motor_state[i].q
            self.joint_vel[i] = msg.motor_state[i].dq

        # IMU
        self.imu_quat_wxyz[0] = msg.imu_state.quaternion[0]
        self.imu_quat_wxyz[1] = msg.imu_state.quaternion[1]
        self.imu_quat_wxyz[2] = msg.imu_state.quaternion[2]
        self.imu_quat_wxyz[3] = msg.imu_state.quaternion[3]

        self.imu_gyro[0] = msg.imu_state.gyroscope[0]
        self.imu_gyro[1] = msg.imu_state.gyroscope[1]
        self.imu_gyro[2] = msg.imu_state.gyroscope[2]

        self.state_received = True

    def _build_obs_dict(self) -> dict:
        """Build observation dict from current robot state."""
        # Convert wxyz -> xyzw
        quat_xyzw = np.array([
            self.imu_quat_wxyz[1],
            self.imu_quat_wxyz[2],
            self.imu_quat_wxyz[3],
            self.imu_quat_wxyz[0],
        ])

        return {
            'base_quat_xyzw': quat_xyzw.astype(np.float32),
            'base_ang_vel': self.imu_gyro.astype(np.float32),
            'dof_pos': self.joint_pos.astype(np.float32),
            'dof_vel': self.joint_vel.astype(np.float32),
        }

    def _apply_safety(self, target_pos: np.ndarray) -> np.ndarray:
        """Apply safety limits to target positions.

        Args:
            target_pos: (29,) target joint positions (absolute)

        Returns:
            (29,) clamped target positions
        """
        if not self.cfg.enable_safety_check:
            return target_pos

        margin = self.cfg.position_limit_margin
        return np.clip(
            target_pos,
            self.pos_lower + margin,
            self.pos_upper - margin,
        )

    def _publish_cmd(self, target_pos: np.ndarray):
        """Publish low-level command via DDS.

        Args:
            target_pos: (29,) absolute target joint positions
        """
        cmd = self._LowCmd_()

        for i in range(self.robot_cfg.num_dofs):
            cmd.motor_cmd[i].mode = 1  # servo mode
            cmd.motor_cmd[i].q = float(target_pos[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = float(self.kp[i])
            cmd.motor_cmd[i].kd = float(self.kd[i])

        self._cmd_pub.Write(cmd)

    def run(self, gait_id: int = 1,
            commands: np.ndarray = None,
            duration: float = 30.0):
        """Run the control loop.

        Args:
            gait_id: 0=stand, 1=walk, 2=run
            commands: (3,) [vx, vy, wz] velocity commands
            duration: Run duration in seconds
        """
        if not self._initialized:
            self._init_dds()

        if commands is None:
            commands = np.array([0.4, 0.0, 0.0], dtype=np.float32)
            if gait_id == 0:
                commands = np.zeros(3, dtype=np.float32)

        self.policy.reset()

        print(f"\nWaiting for robot state...")
        while not self.state_received:
            time.sleep(0.01)
        print("  State received.")

        # Record initial position for startup ramp
        initial_pos = self.joint_pos.copy()

        print(f"Starting control loop (duration={duration}s)")
        print(f"  Gait: {gait_id}, Commands: {commands}")

        start_time = time.time()
        step = 0

        try:
            while True:
                loop_start = time.time()
                elapsed = loop_start - start_time

                if elapsed > duration:
                    break

                # Get policy action
                obs_dict = self._build_obs_dict()
                actions = self.policy.get_action(
                    obs_dict, gait_id=gait_id, commands=commands)

                # Convert action-scale to absolute position
                # target = default + action * action_scale
                target_pos = self.default_dof_pos.copy()
                target_pos[LOCO_DOF_INDICES] += \
                    actions * self.cfg.action_scale

                # Startup ramp: blend from initial to policy output
                if self.cfg.startup_ramp and elapsed < self.cfg.startup_duration:
                    alpha = elapsed / self.cfg.startup_duration
                    # Smooth cubic ramp
                    alpha = 3 * alpha**2 - 2 * alpha**3
                    target_pos = (1 - alpha) * initial_pos + alpha * target_pos

                # Safety
                target_pos = self._apply_safety(target_pos)

                # Publish command
                self._publish_cmd(target_pos)

                step += 1
                if step % 50 == 0:
                    print(f"  Step {step} | Time {elapsed:.1f}s")

                # Control loop pacing
                loop_elapsed = time.time() - loop_start
                sleep_time = self.cfg.control_dt - loop_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt - stopping.")

        # Ramp to default pose on exit
        print("Returning to default pose...")
        for i in range(50):
            alpha = (i + 1) / 50.0
            target = (1 - alpha) * self.joint_pos + alpha * self.default_dof_pos
            target = self._apply_safety(target)
            self._publish_cmd(target)
            time.sleep(self.cfg.control_dt)

        print("Control loop finished.")

