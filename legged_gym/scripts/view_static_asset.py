import os
import isaacgym  # noqa: F401
from isaacgym import gymapi, gymutil


def main():
    custom_parameters = [
        {
            "name": "--asset",
            "type": str,
            "default": "r2_v2_with_shell_no_hand/r2v2_with_shell.xml",
            "help": "Asset path relative to repo root, supports URDF/MJCF(XML) that Isaac Gym can load.",
        },
        {
            "name": "--pos",
            "type": float,
            "nargs": 3,
            "default": [0.0, 0.0, 0.9],
            "help": "Initial actor position x y z",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Run without GUI (for quick load test).",
        },
        {
            "name": "--gpu_physics",
            "action": "store_true",
            "default": False,
            "help": "Use GPU PhysX. Default is CPU PhysX for stable static preview.",
        },
    ]

    args = gymutil.parse_arguments(
        description="Static asset viewer for Isaac Gym",
        custom_parameters=custom_parameters,
    )

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    asset_path = args.asset
    if not os.path.isabs(asset_path):
        asset_path = os.path.join(repo_root, asset_path)

    if not os.path.exists(asset_path):
        raise FileNotFoundError(f"Asset file not found: {asset_path}")

    gym = gymapi.acquire_gym()

    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)

    if args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.use_gpu = args.gpu_physics
        sim_params.physx.num_position_iterations = 8
        sim_params.physx.num_velocity_iterations = 1

    sim_params.use_gpu_pipeline = False

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulation")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    asset_root = os.path.dirname(asset_path)
    asset_file = os.path.basename(asset_path)

    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = True
    asset_options.disable_gravity = True
    asset_options.collapse_fixed_joints = False
    asset_options.flip_visual_attachments = False

    asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

    env = gym.create_env(
        sim,
        gymapi.Vec3(-1.0, -1.0, 0.0),
        gymapi.Vec3(1.0, 1.0, 2.0),
        1,
    )

    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(args.pos[0], args.pos[1], args.pos[2])
    gym.create_actor(env, asset, pose, "static_robot", 0, 0)

    if args.headless:
        print(f"Loaded static asset successfully: {asset_path}")
        gym.destroy_sim(sim)
        return

    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        gym.destroy_sim(sim)
        raise RuntimeError("Failed to create viewer")

    cam_pos = gymapi.Vec3(2.2, 1.8, 1.5)
    cam_target = gymapi.Vec3(0.0, 0.0, 0.9)
    gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

    while not gym.query_viewer_has_closed(viewer):
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    main()
