"""
Create video from MuJoCo pump foil environment.

Uses MuJoCo's built-in renderer for smooth 3D visualization.
"""

import numpy as np
import mujoco
from pathlib import Path
import argparse
from typing import Optional
from tqdm import tqdm


def create_video(
    output_path: str = "mujoco_demo.mp4",
    duration: float = 10.0,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
    policy = None,
):
    """
    Create a video of the pump foil simulation.

    Args:
        output_path: Output video file path
        duration: Video duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        policy: Optional policy function(obs) -> action
    """
    from foil_env_mujoco import PumpFoilEnvMuJoCo, MuJoCoFoilConfig

    # Create environment
    config = MuJoCoFoilConfig(
        initial_height=0.05,  # Start slightly above water
        breach_height=0.4,    # Allow more headroom
        touchdown_depth=-0.3, # Less strict
    )
    env = PumpFoilEnvMuJoCo(config=config, render_mode="rgb_array")

    # Override renderer with higher resolution
    env.renderer = mujoco.Renderer(env.mj_model, height=height, width=width)

    # Configure camera for side view
    env.renderer.update_scene(env.mj_data, camera="side")

    obs, _ = env.reset()

    # Collect frames
    frames = []
    n_frames = int(duration * fps)
    sim_steps_per_frame = int(100 / fps)  # 100Hz sim, variable fps video

    print(f"Recording {n_frames} frames ({duration}s at {fps}fps)")

    for i in tqdm(range(n_frames)):
        # Run simulation steps
        for _ in range(sim_steps_per_frame):
            if policy is not None:
                action = policy(obs)
            else:
                # Default: sinusoidal pumping motion
                t = i / fps
                freq = 2.0  # 2 Hz pumping
                leg_cmd = np.sin(2 * np.pi * freq * t)
                arm_cmd = -np.sin(2 * np.pi * freq * t) * 0.5  # Opposite phase
                waist_cmd = np.sin(2 * np.pi * freq * t) * 0.3
                action = np.array([leg_cmd, arm_cmd, waist_cmd])

            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                print(f"Episode ended at t={i/fps:.1f}s: {info.get('termination', 'timeout')}")
                obs, _ = env.reset()
                break

        # Render frame
        env.renderer.update_scene(env.mj_data, camera="side")
        frame = env.renderer.render()
        frames.append(frame)

    env.close()

    # Write video
    print(f"Writing video to {output_path}")
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved: {output_path}")
    except ImportError:
        print("Install imageio: pip install imageio[ffmpeg]")
        # Fallback: save as images
        from PIL import Image
        for i, frame in enumerate(frames[:10]):
            img = Image.fromarray(frame)
            img.save(f"frame_{i:04d}.png")
        print(f"Saved first 10 frames as PNGs")


def create_comparison_video(
    output_path: str = "mujoco_comparison.mp4",
    duration: float = 10.0,
    fps: int = 30,
):
    """
    Create side-by-side comparison of different pumping frequencies.
    """
    from foil_env_mujoco import PumpFoilEnvMuJoCo, MuJoCoFoilConfig
    from PIL import Image, ImageDraw, ImageFont

    config = MuJoCoFoilConfig(
        initial_height=0.05,
        breach_height=0.4,
        touchdown_depth=-0.3,
    )

    frequencies = [1.0, 2.0, 3.0]  # Hz
    envs = []
    for _ in frequencies:
        env = PumpFoilEnvMuJoCo(config=config, render_mode="rgb_array")
        env.renderer = mujoco.Renderer(env.mj_model, height=360, width=480)
        envs.append(env)

    # Reset all envs
    obs_list = [env.reset()[0] for env in envs]

    frames = []
    n_frames = int(duration * fps)
    sim_steps_per_frame = int(100 / fps)

    print(f"Recording comparison: {frequencies} Hz")

    for i in tqdm(range(n_frames)):
        t = i / fps
        sub_frames = []

        for j, (env, freq, obs) in enumerate(zip(envs, frequencies, obs_list)):
            # Run simulation
            for _ in range(sim_steps_per_frame):
                leg_cmd = np.sin(2 * np.pi * freq * t)
                arm_cmd = -np.sin(2 * np.pi * freq * t) * 0.5
                waist_cmd = np.sin(2 * np.pi * freq * t) * 0.3
                action = np.array([leg_cmd, arm_cmd, waist_cmd])

                obs, reward, terminated, truncated, info = env.step(action)

                if terminated:
                    obs, _ = env.reset()
                    break

            obs_list[j] = obs

            # Render
            env.renderer.update_scene(env.mj_data)
            frame = env.renderer.render()
            sub_frames.append(frame)

        # Combine horizontally
        combined = np.concatenate(sub_frames, axis=1)
        frames.append(combined)

    for env in envs:
        env.close()

    # Write video
    print(f"Writing video to {output_path}")
    try:
        import imageio
        imageio.mimsave(output_path, frames, fps=fps)
        print(f"Saved: {output_path}")
    except ImportError:
        print("Install imageio: pip install imageio[ffmpeg]")


def main():
    parser = argparse.ArgumentParser(description="Create MuJoCo pump foil video")
    parser.add_argument("-o", "--output", default="mujoco_demo.mp4",
                        help="Output video path")
    parser.add_argument("-d", "--duration", type=float, default=10.0,
                        help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--width", type=int, default=1280, help="Video width")
    parser.add_argument("--height", type=int, default=720, help="Video height")
    parser.add_argument("--comparison", action="store_true",
                        help="Create frequency comparison video")

    args = parser.parse_args()

    if args.comparison:
        create_comparison_video(args.output, args.duration, args.fps)
    else:
        create_video(args.output, args.duration, args.fps, args.width, args.height)


if __name__ == "__main__":
    main()
