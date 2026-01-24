"""
Proper foil animation with:
- Board + mast + foil (hydrofoil wing)
- Person standing on board
- Foil rotates with pitch
- Movement through frame showing speed
- Real-time playback speed
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
from matplotlib.animation import FuncAnimation, FFMpegWriter
from stable_baselines3 import PPO
from foil_env.pump_foil_env_4limb import PumpFoilEnv4Limb
import sys


def run_episode(model_path, max_frames=None):
    """Run episode and collect frame data."""
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)
    env = PumpFoilEnv4Limb()

    obs, _ = env.reset(seed=42)
    done = False
    frames = []
    t = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)

        frame = {
            't': t,
            'z': env.state.z,  # Altitude (foil depth)
            'vx': env.state.vx,
            'vz': env.state.vz,
            'theta': env.state.theta,  # Pitch angle
            'x': t * env.state.vx,  # Approximate x position
            'left_leg': env.left_leg_pos,
            'right_leg': env.right_leg_pos,
            'left_arm': env.left_arm_pos,
            'right_arm': env.right_arm_pos,
        }
        frames.append(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        t += env.dt

        if max_frames and len(frames) >= max_frames:
            break

    print(f"Collected {len(frames)} frames, {t:.2f}s")
    return frames, env.dt


def compute_body_positions(frame):
    """Compute key body positions (feet, hands, head) for a frame."""
    z = frame['z']
    theta = frame['theta']
    left_leg = frame['left_leg']
    right_leg = frame['right_leg']
    left_arm = frame['left_arm']
    right_arm = frame['right_arm']

    # Constants
    x_center = 0.0
    water_y = 0.0
    board_thickness = 0.05
    riding_depth = 0.2
    board_center_y = water_y + riding_depth + z

    shin_len = 0.40
    thigh_len = 0.42
    torso_len = 0.50
    upper_arm_len = 0.30
    forearm_len = 0.28
    head_radius = 0.10
    stance_width = 0.25

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # Feet on board
    l_foot = transform((-stance_width/2, board_thickness))
    r_foot = transform((stance_width/2, board_thickness))

    # Body upright
    avg_leg_ext = (left_leg + right_leg) / 2
    leg_height = shin_len + thigh_len * (0.7 + 0.25 * avg_leg_ext)
    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)
    shoulder_world = (hip_world[0], hip_world[1] + torso_len)
    head_world = (shoulder_world[0], shoulder_world[1] + head_radius + 0.02)

    # Hands
    def get_hand_pos(shoulder, arm_ext, side):
        angle = np.pi/2 + arm_ext * 0.6
        elbow = (
            shoulder[0] + side * 0.05 + upper_arm_len * np.sin(angle) * 0.3,
            shoulder[1] - upper_arm_len * np.cos(angle) * 0.8
        )
        wrist = (
            elbow[0] + forearm_len * np.sin(angle + 0.4) * 0.4,
            elbow[1] - forearm_len * np.cos(angle + 0.4) * 0.6
        )
        return wrist

    l_shoulder = (shoulder_world[0] - 0.08, shoulder_world[1] - 0.05)
    r_shoulder = (shoulder_world[0] + 0.08, shoulder_world[1] - 0.05)
    l_hand = get_hand_pos(l_shoulder, left_arm, -1)
    r_hand = get_hand_pos(r_shoulder, right_arm, 1)

    return {
        'l_foot': l_foot, 'r_foot': r_foot,
        'l_hand': l_hand, 'r_hand': r_hand,
        'head': head_world
    }


def draw_frame(ax, frames, frame_idx):
    """Draw a single frame with proper foil visualization and motion trails."""
    ax.clear()

    frame = frames[frame_idx]

    # Unpack state
    z = frame['z']  # Foil depth below surface
    theta = frame['theta']  # Pitch in radians
    vx = frame['vx']
    t = frame['t']
    left_leg = frame['left_leg']
    right_leg = frame['right_leg']
    left_arm = frame['left_arm']
    right_arm = frame['right_arm']

    # Foiler stays centered, water moves
    x_center = 0.0
    water_y = 0.0

    # Foil dimensions
    board_length = 0.6
    board_thickness = 0.05
    mast_length = 0.7
    fuse_length = 0.7
    front_wing_span = 0.5
    rear_stab_span = 0.25

    # Board center position in world
    riding_depth = 0.2
    board_center_y = water_y + riding_depth + z

    # === ROTATION FUNCTION ===
    # Rotate point around origin by angle theta
    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    # === DEFINE RIGID FOIL ASSEMBLY IN LOCAL COORDS ===
    # Local coords: board at y=0, mast goes down (-y), fuse is horizontal (x)
    # All coordinates relative to board center

    # Board corners (local)
    board_local = [
        (-board_length/2, 0),
        (board_length/2, 0),
        (board_length/2, board_thickness),
        (-board_length/2, board_thickness),
    ]

    # Mast: from board center straight down
    mast_top_local = (0, 0)
    mast_bottom_local = (0, -mast_length)

    # Fuselage: horizontal at bottom of mast, perpendicular to mast
    fuse_center_local = (0, -mast_length)
    fuse_front_local = (fuse_length * 0.4, -mast_length)  # Front (direction of travel)
    fuse_rear_local = (-fuse_length * 0.6, -mast_length)  # Rear

    # Front wing: at front of fuse, perpendicular (so vertical in local coords = into page in side view)
    # In 2D side view, we show it as a thick line
    front_wing_local = [
        (fuse_length * 0.4, -mast_length - 0.02),
        (fuse_length * 0.4 + 0.06, -mast_length - 0.02),
        (fuse_length * 0.4 + 0.06, -mast_length + 0.02),
        (fuse_length * 0.4, -mast_length + 0.02),
    ]

    # Rear stab: at rear of fuse
    rear_stab_local = [
        (-fuse_length * 0.6, -mast_length - 0.015),
        (-fuse_length * 0.6 + 0.04, -mast_length - 0.015),
        (-fuse_length * 0.6 + 0.04, -mast_length + 0.015),
        (-fuse_length * 0.6, -mast_length + 0.015),
    ]

    # === ROTATE ENTIRE ASSEMBLY BY PITCH ===
    def transform(local_pt):
        """Rotate by theta and translate to world position."""
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # Transform all parts
    board_world = [transform(p) for p in board_local]
    mast_top_world = transform(mast_top_local)
    mast_bottom_world = transform(mast_bottom_local)
    fuse_front_world = transform(fuse_front_local)
    fuse_rear_world = transform(fuse_rear_local)
    front_wing_world = [transform(p) for p in front_wing_local]
    rear_stab_world = [transform(p) for p in rear_stab_local]

    # === DRAW WATER (moving) ===
    ax.axhspan(-2, water_y, alpha=0.3, color='dodgerblue')
    ax.axhline(y=water_y, color='blue', linewidth=2, alpha=0.5)

    # Moving water streaks
    np.random.seed(int(t * 100) % 1000)
    water_offset = (t * vx * 2) % 8 - 4
    for i in range(15):
        streak_y = np.random.uniform(-1.2, -0.1)
        streak_x = ((i * 0.5 - water_offset) % 6) - 3
        streak_len = 0.15 + vx * 0.03
        ax.plot([streak_x, streak_x - streak_len], [streak_y, streak_y],
                'white', linewidth=1.5, alpha=0.6)

    # Surface ripples
    for i in range(8):
        ripple_x = ((i * 0.8 - water_offset * 0.5) % 6) - 3
        ax.plot([ripple_x, ripple_x - 0.2], [water_y - 0.02, water_y - 0.02],
                'lightblue', linewidth=2, alpha=0.7)

    # === DRAW RIGID FOIL ASSEMBLY ===
    # Board
    board_poly = plt.Polygon(board_world, facecolor='saddlebrown', edgecolor='black', linewidth=2)
    ax.add_patch(board_poly)

    # Mast
    ax.plot([mast_top_world[0], mast_bottom_world[0]],
            [mast_top_world[1], mast_bottom_world[1]],
            'dimgray', linewidth=5, solid_capstyle='round')

    # Fuselage
    ax.plot([fuse_front_world[0], fuse_rear_world[0]],
            [fuse_front_world[1], fuse_rear_world[1]],
            'dimgray', linewidth=4, solid_capstyle='round')

    # Front wing
    front_wing_poly = plt.Polygon(front_wing_world, facecolor='silver', edgecolor='black', linewidth=1.5)
    ax.add_patch(front_wing_poly)

    # Rear stabilizer
    rear_stab_poly = plt.Polygon(rear_stab_world, facecolor='silver', edgecolor='black', linewidth=1)
    ax.add_patch(rear_stab_poly)

    # === DRAW RIDER (body stays upright, feet on tilted board) ===
    # Anthropometric dimensions
    shin_len = 0.40
    thigh_len = 0.42
    torso_len = 0.50
    upper_arm_len = 0.30
    forearm_len = 0.28
    head_radius = 0.10
    stance_width = 0.25

    # === FEET: attached to board surface (rotate with board) ===
    l_foot_local = (-stance_width/2, board_thickness)
    r_foot_local = (stance_width/2, board_thickness)
    l_foot = transform(l_foot_local)
    r_foot = transform(r_foot_local)

    # === BODY: stays upright in world coords ===
    # Hip height depends on leg extension (average of both legs)
    avg_leg_ext = (left_leg + right_leg) / 2
    # Extension: -1 = crouched (low), +1 = extended (high)
    leg_height = shin_len + thigh_len * (0.7 + 0.25 * avg_leg_ext)

    # Hip is above the midpoint of feet, but upright
    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)

    # Shoulder and head above hip (upright)
    shoulder_world = (hip_world[0], hip_world[1] + torso_len)
    head_world = (shoulder_world[0], shoulder_world[1] + head_radius + 0.02)

    # === KNEES: IK to connect feet to hips ===
    def solve_knee(foot, hip, shin_l, thigh_l, leg_ext):
        """Solve for knee position given foot and hip using 2-link IK."""
        dx = hip[0] - foot[0]
        dy = hip[1] - foot[1]
        dist = np.sqrt(dx*dx + dy*dy)

        # Clamp distance to reachable range
        max_reach = shin_l + thigh_l
        min_reach = abs(shin_l - thigh_l)
        dist = np.clip(dist, min_reach + 0.01, max_reach - 0.01)

        # Law of cosines for knee angle
        cos_knee = (shin_l**2 + thigh_l**2 - dist**2) / (2 * shin_l * thigh_l)
        cos_knee = np.clip(cos_knee, -1, 1)

        # Angle from foot to hip
        angle_to_hip = np.arctan2(dy, dx)

        # Angle at foot joint
        cos_foot = (shin_l**2 + dist**2 - thigh_l**2) / (2 * shin_l * dist)
        cos_foot = np.clip(cos_foot, -1, 1)
        angle_at_foot = np.arccos(cos_foot)

        # Knee position (bend forward when crouched)
        knee_angle = angle_to_hip + angle_at_foot
        knee_x = foot[0] + shin_l * np.cos(knee_angle)
        knee_y = foot[1] + shin_l * np.sin(knee_angle)

        return (knee_x, knee_y)

    l_knee = solve_knee(l_foot, hip_world, shin_len, thigh_len, left_leg)
    r_knee = solve_knee(r_foot, hip_world, shin_len, thigh_len, right_leg)

    # Draw legs
    for foot, knee, color in [(l_foot, l_knee, 'royalblue'),
                               (r_foot, r_knee, 'crimson')]:
        ax.plot([foot[0], knee[0]], [foot[1], knee[1]], color, linewidth=6, solid_capstyle='round')
        ax.plot([knee[0], hip_world[0]], [knee[1], hip_world[1]], color, linewidth=8, solid_capstyle='round')
        ax.plot(knee[0], knee[1], 'ko', markersize=6)

    # Torso
    ax.plot([hip_world[0], shoulder_world[0]], [hip_world[1], shoulder_world[1]],
            'darkgreen', linewidth=10, solid_capstyle='round')

    # === ARMS: in world coords, move with arm positions ===
    def get_arm_world(shoulder, arm_ext, side):
        """Arm position in world coords."""
        # Arm angle: neutral is down-forward, arm_ext swings forward/back
        angle = np.pi/2 + arm_ext * 0.6  # More swing range
        elbow = (
            shoulder[0] + side * 0.05 + upper_arm_len * np.sin(angle) * 0.3,
            shoulder[1] - upper_arm_len * np.cos(angle) * 0.8
        )
        wrist = (
            elbow[0] + forearm_len * np.sin(angle + 0.4) * 0.4,
            elbow[1] - forearm_len * np.cos(angle + 0.4) * 0.6
        )
        return elbow, wrist

    l_shoulder_w = (shoulder_world[0] - 0.08, shoulder_world[1] - 0.05)
    r_shoulder_w = (shoulder_world[0] + 0.08, shoulder_world[1] - 0.05)

    l_elbow_w, l_wrist_w = get_arm_world(l_shoulder_w, left_arm, -1)
    r_elbow_w, r_wrist_w = get_arm_world(r_shoulder_w, right_arm, 1)

    for shoulder, elbow, wrist, color in [(l_shoulder_w, l_elbow_w, l_wrist_w, 'royalblue'),
                                           (r_shoulder_w, r_elbow_w, r_wrist_w, 'crimson')]:
        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], color, linewidth=5, alpha=0.8)
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], color, linewidth=4, alpha=0.8)
        ax.plot(elbow[0], elbow[1], 'ko', markersize=4)

    # Head
    head = Circle(head_world, head_radius, facecolor='peachpuff', edgecolor='black', linewidth=2)
    ax.add_patch(head)
    # Eye (looking forward)
    ax.plot(head_world[0] + 0.03, head_world[1] + 0.02, 'ko', markersize=3)

    # === MOTION TRAILS ===
    # Draw fading trails from feet, hands, and head
    trail_length = 15  # Number of frames to trail
    trail_start = max(0, frame_idx - trail_length)

    if frame_idx > 0:
        # Collect positions from previous frames, offset by movement
        trail_points = {'l_foot': [], 'r_foot': [], 'l_hand': [], 'r_hand': [], 'head': []}
        trail_colors = {
            'l_foot': 'royalblue', 'r_foot': 'crimson',
            'l_hand': 'royalblue', 'r_hand': 'crimson',
            'head': 'orange'
        }

        # Current frame's vx for calculating trail offset
        current_vx = frame['vx']
        dt = frames[1]['t'] - frames[0]['t'] if len(frames) > 1 else 0.01

        for i in range(trail_start, frame_idx + 1):
            pos = compute_body_positions(frames[i])
            # Offset trail by how far we've "moved" since that frame
            frames_ago = frame_idx - i
            x_offset = frames_ago * current_vx * dt * 2  # Trail streams behind

            for key in trail_points:
                trail_points[key].append((pos[key][0] - x_offset, pos[key][1]))

        # Draw trails as fading lines
        for key, points in trail_points.items():
            if len(points) < 2:
                continue
            color = trail_colors[key]
            for i in range(len(points) - 1):
                # Fade from transparent to opaque
                alpha = (i + 1) / len(points) * 0.6
                linewidth = 2 if 'foot' in key else (1.5 if 'hand' in key else 3)
                ax.plot([points[i][0], points[i+1][0]],
                        [points[i][1], points[i+1][1]],
                        color=color, linewidth=linewidth, alpha=alpha)

    # === STYLING ===
    ax.set_xlim(-3, 3)
    ax.set_ylim(-1.5, 2.5)
    ax.set_aspect('equal')
    ax.set_facecolor('lightcyan')

    # Info text
    ax.text(0.02, 0.98, f't={t:.2f}s  vx={vx:.1f}m/s  z={z:.2f}m  pitch={np.degrees(theta):.1f}°',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Height (m)')

    return []


def create_video(frames, dt, output_path='foil_animation.mp4', fps=None):
    """Create MP4 video at real-time speed."""
    if fps is None:
        fps = int(1.0 / dt)  # Real-time: 100 fps for dt=0.01

    # Cap at 30fps for smooth playback, adjust frame skip
    target_fps = 30
    frame_skip = max(1, fps // target_fps)
    actual_fps = fps // frame_skip

    frames_to_render = frames[::frame_skip]
    print(f"Rendering {len(frames_to_render)} frames at {actual_fps} fps (real-time speed)")

    fig, ax = plt.subplots(figsize=(12, 8))

    def animate(i):
        return draw_frame(ax, frames_to_render, i)

    anim = FuncAnimation(fig, animate, frames=len(frames_to_render), interval=1000/actual_fps, blit=False)

    # Save as MP4
    try:
        writer = FFMpegWriter(fps=actual_fps, metadata=dict(artist='PumpFoil RL'))
        anim.save(output_path, writer=writer)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg not available ({e}), saving as GIF instead...")
        from matplotlib.animation import PillowWriter
        gif_path = output_path.replace('.mp4', '.gif')
        writer = PillowWriter(fps=actual_fps)
        anim.save(gif_path, writer=writer)
        print(f"Saved: {gif_path}")

    plt.close()


if __name__ == '__main__':
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'checkpoints/model_4limb_modal.zip'
    max_frames = int(sys.argv[2]) if len(sys.argv) > 2 else None

    frames, dt = run_episode(model_path, max_frames)
    output = 'foil_test.mp4' if max_frames else 'foil_animation.mp4'
    create_video(frames, dt, output)
