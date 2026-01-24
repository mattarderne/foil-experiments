"""
Demo: Effect of rider stance position on foil stability.

Shows 3 scenarios:
1. Rider too far forward (nose down, pearling)
2. Rider too far back (nose up, stall)
3. Rider in correct position (stable glide)

No pumping - just passive glide to show the physics.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from matplotlib.animation import FuncAnimation, FFMpegWriter

from foil_env.foil_physics import FoilPhysics, FoilState, CONFIG


def run_glide_simulation(com_offset: float, duration: float = 3.0, dt: float = 0.01):
    """
    Run a passive glide simulation with given rider CoM offset.

    Args:
        com_offset: Rider center of mass offset from CoL (m).
                   Positive = forward, Negative = back
        duration: Simulation duration (s)
        dt: Timestep (s)

    Returns:
        List of frame dictionaries
    """
    config = {**CONFIG, 'rider_com_offset': com_offset}
    physics = FoilPhysics(config)

    # Start at trim condition
    from foil_env.foil_physics import compute_trim_angle
    trim_theta = compute_trim_angle(4.5, config)

    state = FoilState(
        z=0.15,           # Slightly below trim depth
        vx=4.5,           # Cruise speed
        vz=0.0,
        theta=trim_theta,
        omega=0.0
    )

    frames = []
    t = 0.0

    while t < duration:
        forces = physics.calculate_forces(state)

        frame = {
            't': t,
            'z': state.z,
            'vx': state.vx,
            'vz': state.vz,
            'theta': state.theta,
            'omega': state.omega,
            'alpha': forces.alpha,
            'is_stalled': forces.is_stalled,
            'is_breached': forces.is_breached,
            'M_stab': forces.M_stab,
            'M_weight': forces.M_weight,
            'com_offset': com_offset,
        }
        frames.append(frame)

        # Passive glide - no leg force, no pitch torque
        state = physics.step(state, leg_force=0, pitch_torque=0, dt=dt)
        t += dt

        # Stop if crashed
        if state.z < -0.6 or state.z > 0.3:
            break
        if state.vx < 1.0:
            break

    return frames


def draw_foil_frame(ax, frame, label):
    """Draw a single foil frame."""
    ax.clear()

    z = frame['z']
    theta = frame['theta']
    vx = frame['vx']
    t = frame['t']
    com_offset = frame['com_offset']
    is_stalled = frame['is_stalled']

    water_y = 0.0
    riding_depth = 0.2
    board_center_y = water_y + riding_depth + z
    x_center = 0.0

    # Foil dimensions
    board_length = 0.6
    board_thickness = 0.05
    mast_length = 0.7
    fuse_length = 0.7

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # Draw water
    ax.axhspan(-2, water_y, alpha=0.3, color='dodgerblue')
    ax.axhline(y=water_y, color='blue', linewidth=2, alpha=0.5)

    # Water streaks
    water_offset = (t * vx * 2) % 8 - 4
    for i in range(10):
        streak_y = np.random.uniform(-1.0, -0.1)
        streak_x = ((i * 0.6 - water_offset) % 5) - 2.5
        ax.plot([streak_x, streak_x - 0.2], [streak_y, streak_y],
                'white', linewidth=1.5, alpha=0.5)

    # Board
    board_local = [
        (-board_length/2, 0), (board_length/2, 0),
        (board_length/2, board_thickness), (-board_length/2, board_thickness),
    ]
    board_world = [transform(p) for p in board_local]
    board_poly = Polygon(board_world, facecolor='saddlebrown', edgecolor='black', linewidth=2)
    ax.add_patch(board_poly)

    # Mast
    mast_top = transform((0, 0))
    mast_bottom = transform((0, -mast_length))
    ax.plot([mast_top[0], mast_bottom[0]], [mast_top[1], mast_bottom[1]],
            'dimgray', linewidth=5, solid_capstyle='round')

    # Fuselage
    fuse_front = transform((fuse_length * 0.4, -mast_length))
    fuse_rear = transform((-fuse_length * 0.6, -mast_length))
    ax.plot([fuse_front[0], fuse_rear[0]], [fuse_front[1], fuse_rear[1]],
            'dimgray', linewidth=4, solid_capstyle='round')

    # Front wing
    wing_local = [
        (fuse_length * 0.4, -mast_length - 0.02),
        (fuse_length * 0.4 + 0.06, -mast_length - 0.02),
        (fuse_length * 0.4 + 0.06, -mast_length + 0.02),
        (fuse_length * 0.4, -mast_length + 0.02),
    ]
    wing_world = [transform(p) for p in wing_local]
    ax.add_patch(Polygon(wing_world, facecolor='silver', edgecolor='black', linewidth=1.5))

    # Rear stab
    stab_local = [
        (-fuse_length * 0.6, -mast_length - 0.015),
        (-fuse_length * 0.6 + 0.04, -mast_length - 0.015),
        (-fuse_length * 0.6 + 0.04, -mast_length + 0.015),
        (-fuse_length * 0.6, -mast_length + 0.015),
    ]
    stab_world = [transform(p) for p in stab_local]
    ax.add_patch(Polygon(stab_world, facecolor='silver', edgecolor='black', linewidth=1))

    # Simple rider (stick figure)
    stance_width = 0.25
    l_foot = transform((-stance_width/2 + com_offset, board_thickness))
    r_foot = transform((stance_width/2 + com_offset, board_thickness))

    # Hip above feet
    hip = ((l_foot[0] + r_foot[0])/2, (l_foot[1] + r_foot[1])/2 + 0.7)
    shoulder = (hip[0], hip[1] + 0.5)
    head = (shoulder[0], shoulder[1] + 0.15)

    # Draw rider
    ax.plot([l_foot[0], hip[0]], [l_foot[1], hip[1]], 'blue', linewidth=4)
    ax.plot([r_foot[0], hip[0]], [r_foot[1], hip[1]], 'red', linewidth=4)
    ax.plot([hip[0], shoulder[0]], [hip[1], shoulder[1]], 'darkgreen', linewidth=6)
    ax.add_patch(Circle(head, 0.08, facecolor='peachpuff', edgecolor='black', linewidth=2))

    # Mark center of lift (at front wing)
    col_world = transform((fuse_length * 0.4, -mast_length))
    ax.plot(col_world[0], col_world[1], 'g*', markersize=12, label='CoL')

    # Styling
    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.2, 2.0)
    ax.set_aspect('equal')
    ax.set_facecolor('lightcyan')

    # Status
    status_color = 'red' if is_stalled else 'black'
    status_text = 'STALLED!' if is_stalled else ''

    info = f'{label}\nCoM offset: {com_offset:+.2f}m\nt={t:.2f}s  vx={vx:.1f}m/s\nz={z:.2f}m  pitch={np.degrees(theta):.1f}°\n{status_text}'
    ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            color=status_color)

    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Height (m)')


def create_comparison_video(output_path='stance_comparison.mp4'):
    """Create side-by-side comparison video of 3 stance positions."""

    # Run simulations
    print("Running simulations...")
    scenarios = [
        ('Too Far Forward', +0.15),   # 15cm forward - nose down
        ('Correct Position', 0.0),    # Neutral
        ('Too Far Back', -0.20),      # 20cm back - nose up, stall
    ]

    all_frames = {}
    for label, offset in scenarios:
        print(f"  {label} (offset={offset:+.2f}m)...")
        all_frames[label] = run_glide_simulation(offset, duration=4.0)
        print(f"    -> {len(all_frames[label])} frames, ended at t={all_frames[label][-1]['t']:.2f}s")

    # Find max frames
    max_frames = max(len(f) for f in all_frames.values())

    # Pad shorter simulations with last frame
    for label in all_frames:
        while len(all_frames[label]) < max_frames:
            all_frames[label].append(all_frames[label][-1])

    # Create animation
    print(f"Creating animation with {max_frames} frames...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    fig.suptitle('Effect of Rider Stance Position on Foil Stability', fontsize=14)

    # Frame skip for reasonable file size
    frame_skip = 3
    frames_to_render = list(range(0, max_frames, frame_skip))

    def animate(i):
        frame_idx = frames_to_render[i]
        for ax, (label, offset) in zip(axes, scenarios):
            draw_foil_frame(ax, all_frames[label][frame_idx], label)
        return []

    anim = FuncAnimation(fig, animate, frames=len(frames_to_render), interval=33, blit=False)

    try:
        writer = FFMpegWriter(fps=30, metadata=dict(artist='PumpFoil RL'))
        anim.save(output_path, writer=writer)
        print(f"Saved: {output_path}")
    except Exception as e:
        print(f"FFmpeg error ({e}), saving as GIF...")
        from matplotlib.animation import PillowWriter
        gif_path = output_path.replace('.mp4', '.gif')
        writer = PillowWriter(fps=15)
        anim.save(gif_path, writer=writer)
        print(f"Saved: {gif_path}")

    plt.close()

    # Print summary
    print("\n" + "="*60)
    print("STANCE POSITION DEMO SUMMARY")
    print("="*60)
    for label, offset in scenarios:
        frames = all_frames[label]
        final = frames[-1]
        print(f"\n{label} (offset={offset:+.2f}m):")
        print(f"  Duration: {final['t']:.2f}s")
        print(f"  Final vx: {final['vx']:.2f} m/s")
        print(f"  Final z: {final['z']:.2f} m")
        print(f"  Final pitch: {np.degrees(final['theta']):.1f}°")
        print(f"  Stalled: {final['is_stalled']}")
        print(f"  M_weight: {final['M_weight']:.1f} Nm")


if __name__ == '__main__':
    create_comparison_video()
