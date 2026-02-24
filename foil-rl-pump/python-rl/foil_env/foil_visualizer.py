"""
Reusable foil and rider visualization module.

Draws the foil assembly and articulated rider with configurable foil geometry.
"""

import numpy as np
from matplotlib.patches import Polygon, Circle, FancyBboxPatch
from matplotlib.collections import PatchCollection


# Default foil geometry (can be overridden by config)
DEFAULT_FOIL_GEOMETRY = {
    'board_length': 0.6,
    'board_thickness': 0.05,
    'mast_length': 0.70,
    'fuse_length': 0.70,
    'wing_chord': 0.08,  # Visual wing chord
    'wing_span_visual': 0.15,  # Visual half-span for drawing
    'stab_chord': 0.05,
    'stab_span_visual': 0.08,
}

# Rider body proportions
RIDER_GEOMETRY = {
    'shin_len': 0.40,
    'thigh_len': 0.42,
    'torso_len': 0.50,
    'head_radius': 0.10,
    'stance_width': 0.25,
    'shoulder_width': 0.24,
    'upper_arm_len': 0.30,
    'forearm_len': 0.25,
}


def solve_knee_ik(foot, hip, shin_l, thigh_l):
    """Inverse kinematics to find knee position (knees bend forward)."""
    dx = hip[0] - foot[0]
    dy = hip[1] - foot[1]
    dist = np.sqrt(dx*dx + dy*dy)
    dist = np.clip(dist, abs(shin_l - thigh_l) + 0.01, shin_l + thigh_l - 0.01)
    cos_foot = (shin_l**2 + dist**2 - thigh_l**2) / (2 * shin_l * dist)
    cos_foot = np.clip(cos_foot, -1, 1)
    angle_to_hip = np.arctan2(dy, dx)
    angle_at_foot = np.arccos(cos_foot)
    # Subtract angle to place knee in FRONT (forward bend, like human squat)
    return (foot[0] + shin_l * np.cos(angle_to_hip - angle_at_foot),
            foot[1] + shin_l * np.sin(angle_to_hip - angle_at_foot))


def draw_foil_assembly(ax, z, theta, foil_config=None, alpha=1.0, x_center=0.0):
    """
    Draw the foil assembly (board, mast, fuselage, wings).

    Args:
        ax: matplotlib axes
        z: altitude (m) - positive above water
        theta: pitch angle (rad)
        foil_config: dict with foil parameters (S, AR, S_stab, etc.)
        alpha: transparency
        x_center: horizontal center position

    Returns:
        dict with key positions (mast_bottom, wing_pos, stab_pos)
    """
    geom = DEFAULT_FOIL_GEOMETRY.copy()

    # Scale wing/stab visuals based on config if provided
    if foil_config:
        # Scale wing visual size by area ratio
        base_S = 0.135
        wing_scale = np.sqrt(foil_config.get('S', base_S) / base_S)
        geom['wing_span_visual'] *= wing_scale
        geom['wing_chord'] *= wing_scale

        # Scale stab visual size
        base_S_stab = 0.012
        stab_scale = np.sqrt(foil_config.get('S_stab', base_S_stab) / base_S_stab)
        geom['stab_span_visual'] *= stab_scale
        geom['stab_chord'] *= stab_scale

    water_y = 0.0
    riding_depth = 0.2
    mast_len = geom['mast_length']  # 0.7m
    # Board is mast_length above wing, wing is riding_depth below surface at z=0
    # So board is (mast_length - riding_depth) above water at z=0
    board_center_y = water_y + (mast_len - riding_depth) + z

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # Board
    bl = geom['board_length']
    bt = geom['board_thickness']
    board_local = [(-bl/2, 0), (bl/2, 0), (bl/2, bt), (-bl/2, bt)]
    board_world = [transform(p) for p in board_local]
    ax.add_patch(Polygon(board_world, facecolor='saddlebrown', edgecolor='black',
                         linewidth=2, alpha=alpha, zorder=10))

    # Mast
    mast_len = geom['mast_length']
    mast_top = transform((0, 0))
    mast_bottom = transform((0, -mast_len))
    ax.plot([mast_top[0], mast_bottom[0]], [mast_top[1], mast_bottom[1]],
            'dimgray', linewidth=5, alpha=alpha, solid_capstyle='round', zorder=9)

    # Fuselage
    fuse_len = geom['fuse_length']
    fuse_front = transform((fuse_len * 0.4, -mast_len))
    fuse_rear = transform((-fuse_len * 0.6, -mast_len))
    ax.plot([fuse_front[0], fuse_rear[0]], [fuse_front[1], fuse_rear[1]],
            'dimgray', linewidth=4, alpha=alpha, solid_capstyle='round', zorder=8)

    # Main wing (at front of fuselage)
    wc = geom['wing_chord']
    ws = geom['wing_span_visual']
    wing_x = fuse_len * 0.4
    wing_local = [
        (wing_x - wc/2, -mast_len - ws),
        (wing_x + wc/2, -mast_len - ws),
        (wing_x + wc/2, -mast_len + ws),
        (wing_x - wc/2, -mast_len + ws),
    ]
    wing_world = [transform(p) for p in wing_local]
    ax.add_patch(Polygon(wing_world, facecolor='silver', edgecolor='black',
                         linewidth=1.5, alpha=alpha, zorder=7))

    # Stabilizer (at rear of fuselage)
    sc = geom['stab_chord']
    ss = geom['stab_span_visual']
    stab_x = -fuse_len * 0.6
    stab_local = [
        (stab_x - sc/2, -mast_len - ss),
        (stab_x + sc/2, -mast_len - ss),
        (stab_x + sc/2, -mast_len + ss),
        (stab_x - sc/2, -mast_len + ss),
    ]
    stab_world = [transform(p) for p in stab_local]
    ax.add_patch(Polygon(stab_world, facecolor='silver', edgecolor='black',
                         linewidth=1, alpha=alpha, zorder=7))

    return {
        'board_center': (x_center, board_center_y),
        'mast_bottom': mast_bottom,
        'wing_pos': transform((wing_x, -mast_len)),
        'stab_pos': transform((stab_x, -mast_len)),
        'board_thickness': bt,
    }


def draw_rider(ax, foil_positions, theta, left_leg, right_leg, left_arm, right_arm,
               waist, alpha=1.0, max_leg_ext=0.15, max_arm_swing=1.5):
    """
    Draw articulated rider on the foil.

    Args:
        ax: matplotlib axes
        foil_positions: dict from draw_foil_assembly
        theta: board pitch angle (rad)
        left_leg, right_leg: leg extension (-1 to 1 normalized)
        left_arm, right_arm: arm swing (-1 to 1 normalized)
        waist: waist lean angle (rad)
        alpha: transparency
        max_leg_ext: max leg extension (m)
        max_arm_swing: max arm swing (rad)
    """
    geom = RIDER_GEOMETRY
    board_center = foil_positions['board_center']
    bt = foil_positions['board_thickness']

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + board_center[0], ry + board_center[1])

    # Feet positions on board
    sw = geom['stance_width']
    l_foot = transform((-sw/2, bt))
    r_foot = transform((sw/2, bt))

    # Leg extension affects hip height
    # Leg positions are in meters (-max_leg_ext to +max_leg_ext)
    # Normalize to -1 to +1, then map to actual visual height change
    left_leg_norm = np.clip(left_leg / max_leg_ext, -1, 1) if max_leg_ext > 0 else 0
    right_leg_norm = np.clip(right_leg / max_leg_ext, -1, 1) if max_leg_ext > 0 else 0

    avg_leg_ext = (left_leg_norm + right_leg_norm) / 2
    # Base standing height with slight knee bend (0.75 factor)
    # Then add actual leg extension directly (no arbitrary scaling)
    base_stand = geom['shin_len'] + geom['thigh_len'] * 0.75
    leg_height = base_stand + avg_leg_ext * max_leg_ext

    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)

    # Torso with waist lean
    torso_lean = waist
    shoulder_world = (
        hip_world[0] + geom['torso_len'] * np.sin(torso_lean),
        hip_world[1] + geom['torso_len'] * np.cos(torso_lean)
    )
    head_world = (
        shoulder_world[0] + (geom['head_radius'] + 0.02) * np.sin(torso_lean),
        shoulder_world[1] + (geom['head_radius'] + 0.02) * np.cos(torso_lean)
    )

    # Knees via IK
    l_knee = solve_knee_ik(l_foot, hip_world, geom['shin_len'], geom['thigh_len'])
    r_knee = solve_knee_ik(r_foot, hip_world, geom['shin_len'], geom['thigh_len'])

    # Draw legs
    for foot, knee, leg_color in [(l_foot, l_knee, 'royalblue'), (r_foot, r_knee, 'crimson')]:
        # Shin
        ax.plot([foot[0], knee[0]], [foot[1], knee[1]], leg_color,
                linewidth=6, alpha=alpha, solid_capstyle='round', zorder=15)
        # Thigh
        ax.plot([knee[0], hip_world[0]], [knee[1], hip_world[1]], leg_color,
                linewidth=8, alpha=alpha, solid_capstyle='round', zorder=15)
        # Knee joint
        ax.plot(knee[0], knee[1], 'ko', markersize=5, alpha=alpha, zorder=16)

    # Hip joint
    ax.plot(hip_world[0], hip_world[1], 'ko', markersize=6, alpha=alpha, zorder=16)

    # Torso
    ax.plot([hip_world[0], shoulder_world[0]], [hip_world[1], shoulder_world[1]],
            'darkgreen', linewidth=10, alpha=alpha, solid_capstyle='round', zorder=14)

    # Arms - use raw radians directly (no normalization needed)
    # arm positions are already in radians, use them directly
    l_arm_rad = left_arm
    r_arm_rad = right_arm

    shoulder_offset = geom['shoulder_width'] / 2
    l_shoulder = (shoulder_world[0] - shoulder_offset * np.cos(torso_lean),
                  shoulder_world[1] - shoulder_offset * np.sin(torso_lean) - 0.05)
    r_shoulder = (shoulder_world[0] + shoulder_offset * np.cos(torso_lean),
                  shoulder_world[1] + shoulder_offset * np.sin(torso_lean) - 0.05)

    def get_arm_points(shoulder, arm_rad, side):
        base_angle = np.pi / 2 + torso_lean
        swing_angle = base_angle - arm_rad  # Direct radians - no scaling
        elbow = (shoulder[0] + side * 0.02 + geom['upper_arm_len'] * np.sin(swing_angle),
                 shoulder[1] - geom['upper_arm_len'] * np.cos(swing_angle))
        forearm_angle = swing_angle + 0.3
        wrist = (elbow[0] + geom['forearm_len'] * np.sin(forearm_angle) * 0.8,
                 elbow[1] - geom['forearm_len'] * np.cos(forearm_angle) * 0.8)
        return elbow, wrist

    l_elbow, l_wrist = get_arm_points(l_shoulder, l_arm_rad, -1)
    r_elbow, r_wrist = get_arm_points(r_shoulder, r_arm_rad, 1)

    for shoulder, elbow, wrist, arm_color in [(l_shoulder, l_elbow, l_wrist, 'royalblue'),
                                               (r_shoulder, r_elbow, r_wrist, 'crimson')]:
        # Upper arm
        ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], arm_color,
                linewidth=5, alpha=alpha*0.9, solid_capstyle='round', zorder=13)
        # Forearm
        ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], arm_color,
                linewidth=4, alpha=alpha*0.9, solid_capstyle='round', zorder=13)
        # Elbow joint
        ax.plot(elbow[0], elbow[1], 'ko', markersize=4, alpha=alpha, zorder=14)

    # Head
    head = Circle(head_world, geom['head_radius'], facecolor='peachpuff',
                  edgecolor='black', linewidth=2, alpha=alpha, zorder=17)
    ax.add_patch(head)
    # Eye
    ax.plot(head_world[0] + 0.03, head_world[1] + 0.02, 'ko', markersize=3, alpha=alpha, zorder=18)


def draw_water(ax, t=0.0, xlim=(-1.5, 1.5), is_crashed=False, vx=4.5):
    """Draw water surface with animated streaks that flow based on velocity."""
    ax.axhspan(-1.5, 0.0, alpha=0.3, color='dodgerblue', zorder=1)
    ax.axhline(y=0.0, color='blue', linewidth=2, alpha=0.5, zorder=2)

    # Water streaks - flow smoothly based on velocity
    if not is_crashed:
        # Fixed streak positions that scroll with time
        np.random.seed(42)  # Fixed seed for consistent streak pattern
        n_streaks = 8
        streak_spacing = (xlim[1] - xlim[0] + 1.0) / n_streaks

        for j in range(n_streaks):
            # Base y position (fixed per streak)
            streak_y = -0.1 - (j % 4) * 0.15 - np.random.uniform(0, 0.1)

            # X position scrolls with time based on velocity
            base_x = xlim[0] + j * streak_spacing
            scroll_offset = (t * vx * 0.3) % (xlim[1] - xlim[0] + 1.0)  # Slower visual scroll
            streak_x = base_x - scroll_offset

            # Wrap around
            if streak_x < xlim[0] - 0.5:
                streak_x += (xlim[1] - xlim[0] + 1.0)

            if xlim[0] - 0.2 <= streak_x <= xlim[1]:
                streak_len = 0.08 + vx * 0.015  # Length scales with speed
                ax.plot([streak_x, streak_x - streak_len], [streak_y, streak_y],
                        'white', linewidth=1.5, alpha=0.4, zorder=3)


def draw_crash_indicator(ax, x=0.0, y=0.5, size=0.3):
    """Draw red X crash indicator."""
    ax.plot([x - size, x + size], [y - size, y + size], 'r-', linewidth=4, alpha=0.8, zorder=100)
    ax.plot([x - size, x + size], [y + size, y - size], 'r-', linewidth=4, alpha=0.8, zorder=100)


def compute_body_positions(z, theta, left_leg, right_leg, left_arm, right_arm, waist,
                           max_leg_ext=0.15, max_arm_swing=1.5, x_center=0.0):
    """
    Compute world coordinates of key body parts for motion trails.

    Returns dict with: l_foot, r_foot, l_hand, r_hand, head
    """
    geom = RIDER_GEOMETRY.copy()
    foil_geom = DEFAULT_FOIL_GEOMETRY.copy()

    water_y = 0.0
    riding_depth = 0.2
    mast_len = foil_geom['mast_length']  # 0.7m
    board_thickness = 0.05
    # Board is (mast_length - riding_depth) above water at z=0
    board_center_y = water_y + (mast_len - riding_depth) + z

    def rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    def transform(local_pt):
        rx, ry = rotate(local_pt[0], local_pt[1], theta)
        return (rx + x_center, ry + board_center_y)

    # Feet on board
    sw = geom['stance_width']
    l_foot = transform((-sw/2, board_thickness))
    r_foot = transform((sw/2, board_thickness))

    # Leg positions are in meters (-max_leg_ext to +max_leg_ext)
    # Normalize to -1 to +1, then map to actual visual height change
    left_leg_norm = np.clip(left_leg / max_leg_ext, -1, 1) if max_leg_ext > 0 else 0
    right_leg_norm = np.clip(right_leg / max_leg_ext, -1, 1) if max_leg_ext > 0 else 0

    avg_leg_ext = (left_leg_norm + right_leg_norm) / 2
    # Base standing height with slight knee bend (0.75 factor)
    # Then add actual leg extension directly (no arbitrary scaling)
    base_stand = geom['shin_len'] + geom['thigh_len'] * 0.75
    leg_height = base_stand + avg_leg_ext * max_leg_ext

    feet_mid_x = (l_foot[0] + r_foot[0]) / 2
    feet_mid_y = (l_foot[1] + r_foot[1]) / 2
    hip_world = (feet_mid_x, feet_mid_y + leg_height)

    # Torso with waist lean
    torso_lean = waist
    shoulder_world = (
        hip_world[0] + geom['torso_len'] * np.sin(torso_lean),
        hip_world[1] + geom['torso_len'] * np.cos(torso_lean)
    )
    head_world = (
        shoulder_world[0] + (geom['head_radius'] + 0.02) * np.sin(torso_lean),
        shoulder_world[1] + (geom['head_radius'] + 0.02) * np.cos(torso_lean)
    )

    # Arms - use raw radians directly
    l_arm_rad = left_arm
    r_arm_rad = right_arm

    # Shoulders and hands
    shoulder_offset = geom['shoulder_width'] / 2
    l_shoulder = (shoulder_world[0] - shoulder_offset * np.cos(torso_lean),
                  shoulder_world[1] - shoulder_offset * np.sin(torso_lean) - 0.05)
    r_shoulder = (shoulder_world[0] + shoulder_offset * np.cos(torso_lean),
                  shoulder_world[1] + shoulder_offset * np.sin(torso_lean) - 0.05)

    def get_wrist(shoulder, arm_rad, side):
        base_angle = np.pi / 2 + torso_lean
        swing_angle = base_angle - arm_rad  # Direct radians
        elbow = (shoulder[0] + side * 0.02 + geom['upper_arm_len'] * np.sin(swing_angle),
                 shoulder[1] - geom['upper_arm_len'] * np.cos(swing_angle))
        forearm_angle = swing_angle + 0.3
        wrist = (elbow[0] + geom['forearm_len'] * np.sin(forearm_angle) * 0.8,
                 elbow[1] - geom['forearm_len'] * np.cos(forearm_angle) * 0.8)
        return wrist

    l_hand = get_wrist(l_shoulder, l_arm_rad, -1)
    r_hand = get_wrist(r_shoulder, r_arm_rad, 1)

    return {
        'l_foot': l_foot, 'r_foot': r_foot,
        'l_hand': l_hand, 'r_hand': r_hand,
        'head': head_world
    }


def draw_motion_trails(ax, trail_positions, current_vx, dt, alpha=1.0):
    """
    Draw motion trails for body parts.

    Args:
        ax: matplotlib axes
        trail_positions: list of dicts from compute_body_positions (oldest to newest)
        current_vx: current velocity for offset calculation
        dt: time step between frames
        alpha: base transparency
    """
    if len(trail_positions) < 2:
        return

    trail_colors = {
        'l_foot': 'royalblue', 'r_foot': 'crimson',
        'l_hand': 'royalblue', 'r_hand': 'crimson',
        'head': 'orange'
    }

    # Build offset trail points (streaming behind the current position)
    for key in trail_colors:
        points = []
        for i, pos in enumerate(trail_positions):
            frames_ago = len(trail_positions) - 1 - i
            x_offset = frames_ago * current_vx * dt  # Trail streams behind at actual velocity
            points.append((pos[key][0] - x_offset, pos[key][1]))

        # Draw trail as fading lines
        color = trail_colors[key]
        for i in range(len(points) - 1):
            trail_alpha = (i + 1) / len(points) * 0.6 * alpha
            linewidth = 2 if 'foot' in key else (1.5 if 'hand' in key else 3)
            ax.plot([points[i][0], points[i+1][0]],
                    [points[i][1], points[i+1][1]],
                    color=color, linewidth=linewidth, alpha=trail_alpha, zorder=5)


def draw_foil_and_rider(ax, z, theta, left_leg, right_leg, left_arm, right_arm, waist,
                        t=0.0, foil_config=None, alpha=1.0, is_crashed=False,
                        max_leg_ext=0.15, max_arm_swing=1.5, vx=4.5, draw_water_bg=True):
    """
    Complete drawing of foil assembly and rider.

    This is the main entry point for visualization.

    Args:
        ax: matplotlib axes
        z: altitude (m)
        theta: pitch angle (rad)
        left_leg, right_leg: leg positions (m or normalized)
        left_arm, right_arm: arm positions (rad or normalized)
        waist: waist angle (rad)
        t: time (for water animation)
        foil_config: dict with foil parameters
        alpha: transparency
        is_crashed: whether to show crash indicator
        max_leg_ext: max leg extension for normalization
        max_arm_swing: max arm swing for normalization
        vx: horizontal velocity for water animation
        draw_water_bg: whether to draw water (set False if drawing separately)
    """
    # Draw water
    if draw_water_bg:
        draw_water(ax, t=t, is_crashed=is_crashed, vx=vx)

    # Draw foil
    foil_pos = draw_foil_assembly(ax, z, theta, foil_config=foil_config, alpha=alpha)

    # Draw rider
    draw_rider(ax, foil_pos, theta, left_leg, right_leg, left_arm, right_arm, waist,
               alpha=alpha, max_leg_ext=max_leg_ext, max_arm_swing=max_arm_swing)

    # Crash indicator
    if is_crashed:
        board_center = foil_pos['board_center']
        draw_crash_indicator(ax, x=board_center[0], y=board_center[1] + 0.5)
