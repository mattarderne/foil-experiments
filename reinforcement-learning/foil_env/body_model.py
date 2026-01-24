"""
Simple 2D Articulated Body Model for Pump Foiling

A rider standing on a board with:
- Torso (fixed relative to hips)
- Two legs (hip -> knee -> ankle)
- Two arms (shoulder -> elbow -> wrist)

All motion is in the vertical plane (sagittal plane).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class BodySegment:
    """A single body segment (limb part)."""
    name: str
    length: float  # m
    mass: float    # kg
    angle: float = 0.0  # rad, relative to parent
    angular_velocity: float = 0.0


@dataclass
class BodyState:
    """Full body state for visualization and physics."""
    # Base position (board/feet position)
    board_x: float = 0.0
    board_z: float = 0.0
    board_pitch: float = 0.0  # rad

    # Waist/torso lean angle (forward lean positive)
    waist_lean: float = 0.1  # rad, default slight forward lean
    waist_lean_vel: float = 0.0  # rad/s

    # Leg joint angles (relative to parent segment)
    # Positive = flexion (bending)
    left_hip: float = 0.0
    left_knee: float = 0.0
    right_hip: float = 0.0
    right_knee: float = 0.0

    # Arm joint angles
    left_shoulder: float = 0.0
    left_elbow: float = 0.0
    right_shoulder: float = 0.0
    right_elbow: float = 0.0

    # Velocities
    left_hip_vel: float = 0.0
    left_knee_vel: float = 0.0
    right_hip_vel: float = 0.0
    right_knee_vel: float = 0.0
    left_shoulder_vel: float = 0.0
    left_elbow_vel: float = 0.0
    right_shoulder_vel: float = 0.0
    right_elbow_vel: float = 0.0


class BodyModel:
    """
    2D articulated body model for pump foiling.

    Coordinate system:
    - X: forward (direction of travel)
    - Z: up (vertical)
    - Angles: positive = counter-clockwise

    Body structure:
    - Feet on board
    - Legs: thigh (hip to knee) + shin (knee to ankle)
    - Torso: fixed upright relative to hips
    - Arms: upper arm (shoulder to elbow) + forearm (elbow to wrist)
    """

    # Anthropometric data (average adult male)
    THIGH_LENGTH = 0.45  # m
    SHIN_LENGTH = 0.45   # m
    TORSO_LENGTH = 0.50  # m
    UPPER_ARM_LENGTH = 0.30  # m
    FOREARM_LENGTH = 0.25  # m

    # Masses (approximate)
    THIGH_MASS = 10.0  # kg
    SHIN_MASS = 5.0    # kg
    TORSO_MASS = 35.0  # kg
    UPPER_ARM_MASS = 3.0  # kg
    FOREARM_MASS = 2.0  # kg

    # Joint limits (radians)
    HIP_MIN, HIP_MAX = -0.5, 1.5      # -30 to 85 degrees
    KNEE_MIN, KNEE_MAX = 0.0, 2.5     # 0 to 145 degrees (only bends one way)
    SHOULDER_MIN, SHOULDER_MAX = -1.0, 2.5  # -60 to 145 degrees
    ELBOW_MIN, ELBOW_MAX = 0.0, 2.5   # 0 to 145 degrees

    # Stance width (for 3D projection)
    STANCE_WIDTH = 0.30  # m between feet
    SHOULDER_WIDTH = 0.40  # m between shoulders

    def __init__(self):
        self.state = BodyState()

    def reset(self, board_z: float = 0.2, board_pitch: float = 0.0):
        """Reset to standing position on board."""
        self.state = BodyState(
            board_x=0.0,
            board_z=board_z,
            board_pitch=board_pitch,
            # Waist: slight forward lean
            waist_lean=0.1,
            waist_lean_vel=0.0,
            # Standing: slight knee bend
            left_hip=0.1,
            left_knee=0.2,
            right_hip=0.1,
            right_knee=0.2,
            # Arms relaxed at sides
            left_shoulder=0.0,
            left_elbow=0.2,
            right_shoulder=0.0,
            right_elbow=0.2,
        )
        return self.state

    def get_joint_positions(self) -> dict:
        """
        Compute world positions of all joints.

        Leg kinematics: ankle -> knee -> hip (chain going up)
        - hip_angle: angle of thigh from vertical (positive = forward lean)
        - knee_angle: angle of shin from thigh (positive = bent)

        Returns dict with joint positions.
        """
        s = self.state
        positions = {}

        # Feet/ankles on board
        positions['left_ankle'] = (s.board_x - 0.05, s.board_z)
        positions['right_ankle'] = (s.board_x + 0.05, s.board_z)

        # Left leg chain: ankle -> knee -> hip
        # shin_angle: angle of shin from vertical
        shin_angle_left = s.board_pitch + s.left_knee * 0.5  # Shin tilts back when knee bent
        positions['left_knee'] = (
            positions['left_ankle'][0] - self.SHIN_LENGTH * np.sin(shin_angle_left),
            positions['left_ankle'][1] + self.SHIN_LENGTH * np.cos(shin_angle_left)
        )

        # Thigh angle from vertical
        thigh_angle_left = s.board_pitch - s.left_hip + s.left_knee * 0.3
        positions['left_hip'] = (
            positions['left_knee'][0] - self.THIGH_LENGTH * np.sin(thigh_angle_left),
            positions['left_knee'][1] + self.THIGH_LENGTH * np.cos(thigh_angle_left)
        )

        # Right leg (same logic)
        shin_angle_right = s.board_pitch + s.right_knee * 0.5
        positions['right_knee'] = (
            positions['right_ankle'][0] - self.SHIN_LENGTH * np.sin(shin_angle_right),
            positions['right_ankle'][1] + self.SHIN_LENGTH * np.cos(shin_angle_right)
        )

        thigh_angle_right = s.board_pitch - s.right_hip + s.right_knee * 0.3
        positions['right_hip'] = (
            positions['right_knee'][0] - self.THIGH_LENGTH * np.sin(thigh_angle_right),
            positions['right_knee'][1] + self.THIGH_LENGTH * np.cos(thigh_angle_right)
        )

        # Torso base at hip center
        hip_center = (
            (positions['left_hip'][0] + positions['right_hip'][0]) / 2,
            (positions['left_hip'][1] + positions['right_hip'][1]) / 2
        )
        positions['torso_base'] = hip_center

        # Torso extends upward with waist lean
        torso_lean = s.waist_lean
        positions['torso_top'] = (
            hip_center[0] - self.TORSO_LENGTH * np.sin(torso_lean),
            hip_center[1] + self.TORSO_LENGTH * np.cos(torso_lean)
        )

        # Shoulders
        positions['left_shoulder'] = (positions['torso_top'][0] - 0.08, positions['torso_top'][1] - 0.05)
        positions['right_shoulder'] = (positions['torso_top'][0] + 0.08, positions['torso_top'][1] - 0.05)

        # Arms
        for side in ['left', 'right']:
            shoulder = positions[f'{side}_shoulder']
            arm_angle = getattr(s, f'{side}_shoulder')  # Forward/back swing

            # Upper arm hangs down, swings forward/back
            elbow_x = shoulder[0] + self.UPPER_ARM_LENGTH * np.sin(arm_angle)
            elbow_z = shoulder[1] - self.UPPER_ARM_LENGTH * np.cos(arm_angle * 0.3)  # Less vertical motion
            positions[f'{side}_elbow'] = (elbow_x, elbow_z)

            # Forearm
            elbow_angle = getattr(s, f'{side}_elbow')
            forearm_angle = arm_angle + elbow_angle
            positions[f'{side}_wrist'] = (
                elbow_x + self.FOREARM_LENGTH * np.sin(forearm_angle),
                elbow_z - self.FOREARM_LENGTH * np.cos(forearm_angle * 0.3)
            )

        # Head
        positions['head'] = (
            positions['torso_top'][0],
            positions['torso_top'][1] + 0.12
        )

        return positions

    def get_center_of_mass(self) -> Tuple[float, float]:
        """Compute body center of mass."""
        positions = self.get_joint_positions()

        # Approximate CoM of each segment as midpoint
        segments = [
            # (start, end, mass)
            (positions['left_ankle'], positions['left_knee'], self.SHIN_MASS),
            (positions['left_knee'], positions['left_hip'], self.THIGH_MASS),
            (positions['right_ankle'], positions['right_knee'], self.SHIN_MASS),
            (positions['right_knee'], positions['right_hip'], self.THIGH_MASS),
            (positions['torso_base'], positions['torso_top'], self.TORSO_MASS),
            (positions['left_shoulder'], positions['left_elbow'], self.UPPER_ARM_MASS),
            (positions['left_elbow'], positions['left_wrist'], self.FOREARM_MASS),
            (positions['right_shoulder'], positions['right_elbow'], self.UPPER_ARM_MASS),
            (positions['right_elbow'], positions['right_wrist'], self.FOREARM_MASS),
        ]

        total_mass = sum(m for _, _, m in segments)
        com_x = sum((s[0] + e[0]) / 2 * m for s, e, m in segments) / total_mass
        com_z = sum((s[1] + e[1]) / 2 * m for s, e, m in segments) / total_mass

        return com_x, com_z

    def get_hip_height(self) -> float:
        """Get average hip height above board."""
        positions = self.get_joint_positions()
        avg_hip_z = (positions['left_hip'][1] + positions['right_hip'][1]) / 2
        return avg_hip_z - self.state.board_z

    def set_pose_from_leg_extension(self, extension: float):
        """
        Set body pose from a simple leg extension value.

        extension: -1 (crouched) to +1 (extended)
        """
        # Map extension to hip and knee angles
        # Crouched: hip=1.0, knee=2.0
        # Extended: hip=0.0, knee=0.0
        t = (extension + 1) / 2  # 0 to 1

        hip_angle = 1.0 * (1 - t)
        knee_angle = 2.0 * (1 - t)

        self.state.left_hip = hip_angle
        self.state.left_knee = knee_angle
        self.state.right_hip = hip_angle
        self.state.right_knee = knee_angle

        return self.get_hip_height()

    def set_arm_pose(self, left_swing: float, right_swing: float):
        """
        Set arm poses from swing values.

        swing: -1 (back) to +1 (forward)
        """
        # Map to shoulder angle
        self.state.left_shoulder = left_swing * 1.5
        self.state.right_shoulder = right_swing * 1.5

        # Keep elbows slightly bent
        self.state.left_elbow = 0.3
        self.state.right_elbow = 0.3

    def set_waist_lean(self, angle: float, velocity: float = 0.0):
        """
        Set the waist/torso lean angle.

        angle: lean angle in radians (positive = forward lean)
        velocity: lean angular velocity in rad/s
        """
        self.state.waist_lean = angle
        self.state.waist_lean_vel = velocity


def test_body_model():
    """Test and visualize the body model."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    model = BodyModel()

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle('Body Model Test - Different Poses', fontsize=14)

    poses = [
        ("Standing", 0.0, 0.0, 0.0),
        ("Crouched", -0.8, 0.0, 0.0),
        ("Extended", 0.8, 0.0, 0.0),
        ("Pumping", 0.3, 0.8, -0.8),  # legs extended, arms swinging opposite
    ]

    for ax, (name, leg_ext, left_arm, right_arm) in zip(axes, poses):
        model.reset()
        model.set_pose_from_leg_extension(leg_ext)
        model.set_arm_pose(left_arm, right_arm)

        positions = model.get_joint_positions()
        com = model.get_center_of_mass()
        hip_height = model.get_hip_height()

        # Draw body
        # Board
        ax.plot([-0.3, 0.3], [0.2, 0.2], 'brown', linewidth=8, solid_capstyle='round')

        # Legs
        for side, color in [('left', 'blue'), ('right', 'red')]:
            ankle = positions[f'{side}_ankle']
            knee = positions[f'{side}_knee']
            hip = positions[f'{side}_hip']

            ax.plot([ankle[0], knee[0]], [ankle[1], knee[1]], color, linewidth=4)
            ax.plot([knee[0], hip[0]], [knee[1], hip[1]], color, linewidth=5)
            ax.plot(knee[0], knee[1], 'ko', markersize=6)

        # Torso
        ax.plot([positions['torso_base'][0], positions['torso_top'][0]],
                [positions['torso_base'][1], positions['torso_top'][1]],
                'darkgreen', linewidth=8)

        # Arms
        for side, color in [('left', 'blue'), ('right', 'red')]:
            shoulder = positions[f'{side}_shoulder']
            elbow = positions[f'{side}_elbow']
            wrist = positions[f'{side}_wrist']

            ax.plot([shoulder[0], elbow[0]], [shoulder[1], elbow[1]], color, linewidth=3, alpha=0.7)
            ax.plot([elbow[0], wrist[0]], [elbow[1], wrist[1]], color, linewidth=2, alpha=0.7)

        # Head
        head = positions['head']
        ax.add_patch(Circle(head, 0.08, facecolor='peachpuff', edgecolor='black'))

        # CoM marker
        ax.plot(com[0], com[1], 'g*', markersize=15, label=f'CoM')

        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.1, 1.5)
        ax.set_aspect('equal')
        ax.set_title(f'{name}\nHip height: {hip_height:.2f}m')
        ax.axhline(y=0.2, color='blue', alpha=0.3, linestyle='--', label='Water surface')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('body_model_test.png', dpi=150)
    print('Saved: body_model_test.png')

    # Test hip height range
    print('\n=== Hip Height Range ===')
    model.reset()
    for ext in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        h = model.set_pose_from_leg_extension(ext)
        print(f'  Extension {ext:+.1f}: hip height = {h:.2f}m')


if __name__ == '__main__':
    test_body_model()
