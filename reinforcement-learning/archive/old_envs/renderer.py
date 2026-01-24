"""
Foil Renderer - Visualization for the pump foil environment.
"""

import numpy as np
from typing import Optional, Dict, Any

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Polygon
    from matplotlib.transforms import Affine2D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class FoilRenderer:
    """Renderer for visualizing the pump foil simulation."""

    def __init__(self, render_mode: str = "human"):
        """
        Initialize the renderer.

        Args:
            render_mode: "human" for interactive display, "rgb_array" for image output
        """
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for rendering")

        self.render_mode = render_mode
        self.fig = None
        self.ax_main = None
        self.ax_altitude = None
        self.ax_velocity = None
        self.ax_power = None

        # History for plotting
        self.history_len = 500
        self.z_history = []
        self.vx_history = []
        self.power_history = []
        self.alpha_history = []
        self.t_history = []

        self._setup_figure()

    def _setup_figure(self):
        """Set up the matplotlib figure and axes."""
        self.fig = plt.figure(figsize=(14, 8))

        # Main visualization (top left, larger)
        self.ax_main = self.fig.add_subplot(2, 2, (1, 3))
        self.ax_main.set_xlim(-2, 3)
        self.ax_main.set_ylim(-0.8, 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.axhline(y=0, color='blue', linestyle='-', linewidth=2, alpha=0.5)
        self.ax_main.axhline(y=0.2, color='blue', linestyle='--', linewidth=1, alpha=0.3)
        self.ax_main.fill_between([-2, 3], [0, 0], [0.5, 0.5], color='lightblue', alpha=0.3)
        self.ax_main.set_xlabel('Position (m)')
        self.ax_main.set_ylabel('Depth (m)')
        self.ax_main.set_title('Pump Foil Simulation')
        self.ax_main.grid(True, alpha=0.3)

        # Altitude history (top right)
        self.ax_altitude = self.fig.add_subplot(2, 2, 2)
        self.ax_altitude.set_xlim(0, self.history_len)
        self.ax_altitude.set_ylim(-0.3, 0.4)
        self.ax_altitude.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='Target')
        self.ax_altitude.axhline(y=0, color='blue', linestyle='-', alpha=0.3, label='Surface')
        self.ax_altitude.set_xlabel('Step')
        self.ax_altitude.set_ylabel('Altitude (m)')
        self.ax_altitude.set_title('Altitude History')
        self.ax_altitude.legend(loc='upper right')
        self.ax_altitude.grid(True, alpha=0.3)

        # Velocity history (bottom right top)
        self.ax_velocity = self.fig.add_subplot(4, 2, 6)
        self.ax_velocity.set_xlim(0, self.history_len)
        self.ax_velocity.set_ylim(0, 8)
        self.ax_velocity.axhline(y=4.5, color='green', linestyle='--', alpha=0.5, label='Target')
        self.ax_velocity.set_xlabel('Step')
        self.ax_velocity.set_ylabel('Velocity (m/s)')
        self.ax_velocity.legend(loc='upper right')
        self.ax_velocity.grid(True, alpha=0.3)

        # Power history (bottom right bottom)
        self.ax_power = self.fig.add_subplot(4, 2, 8)
        self.ax_power.set_xlim(0, self.history_len)
        self.ax_power.set_ylim(0, 500)
        self.ax_power.axhline(y=400, color='red', linestyle='--', alpha=0.5, label='Max')
        self.ax_power.set_xlabel('Step')
        self.ax_power.set_ylabel('Power (W)')
        self.ax_power.legend(loc='upper right')
        self.ax_power.grid(True, alpha=0.3)

        self.fig.tight_layout()

        if self.render_mode == "human":
            plt.ion()
            plt.show()

    def render(self, state, forces, info: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Render the current state.

        Args:
            state: FoilState object
            forces: ForceOutput object
            info: Additional info dictionary

        Returns:
            RGB array if render_mode is "rgb_array", None otherwise
        """
        # Update history
        self.z_history.append(state.z)
        self.vx_history.append(state.vx)
        self.power_history.append(info.get('power', 0))
        self.alpha_history.append(np.degrees(forces.alpha))

        if len(self.z_history) > self.history_len:
            self.z_history.pop(0)
            self.vx_history.pop(0)
            self.power_history.pop(0)
            self.alpha_history.pop(0)

        # Clear main axis
        self.ax_main.clear()
        self._setup_main_axis()

        # Draw foil
        self._draw_foil(state, forces)

        # Draw force vectors
        self._draw_forces(state, forces)

        # Update history plots
        self._update_history_plots()

        # Add info text
        info_text = (
            f"Step: {info.get('step_count', 0)}\n"
            f"Alt: {state.z:.3f}m\n"
            f"Vel: {state.vx:.2f}m/s\n"
            f"AoA: {np.degrees(forces.alpha):.1f}deg\n"
            f"Power: {info.get('power', 0):.0f}W\n"
            f"Stalled: {forces.is_stalled}"
        )
        self.ax_main.text(
            2.5, 0.4, info_text,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        if self.render_mode == "rgb_array":
            # Convert to RGB array
            self.fig.canvas.draw()
            buf = self.fig.canvas.buffer_rgba()
            image = np.asarray(buf)
            return image[:, :, :3]  # Remove alpha channel

        return None

    def _setup_main_axis(self):
        """Set up the main visualization axis."""
        self.ax_main.set_xlim(-2, 3)
        self.ax_main.set_ylim(-0.8, 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.axhline(y=0, color='blue', linestyle='-', linewidth=2, alpha=0.5)
        self.ax_main.axhline(y=0.2, color='blue', linestyle='--', linewidth=1, alpha=0.3)
        self.ax_main.fill_between([-2, 3], [0, 0], [0.5, 0.5], color='lightblue', alpha=0.3)
        self.ax_main.set_xlabel('Position (m)')
        self.ax_main.set_ylabel('Depth (m)')
        self.ax_main.set_title('Pump Foil Simulation')
        self.ax_main.grid(True, alpha=0.3)

    def _draw_foil(self, state, forces):
        """Draw the foil assembly."""
        # Center position (we draw at x=0 for simplicity, state.x is used elsewhere)
        cx, cz = 0, state.z
        theta = state.theta

        # Board (simplified rectangle)
        board_len = 0.6
        board_height = 0.05

        # Mast
        mast_len = 0.25
        mast_x = 0.1

        # Front wing
        wing_span = 0.3
        wing_chord = 0.15

        # Draw board
        board_corners = np.array([
            [-board_len/2, board_height/2],
            [board_len/2, board_height/2],
            [board_len/2, -board_height/2],
            [-board_len/2, -board_height/2],
        ])

        # Rotation matrix
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Transform board
        board_rot = (R @ board_corners.T).T
        board_rot[:, 0] += cx
        board_rot[:, 1] += cz

        board_patch = Polygon(board_rot, closed=True, facecolor='brown', edgecolor='black', linewidth=2)
        self.ax_main.add_patch(board_patch)

        # Draw mast
        mast_top = np.array([mast_x, 0])
        mast_bottom = np.array([mast_x, -mast_len])
        mast_top_rot = R @ mast_top + np.array([cx, cz])
        mast_bottom_rot = R @ mast_bottom + np.array([cx, cz])
        self.ax_main.plot(
            [mast_top_rot[0], mast_bottom_rot[0]],
            [mast_top_rot[1], mast_bottom_rot[1]],
            'k-', linewidth=3
        )

        # Draw front wing
        wing_center = np.array([mast_x, -mast_len])
        wing_center_rot = R @ wing_center + np.array([cx, cz])

        wing_corners = np.array([
            [-wing_span/2, wing_chord/2],
            [wing_span/2, wing_chord/2],
            [wing_span/2, -wing_chord/2],
            [-wing_span/2, -wing_chord/2],
        ])
        # Wing rotated by board angle
        wing_rot = (R @ wing_corners.T).T
        wing_rot[:, 0] += wing_center_rot[0]
        wing_rot[:, 1] += wing_center_rot[1]

        # Color based on stall
        wing_color = 'red' if forces.is_stalled else 'darkblue'
        wing_patch = Polygon(wing_rot, closed=True, facecolor=wing_color, edgecolor='black', alpha=0.7)
        self.ax_main.add_patch(wing_patch)

        # Draw stabilizer
        stab_offset = 0.4
        stab_center = np.array([mast_x - stab_offset, -mast_len])
        stab_center_rot = R @ stab_center + np.array([cx, cz])

        stab_span = 0.1
        stab_chord = 0.04
        stab_corners = np.array([
            [-stab_span/2, stab_chord/2],
            [stab_span/2, stab_chord/2],
            [stab_span/2, -stab_chord/2],
            [-stab_span/2, -stab_chord/2],
        ])
        stab_rot = (R @ stab_corners.T).T
        stab_rot[:, 0] += stab_center_rot[0]
        stab_rot[:, 1] += stab_center_rot[1]

        stab_patch = Polygon(stab_rot, closed=True, facecolor='gray', edgecolor='black', alpha=0.7)
        self.ax_main.add_patch(stab_patch)

    def _draw_forces(self, state, forces):
        """Draw force vectors."""
        cx, cz = 0, state.z
        scale = 0.001  # Scale forces for visualization

        # Lift (blue, perpendicular to velocity)
        if abs(forces.Lz) > 1:
            self.ax_main.arrow(
                cx, cz, forces.Lx * scale, forces.Lz * scale,
                head_width=0.03, head_length=0.02, fc='blue', ec='blue', alpha=0.7
            )

        # Drag (red, opposite to velocity)
        if abs(forces.Dx) > 1:
            self.ax_main.arrow(
                cx, cz, forces.Dx * scale, forces.Dz * scale,
                head_width=0.03, head_length=0.02, fc='red', ec='red', alpha=0.7
            )

        # Net thrust (green, horizontal)
        if abs(forces.F_hydro_x) > 1:
            self.ax_main.arrow(
                cx - 0.5, cz, forces.F_hydro_x * scale * 2, 0,
                head_width=0.03, head_length=0.02, fc='green', ec='green', alpha=0.9
            )

    def _update_history_plots(self):
        """Update the history plots."""
        steps = list(range(len(self.z_history)))

        # Altitude
        self.ax_altitude.clear()
        self.ax_altitude.set_xlim(0, self.history_len)
        self.ax_altitude.set_ylim(-0.3, 0.4)
        self.ax_altitude.axhline(y=0.15, color='green', linestyle='--', alpha=0.5)
        self.ax_altitude.axhline(y=0, color='blue', linestyle='-', alpha=0.3)
        self.ax_altitude.plot(steps, self.z_history, 'b-', linewidth=1)
        self.ax_altitude.set_xlabel('Step')
        self.ax_altitude.set_ylabel('Altitude (m)')
        self.ax_altitude.grid(True, alpha=0.3)

        # Velocity
        self.ax_velocity.clear()
        self.ax_velocity.set_xlim(0, self.history_len)
        self.ax_velocity.set_ylim(0, 8)
        self.ax_velocity.axhline(y=4.5, color='green', linestyle='--', alpha=0.5)
        self.ax_velocity.plot(steps, self.vx_history, 'b-', linewidth=1)
        self.ax_velocity.set_xlabel('Step')
        self.ax_velocity.set_ylabel('Velocity (m/s)')
        self.ax_velocity.grid(True, alpha=0.3)

        # Power
        self.ax_power.clear()
        self.ax_power.set_xlim(0, self.history_len)
        self.ax_power.set_ylim(0, 500)
        self.ax_power.axhline(y=400, color='red', linestyle='--', alpha=0.5)
        self.ax_power.plot(steps, self.power_history, 'orange', linewidth=1)
        self.ax_power.set_xlabel('Step')
        self.ax_power.set_ylabel('Power (W)')
        self.ax_power.grid(True, alpha=0.3)

    def close(self):
        """Close the renderer."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None

    def reset(self):
        """Reset the history."""
        self.z_history = []
        self.vx_history = []
        self.power_history = []
        self.alpha_history = []
