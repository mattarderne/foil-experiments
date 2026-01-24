#!/usr/bin/env python
"""
Interactive visualization for the Pump Foil environment.

Usage:
    python visualize.py                    # Watch random policy
    python visualize.py --model MODEL_PATH # Watch trained model
    python visualize.py --manual           # Manual control with keyboard
    python visualize.py --pumping          # Watch sinusoidal pumping
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.animation import FuncAnimation

from foil_env import PumpFoilEnv

try:
    from stable_baselines3 import PPO, SAC
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False


class FoilVisualizer:
    """Real-time visualization of the pump foil simulation."""

    def __init__(self, env, model=None, mode='random', pump_freq=1.5):
        self.env = env
        self.model = model
        self.mode = mode
        self.pump_freq = pump_freq

        # State
        self.obs = None
        self.info = None
        self.step_count = 0
        self.total_reward = 0
        self.episode_count = 0
        self.running = True

        # History for plots
        self.max_history = 300
        self.z_history = []
        self.vx_history = []
        self.power_history = []
        self.action_history = []

        # Manual control
        self.manual_leg = 0.0
        self.manual_pitch = 0.0

        self._setup_figure()

    def _setup_figure(self):
        """Set up the matplotlib figure."""
        self.fig = plt.figure(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('Pump Foil Simulation')

        # Main visualization
        self.ax_main = self.fig.add_subplot(2, 2, 1)
        self.ax_main.set_xlim(-1.5, 1.5)
        self.ax_main.set_ylim(-0.8, 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('Side View')
        self.ax_main.set_xlabel('Position (m)')
        self.ax_main.set_ylabel('Depth (m)')

        # Water surface
        self.ax_main.axhline(y=0, color='blue', linewidth=2, alpha=0.5)
        self.ax_main.fill_between([-2, 2], [0, 0], [0.5, 0.5], color='lightblue', alpha=0.3)

        # Target altitude line
        self.ax_main.axhline(y=0.15, color='green', linestyle='--', alpha=0.5)

        # Altitude history
        self.ax_alt = self.fig.add_subplot(2, 2, 2)
        self.ax_alt.set_xlim(0, self.max_history)
        self.ax_alt.set_ylim(-0.6, 0.4)
        self.ax_alt.axhline(y=0.15, color='green', linestyle='--', alpha=0.5)
        self.ax_alt.axhline(y=0, color='blue', alpha=0.3)
        self.ax_alt.set_title('Altitude History')
        self.ax_alt.set_xlabel('Step')
        self.ax_alt.set_ylabel('Altitude (m)')
        self.ax_alt.grid(True, alpha=0.3)

        # Velocity history
        self.ax_vel = self.fig.add_subplot(2, 2, 3)
        self.ax_vel.set_xlim(0, self.max_history)
        self.ax_vel.set_ylim(0, 8)
        self.ax_vel.axhline(y=4.5, color='green', linestyle='--', alpha=0.5)
        self.ax_vel.set_title('Velocity History')
        self.ax_vel.set_xlabel('Step')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        self.ax_vel.grid(True, alpha=0.3)

        # Info panel
        self.ax_info = self.fig.add_subplot(2, 2, 4)
        self.ax_info.axis('off')

        plt.tight_layout()

        # Connect keyboard events for manual mode
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
        self.fig.canvas.mpl_connect('close_event', self._on_close)

    def _on_key_press(self, event):
        """Handle key press for manual control."""
        if event.key == 'up':
            self.manual_leg = 0.8
        elif event.key == 'down':
            self.manual_leg = -0.8
        elif event.key == 'left':
            self.manual_pitch = -0.5
        elif event.key == 'right':
            self.manual_pitch = 0.5
        elif event.key == 'r':
            self._reset()
        elif event.key == 'q':
            self.running = False
            plt.close(self.fig)

    def _on_key_release(self, event):
        """Handle key release for manual control."""
        if event.key in ['up', 'down']:
            self.manual_leg = 0.0
        elif event.key in ['left', 'right']:
            self.manual_pitch = 0.0

    def _on_close(self, event):
        """Handle window close."""
        self.running = False

    def _reset(self):
        """Reset the environment."""
        self.obs, self.info = self.env.reset()
        self.step_count = 0
        self.total_reward = 0
        self.episode_count += 1
        self.z_history = []
        self.vx_history = []
        self.power_history = []
        self.action_history = []

    def _get_action(self):
        """Get action based on mode."""
        if self.mode == 'manual':
            return np.array([self.manual_leg, self.manual_pitch], dtype=np.float32)
        elif self.mode == 'pumping':
            t = self.step_count * 0.01
            phase = 2 * np.pi * self.pump_freq * t
            leg = 0.6 * np.sin(phase)
            pitch = 0.3 * np.sin(phase + np.pi/4)
            return np.array([leg, pitch], dtype=np.float32)
        elif self.mode == 'model' and self.model is not None:
            action, _ = self.model.predict(self.obs, deterministic=True)
            return action
        else:
            return self.env.action_space.sample()

    def _draw_foil(self, z, theta):
        """Draw the foil assembly."""
        self.ax_main.clear()

        # Water surface
        self.ax_main.axhline(y=0, color='blue', linewidth=2, alpha=0.5)
        self.ax_main.fill_between([-2, 2], [0, 0], [0.5, 0.5], color='lightblue', alpha=0.3)
        self.ax_main.axhline(y=0.15, color='green', linestyle='--', alpha=0.5, label='Target')
        self.ax_main.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3, label='Crash')

        cx, cz = 0, z
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        # Board
        board_len, board_h = 0.6, 0.05
        board = np.array([[-board_len/2, board_h/2], [board_len/2, board_h/2],
                         [board_len/2, -board_h/2], [-board_len/2, -board_h/2]])
        board_rot = (R @ board.T).T + np.array([cx, cz])
        self.ax_main.add_patch(Polygon(board_rot, closed=True, facecolor='saddlebrown', edgecolor='black', linewidth=2))

        # Mast
        mast_len = 0.25
        mast_top = R @ np.array([0.1, 0]) + np.array([cx, cz])
        mast_bot = R @ np.array([0.1, -mast_len]) + np.array([cx, cz])
        self.ax_main.plot([mast_top[0], mast_bot[0]], [mast_top[1], mast_bot[1]], 'k-', linewidth=3)

        # Front wing
        wing_center = R @ np.array([0.1, -mast_len]) + np.array([cx, cz])
        wing = np.array([[-0.15, 0.03], [0.15, 0.03], [0.15, -0.03], [-0.15, -0.03]])
        wing_rot = (R @ wing.T).T + wing_center
        color = 'red' if self.info.get('is_stalled', False) else 'darkblue'
        self.ax_main.add_patch(Polygon(wing_rot, closed=True, facecolor=color, edgecolor='black', alpha=0.8))

        # Stabilizer
        stab_center = R @ np.array([-0.3, -mast_len]) + np.array([cx, cz])
        stab = np.array([[-0.05, 0.015], [0.05, 0.015], [0.05, -0.015], [-0.05, -0.015]])
        stab_rot = (R @ stab.T).T + stab_center
        self.ax_main.add_patch(Polygon(stab_rot, closed=True, facecolor='gray', edgecolor='black', alpha=0.8))

        # Rider (simple stick figure)
        rider_base = R @ np.array([0, 0.05]) + np.array([cx, cz])
        rider_top = R @ np.array([0, 0.35]) + np.array([cx, cz])
        self.ax_main.plot([rider_base[0], rider_top[0]], [rider_base[1], rider_top[1]], 'k-', linewidth=4)
        self.ax_main.add_patch(Circle(rider_top, 0.05, facecolor='peachpuff', edgecolor='black'))

        self.ax_main.set_xlim(-1.5, 1.5)
        self.ax_main.set_ylim(-0.8, 0.5)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title(f'Side View - Episode {self.episode_count}')
        self.ax_main.legend(loc='upper right', fontsize=8)
        self.ax_main.grid(True, alpha=0.3)

    def _update_plots(self):
        """Update the history plots."""
        steps = list(range(len(self.z_history)))

        # Altitude
        self.ax_alt.clear()
        self.ax_alt.set_xlim(0, self.max_history)
        self.ax_alt.set_ylim(-0.6, 0.4)
        self.ax_alt.axhline(y=0.15, color='green', linestyle='--', alpha=0.5)
        self.ax_alt.axhline(y=0, color='blue', alpha=0.3)
        self.ax_alt.axhline(y=-0.5, color='red', linestyle='--', alpha=0.3)
        if steps:
            self.ax_alt.plot(steps, self.z_history, 'b-', linewidth=1.5)
        self.ax_alt.set_title('Altitude')
        self.ax_alt.grid(True, alpha=0.3)

        # Velocity
        self.ax_vel.clear()
        self.ax_vel.set_xlim(0, self.max_history)
        self.ax_vel.set_ylim(0, 8)
        self.ax_vel.axhline(y=4.5, color='green', linestyle='--', alpha=0.5)
        self.ax_vel.axhline(y=1.0, color='red', linestyle='--', alpha=0.3)
        if steps:
            self.ax_vel.plot(steps, self.vx_history, 'b-', linewidth=1.5)
        self.ax_vel.set_title('Velocity')
        self.ax_vel.grid(True, alpha=0.3)

    def _update_info(self, action):
        """Update the info panel."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        info_text = f"""
MODE: {self.mode.upper()}
{'(Arrow keys to control, R to reset, Q to quit)' if self.mode == 'manual' else ''}

STEP: {self.step_count}
TOTAL REWARD: {self.total_reward:.1f}

ALTITUDE: {self.info['z']:.3f} m
VELOCITY: {self.info['vx']:.2f} m/s
PITCH: {self.info['theta_deg']:.1f} deg
AoA: {self.info['alpha_deg']:.1f} deg
STALLED: {self.info['is_stalled']}

POWER: {self.info['power']:.0f} W

ACTIONS:
  Leg Force: {action[0]:.2f}
  Pitch Torque: {action[1]:.2f}
"""
        self.ax_info.text(0.1, 0.9, info_text, transform=self.ax_info.transAxes,
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    def _update(self, frame):
        """Animation update function."""
        if not self.running:
            return

        # Get action
        action = self._get_action()

        # Step environment
        self.obs, reward, terminated, truncated, self.info = self.env.step(action)
        self.step_count += 1
        self.total_reward += reward

        # Update history
        self.z_history.append(self.info['z'])
        self.vx_history.append(self.info['vx'])
        self.power_history.append(self.info['power'])
        self.action_history.append(action.copy())

        if len(self.z_history) > self.max_history:
            self.z_history.pop(0)
            self.vx_history.pop(0)
            self.power_history.pop(0)
            self.action_history.pop(0)

        # Draw
        theta = np.radians(self.info['theta_deg'])
        self._draw_foil(self.info['z'], theta)
        self._update_plots()
        self._update_info(action)

        # Reset if done
        if terminated or truncated:
            print(f"Episode {self.episode_count} ended: {self.info.get('termination_reason', 'truncated')}")
            print(f"  Steps: {self.step_count}, Reward: {self.total_reward:.1f}")
            self._reset()

    def run(self):
        """Run the visualization."""
        self._reset()

        ani = FuncAnimation(self.fig, self._update, interval=10, blit=False, cache_frame_data=False)
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Visualize Pump Foil simulation")
    parser.add_argument('--model', '-m', type=str, help='Path to trained model')
    parser.add_argument('--manual', action='store_true', help='Manual control mode')
    parser.add_argument('--pumping', action='store_true', help='Sinusoidal pumping mode')
    parser.add_argument('--freq', type=float, default=1.5, help='Pumping frequency (Hz)')

    args = parser.parse_args()

    env = PumpFoilEnv()

    model = None
    if args.model:
        if not HAS_SB3:
            print("Error: stable-baselines3 required to load model")
            return
        try:
            model = PPO.load(args.model)
        except:
            model = SAC.load(args.model)
        mode = 'model'
        print(f"Loaded model: {args.model}")
    elif args.manual:
        mode = 'manual'
        print("Manual control mode - use arrow keys")
    elif args.pumping:
        mode = 'pumping'
        print(f"Sinusoidal pumping at {args.freq} Hz")
    else:
        mode = 'random'
        print("Random policy mode")

    viz = FoilVisualizer(env, model=model, mode=mode, pump_freq=args.freq)
    viz.run()

    env.close()


if __name__ == '__main__':
    main()
