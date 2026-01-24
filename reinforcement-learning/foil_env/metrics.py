"""
Metrics for analyzing pump foil behavior.

Detects actual oscillation (not just min-max range) and coordination.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import signal


def analyze_pumping(
    positions: np.ndarray,
    dt: float,
    min_freq: float = 0.5,
    max_freq: float = 5.0,
) -> Dict:
    """
    Analyze pumping behavior using FFT to detect oscillation.

    Args:
        positions: Array of limb positions over time
        dt: Time step between samples
        min_freq: Minimum frequency to consider (Hz)
        max_freq: Maximum frequency to consider (Hz)

    Returns:
        Dict with:
            - frequency: Dominant oscillation frequency (Hz)
            - amplitude: Oscillation amplitude (half peak-to-peak)
            - is_oscillating: True if actual oscillation detected
            - quality: Oscillation quality score (0-1)
            - raw_range: Simple min-max range (for comparison)
    """
    positions = np.asarray(positions)
    n = len(positions)

    if n < 10:
        return {
            'frequency': 0.0,
            'amplitude': 0.0,
            'is_oscillating': False,
            'quality': 0.0,
            'raw_range': 0.0,
        }

    # Raw range (old metric)
    raw_range = np.max(positions) - np.min(positions)

    # Detrend to remove drift
    detrended = signal.detrend(positions, type='linear')

    # FFT analysis
    fs = 1.0 / dt  # Sample rate
    freqs = np.fft.rfftfreq(n, dt)
    fft_mag = np.abs(np.fft.rfft(detrended))

    # Find frequencies in valid range
    valid_mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid_mask):
        return {
            'frequency': 0.0,
            'amplitude': 0.0,
            'is_oscillating': False,
            'quality': 0.0,
            'raw_range': raw_range,
        }

    valid_freqs = freqs[valid_mask]
    valid_mags = fft_mag[valid_mask]

    # Find dominant frequency
    peak_idx = np.argmax(valid_mags)
    dominant_freq = valid_freqs[peak_idx]
    peak_mag = valid_mags[peak_idx]

    # Calculate amplitude from FFT magnitude
    # FFT magnitude = N/2 * amplitude for a pure sine
    amplitude = 2 * peak_mag / n

    # Quality: ratio of peak power to total power in valid range
    total_power = np.sum(valid_mags ** 2)
    peak_power = peak_mag ** 2
    quality = peak_power / (total_power + 1e-10)

    # Determine if actually oscillating
    # Need: sufficient amplitude, reasonable quality, multiple cycles
    duration = n * dt
    expected_cycles = duration * dominant_freq
    is_oscillating = (
        amplitude > 0.01 and  # At least 1cm for legs, 0.01 rad for arms
        quality > 0.2 and     # At least 20% of power in dominant freq
        expected_cycles > 1.5  # At least 1.5 cycles observed
    )

    return {
        'frequency': float(dominant_freq),
        'amplitude': float(amplitude),
        'is_oscillating': bool(is_oscillating),
        'quality': float(quality),
        'raw_range': float(raw_range),
    }


def analyze_coordination(
    leg_pos: np.ndarray,
    arm_pos: np.ndarray,
    dt: float,
) -> Dict:
    """
    Analyze coordination between leg and arm movements.

    Args:
        leg_pos: Leg position over time
        arm_pos: Arm position over time
        dt: Time step

    Returns:
        Dict with:
            - phase_offset: Phase difference (radians, 0 = in sync)
            - velocity_correlation: Correlation of velocities (-1 to 1)
            - is_coordinated: True if movements are coordinated
            - correlation_abs: Absolute correlation (coordination strength)
    """
    leg_pos = np.asarray(leg_pos)
    arm_pos = np.asarray(arm_pos)
    n = min(len(leg_pos), len(arm_pos))

    if n < 20:
        return {
            'phase_offset': 0.0,
            'velocity_correlation': 0.0,
            'is_coordinated': False,
            'correlation_abs': 0.0,
        }

    leg_pos = leg_pos[:n]
    arm_pos = arm_pos[:n]

    # Compute velocities
    leg_vel = np.diff(leg_pos) / dt
    arm_vel = np.diff(arm_pos) / dt

    # Correlation of velocities
    leg_vel_centered = leg_vel - np.mean(leg_vel)
    arm_vel_centered = arm_vel - np.mean(arm_vel)

    leg_std = np.std(leg_vel_centered)
    arm_std = np.std(arm_vel_centered)

    if leg_std < 1e-6 or arm_std < 1e-6:
        return {
            'phase_offset': 0.0,
            'velocity_correlation': 0.0,
            'is_coordinated': False,
            'correlation_abs': 0.0,
        }

    correlation = np.mean(leg_vel_centered * arm_vel_centered) / (leg_std * arm_std)

    # Phase offset via cross-correlation
    cross_corr = np.correlate(leg_vel_centered, arm_vel_centered, mode='full')
    lag_idx = np.argmax(np.abs(cross_corr)) - (len(arm_vel_centered) - 1)
    lag_time = lag_idx * dt

    # Estimate phase offset assuming ~2Hz pumping
    estimated_period = 0.5  # 2Hz
    phase_offset = (lag_time / estimated_period) * 2 * np.pi
    phase_offset = np.mod(phase_offset + np.pi, 2 * np.pi) - np.pi  # Wrap to [-π, π]

    # Coordinated if correlation is strong (positive or negative)
    is_coordinated = abs(correlation) > 0.3

    return {
        'phase_offset': float(phase_offset),
        'velocity_correlation': float(correlation),
        'is_coordinated': bool(is_coordinated),
        'correlation_abs': float(abs(correlation)),
    }


def analyze_episode(
    frames: List[Dict],
    dt: float = 0.01,
    max_leg_ext: float = 0.15,
    max_arm_swing: float = 2.0,
) -> Dict:
    """
    Full analysis of an episode.

    Args:
        frames: List of frame dicts with 't', 'z', 'vx', 'left_leg', 'right_leg',
                'left_arm', 'right_arm', 'waist' keys
        dt: Time step
        max_leg_ext: Maximum leg extension (for normalization)
        max_arm_swing: Maximum arm swing (for normalization)

    Returns:
        Comprehensive metrics dict
    """
    if not frames:
        return {'duration': 0.0, 'error': 'no frames'}

    # Extract arrays
    leg_pos = np.array([(f['left_leg'] + f['right_leg']) / 2 for f in frames])
    arm_pos = np.array([(f['left_arm'] + f['right_arm']) / 2 for f in frames])
    altitudes = np.array([f['z'] for f in frames])
    velocities = np.array([f['vx'] for f in frames])

    duration = len(frames) * dt

    # Analyze leg pumping
    leg_metrics = analyze_pumping(leg_pos, dt)

    # Analyze arm movement
    arm_metrics = analyze_pumping(arm_pos, dt)

    # Analyze coordination
    coord_metrics = analyze_coordination(leg_pos, arm_pos, dt)

    # Normalized metrics
    leg_range_pct = (leg_metrics['raw_range'] / (2 * max_leg_ext)) * 100
    leg_amplitude_pct = (leg_metrics['amplitude'] / max_leg_ext) * 100

    arm_range_deg = np.degrees(arm_metrics['raw_range'])
    arm_amplitude_deg = np.degrees(arm_metrics['amplitude'])

    return {
        'duration': duration,

        # Leg metrics
        'leg_frequency': leg_metrics['frequency'],
        'leg_amplitude': leg_metrics['amplitude'],
        'leg_amplitude_pct': leg_amplitude_pct,
        'leg_range_pct': leg_range_pct,  # Old metric for comparison
        'leg_is_oscillating': leg_metrics['is_oscillating'],
        'leg_quality': leg_metrics['quality'],

        # Arm metrics
        'arm_frequency': arm_metrics['frequency'],
        'arm_amplitude_deg': arm_amplitude_deg,
        'arm_range_deg': arm_range_deg,  # Old metric for comparison
        'arm_is_oscillating': arm_metrics['is_oscillating'],
        'arm_quality': arm_metrics['quality'],

        # Coordination
        'phase_offset': coord_metrics['phase_offset'],
        'velocity_correlation': coord_metrics['velocity_correlation'],
        'is_coordinated': coord_metrics['is_coordinated'],

        # Flight metrics
        'avg_altitude': float(np.mean(altitudes)),
        'altitude_std': float(np.std(altitudes)),
        'avg_velocity': float(np.mean(velocities)),
        'velocity_std': float(np.std(velocities)),
    }


def format_metrics(metrics: Dict, brief: bool = False) -> str:
    """Format metrics for display."""
    if 'error' in metrics:
        return f"Error: {metrics['error']}"

    if brief:
        leg_status = "✓ pumping" if metrics.get('leg_is_oscillating') else "✗ static"
        arm_status = "✓ oscillating" if metrics.get('arm_is_oscillating') else "✗ drifted"
        return (
            f"{metrics['duration']:.1f}s | "
            f"leg: {metrics['leg_frequency']:.1f}Hz {leg_status} | "
            f"arm: {arm_status}"
        )

    lines = [
        f"Duration: {metrics['duration']:.1f}s",
        "",
        "LEG PUMPING:",
        f"  Frequency: {metrics['leg_frequency']:.1f} Hz",
        f"  Amplitude: {metrics['leg_amplitude_pct']:.0f}% (oscillation)",
        f"  Range:     {metrics['leg_range_pct']:.0f}% (min-max)",
        f"  Quality:   {metrics['leg_quality']:.0%}",
        f"  Status:    {'✓ Pumping' if metrics['leg_is_oscillating'] else '✗ Not pumping'}",
        "",
        "ARM MOVEMENT:",
        f"  Frequency: {metrics['arm_frequency']:.1f} Hz",
        f"  Amplitude: {metrics['arm_amplitude_deg']:.0f}° (oscillation)",
        f"  Range:     {metrics['arm_range_deg']:.0f}° (min-max drift)",
        f"  Quality:   {metrics['arm_quality']:.0%}",
        f"  Status:    {'✓ Oscillating' if metrics['arm_is_oscillating'] else '✗ Drifted/static'}",
        "",
        "COORDINATION:",
        f"  Velocity correlation: {metrics['velocity_correlation']:.2f}",
        f"  Phase offset: {np.degrees(metrics['phase_offset']):.0f}°",
        f"  Status: {'✓ Coordinated' if metrics['is_coordinated'] else '✗ Not coordinated'}",
    ]
    return "\n".join(lines)
