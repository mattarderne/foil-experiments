"""
Process a video with MediaPipe pose detection and create skeleton overlays.
Outputs:
  - {name}_skeleton.mp4 - skeleton overlay on original video
  - {name}_dots_trace.mp4 - just dots and trajectory traces on black background
  - {name}_body.npy - raw pose data
  - {name}_features.json - extracted biomechanical features
"""
import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# Body landmark indices (MediaPipe Pose)
BODY_LANDMARKS = {
    'nose': 0, 'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8, 'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12, 'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16, 'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20, 'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24, 'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28, 'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32
}

# Skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (27, 29), (27, 31), (28, 30), (28, 32),  # feet
    (0, 11), (0, 12),  # neck to shoulders (approximate)
]

# Colors for traces (BGR)
COLORS = {
    'left_hand': (255, 100, 100),   # blue
    'right_hand': (100, 100, 255),  # red
    'left_foot': (255, 255, 100),   # cyan
    'right_foot': (100, 255, 255),  # yellow
    'hip_center': (100, 255, 100),  # green
}


def draw_landmarks_on_image(rgb_image, landmarks, width, height):
    """Draw pose landmarks and skeleton on image."""
    annotated_image = np.copy(rgb_image)

    # Draw skeleton connections
    for conn in SKELETON_CONNECTIONS:
        pt1 = (int(landmarks[conn[0], 0]), int(landmarks[conn[0], 1]))
        pt2 = (int(landmarks[conn[1], 0]), int(landmarks[conn[1], 1]))
        cv2.line(annotated_image, pt1, pt2, (0, 255, 0), 2)

    # Draw landmark dots
    for i in range(33):
        pt = (int(landmarks[i, 0]), int(landmarks[i, 1]))
        cv2.circle(annotated_image, pt, 5, (255, 0, 0), -1)

    return annotated_image


def process_video(video_path, output_dir=None, model_path=None):
    """Process video and extract pose data."""
    video_path = Path(video_path)
    if output_dir is None:
        output_dir = video_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if model_path is None:
        model_path = video_path.parent / "pose_landmarker_full.task"

    name = video_path.stem
    # Clean up name for output files
    clean_name = name.split('[')[0].strip().replace(' ', '_').replace('|', '').replace('｜', '')
    if len(clean_name) > 40:
        clean_name = clean_name[:40]

    print(f"Processing: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames ({total_frames/fps:.1f}s)")

    # Create pose landmarker
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)

    # Storage for pose data
    all_landmarks = []

    # Video writers
    skeleton_path = output_dir / f"{clean_name}_skeleton.mp4"
    dots_path = output_dir / f"{clean_name}_dots_trace.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    skeleton_writer = cv2.VideoWriter(str(skeleton_path), fourcc, fps, (width, height))
    dots_writer = cv2.VideoWriter(str(dots_path), fourcc, fps, (width, height))

    # Trajectory histories
    trace_length = int(fps * 2)  # 2 seconds of trace
    trajectories = {k: deque(maxlen=trace_length) for k in COLORS.keys()}

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Process with MediaPipe
        timestamp_ms = int(frame_idx * 1000 / fps)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Extract landmarks
        if results.pose_landmarks and len(results.pose_landmarks) > 0:
            pose_lms = results.pose_landmarks[0]
            landmarks = np.zeros((33, 3))
            for i, lm in enumerate(pose_lms):
                landmarks[i] = [lm.x * width, lm.y * height, lm.z * width]
            all_landmarks.append(landmarks)

            # Get key points for traces
            left_wrist = landmarks[15][:2].astype(int)
            right_wrist = landmarks[16][:2].astype(int)
            left_ankle = landmarks[27][:2].astype(int)
            right_ankle = landmarks[28][:2].astype(int)
            hip_center = ((landmarks[23][:2] + landmarks[24][:2]) / 2).astype(int)

            trajectories['left_hand'].append(tuple(left_wrist))
            trajectories['right_hand'].append(tuple(right_wrist))
            trajectories['left_foot'].append(tuple(left_ankle))
            trajectories['right_foot'].append(tuple(right_ankle))
            trajectories['hip_center'].append(tuple(hip_center))

            # --- Skeleton overlay on original ---
            skeleton_frame = draw_landmarks_on_image(rgb_frame, landmarks, width, height)
            skeleton_frame = cv2.cvtColor(skeleton_frame, cv2.COLOR_RGB2BGR)
            skeleton_writer.write(skeleton_frame)

            # --- Dots and traces on black background ---
            dots_frame = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw traces
            for key, color in COLORS.items():
                points = list(trajectories[key])
                for i in range(1, len(points)):
                    alpha = i / len(points)
                    cv2.line(dots_frame, points[i-1], points[i],
                            tuple(int(c * alpha) for c in color), 2)

            # Draw skeleton
            for conn in SKELETON_CONNECTIONS:
                pt1 = landmarks[conn[0]][:2].astype(int)
                pt2 = landmarks[conn[1]][:2].astype(int)
                cv2.line(dots_frame, tuple(pt1), tuple(pt2), (200, 200, 200), 2)

            # Draw landmark dots
            for i in range(33):
                pt = landmarks[i][:2].astype(int)
                cv2.circle(dots_frame, tuple(pt), 4, (255, 255, 255), -1)

            dots_writer.write(dots_frame)
        else:
            # No pose detected, append zeros
            all_landmarks.append(np.zeros((33, 3)))
            skeleton_writer.write(frame)
            dots_writer.write(np.zeros((height, width, 3), dtype=np.uint8))

        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"  Frame {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.0f}%)")

    cap.release()
    skeleton_writer.release()
    dots_writer.release()
    landmarker.close()

    # Save pose data as numpy array
    pose_data = np.array(all_landmarks)
    npy_path = output_dir / f"{clean_name}_body.npy"
    np.save(str(npy_path), pose_data)
    print(f"  Pose data shape: {pose_data.shape}")

    # Extract and save features
    features = extract_features(pose_data, fps)
    features_path = output_dir / f"{clean_name}_features.json"
    with open(features_path, 'w') as f:
        json.dump(features, f, indent=2)

    print(f"  Outputs:")
    print(f"    {skeleton_path.name}")
    print(f"    {dots_path.name}")
    print(f"    {npy_path.name}")
    print(f"    {features_path.name}")

    return pose_data, features


def extract_features(pose_data, fps):
    """Extract biomechanical features from pose data."""
    from scipy.signal import find_peaks

    n_frames = len(pose_data)
    dt = 1.0 / fps

    # Key landmarks
    left_hip = pose_data[:, 23, :]
    right_hip = pose_data[:, 24, :]
    hip_center = (left_hip + right_hip) / 2

    left_shoulder = pose_data[:, 11, :]
    right_shoulder = pose_data[:, 12, :]
    shoulder_center = (left_shoulder + right_shoulder) / 2

    left_ankle = pose_data[:, 27, :]
    right_ankle = pose_data[:, 28, :]
    ankle_center = (left_ankle + right_ankle) / 2

    left_wrist = pose_data[:, 15, :]
    right_wrist = pose_data[:, 16, :]

    # Velocities
    hip_velocity = np.gradient(hip_center, dt, axis=0)
    ankle_velocity = np.gradient(ankle_center, dt, axis=0)

    # Accelerations
    hip_acceleration = np.gradient(hip_velocity, dt, axis=0)

    # Vertical motion (Y in screen coords)
    hip_y = hip_center[:, 1]
    ankle_y = ankle_center[:, 1]

    # Find pump cycles (local minima/maxima in hip Y)
    peaks, _ = find_peaks(hip_y, distance=int(fps * 0.3))
    valleys, _ = find_peaks(-hip_y, distance=int(fps * 0.3))

    # Calculate pump frequency
    if len(peaks) > 1:
        pump_period = np.mean(np.diff(peaks)) / fps
        pump_frequency = 1.0 / pump_period if pump_period > 0 else 0
    else:
        pump_period = 0
        pump_frequency = 0

    # Body angle (torso lean)
    torso_vector = shoulder_center - hip_center
    body_angle = np.arctan2(torso_vector[:, 0], -torso_vector[:, 1]) * 180 / np.pi

    # Leg extension (hip to ankle distance)
    leg_length = np.linalg.norm(ankle_center - hip_center, axis=1)

    # Arm position relative to body
    left_arm_height = left_wrist[:, 1] - shoulder_center[:, 1]
    right_arm_height = right_wrist[:, 1] - shoulder_center[:, 1]

    features = {
        'metadata': {
            'n_frames': int(n_frames),
            'fps': float(fps),
            'duration_seconds': float(n_frames / fps),
        },
        'pump_analysis': {
            'num_pump_cycles': int(len(peaks)),
            'pump_frequency_hz': float(pump_frequency),
            'pump_period_seconds': float(pump_period),
            'peak_frames': [int(x) for x in peaks.tolist()] if len(peaks) > 0 else [],
            'valley_frames': [int(x) for x in valleys.tolist()] if len(valleys) > 0 else [],
        },
        'statistics': {
            'hip_y_range': float(hip_y.max() - hip_y.min()),
            'hip_y_std': float(np.std(hip_y)),
            'body_angle_mean': float(np.mean(body_angle)),
            'body_angle_std': float(np.std(body_angle)),
            'leg_length_mean': float(np.mean(leg_length)),
            'leg_length_std': float(np.std(leg_length)),
            'hip_velocity_mean': float(np.mean(np.linalg.norm(hip_velocity, axis=1))),
            'hip_velocity_max': float(np.max(np.linalg.norm(hip_velocity, axis=1))),
        },
        'time_series': {
            'hip_y': [float(x) for x in hip_y.tolist()],
            'body_angle': [float(x) for x in body_angle.tolist()],
            'leg_length': [float(x) for x in leg_length.tolist()],
            'hip_velocity_magnitude': [float(x) for x in np.linalg.norm(hip_velocity, axis=1).tolist()],
        }
    }

    return features


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process video with MediaPipe pose detection')
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('-o', '--output', help='Output directory (default: same as video)')
    parser.add_argument('--model', help='Path to pose landmarker model')
    args = parser.parse_args()

    process_video(args.video, args.output, args.model)
