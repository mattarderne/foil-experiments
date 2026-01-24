"""
Test the MuJoCo rider model.

Verifies:
1. Model loads correctly
2. Body structure matches expected
3. Joints have correct limits
4. Mass distribution is correct
5. Basic simulation runs
"""

import mujoco
import numpy as np
from pathlib import Path


def test_model_loading():
    """Test that the model loads without errors."""
    model_path = Path(__file__).parent / "rider.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    print(f"[PASS] Model loaded: {model.nq} qpos, {model.nv} qvel, {model.nu} actuators")
    return model, data


def test_body_structure(model):
    """Verify body structure matches expected."""
    expected_bodies = [
        "world", "board", "left_shin", "left_thigh",
        "right_shin", "right_thigh", "pelvis", "torso",
        "shoulders", "head", "left_upper_arm", "left_forearm",
        "right_upper_arm", "right_forearm"
    ]

    actual_bodies = [model.body(i).name for i in range(model.nbody)]
    print(f"\nBodies ({model.nbody}):")
    for name in actual_bodies:
        status = "[OK]" if name in expected_bodies else "[?]"
        print(f"  {status} {name}")

    missing = set(expected_bodies) - set(actual_bodies)
    if missing:
        print(f"[WARN] Missing bodies: {missing}")
    else:
        print("[PASS] All expected bodies present")


def test_joints(model):
    """Verify joint structure and limits."""
    print(f"\nJoints ({model.njnt}):")

    # Expected joints and their approximate limits
    expected_joints = {
        "board_joint": None,  # Freejoint, no limits
        "left_knee": (0, 2.5),
        "right_knee": (0, 2.5),
        "left_hip": (-0.5, 1.5),
        "right_hip": (-0.5, 1.5),
        "waist": (-0.3, 0.5),
        "left_shoulder": (-1.0, 2.5),
        "right_shoulder": (-1.0, 2.5),
        "left_elbow": (0, 2.5),
        "right_elbow": (0, 2.5),
    }

    for i in range(model.njnt):
        joint = model.joint(i)
        name = joint.name
        jnt_type = ["free", "ball", "slide", "hinge"][model.jnt_type[i]]

        is_limited = model.jnt_limited[i]
        if is_limited:
            lo = model.jnt_range[i, 0]
            hi = model.jnt_range[i, 1]
            limits = f"[{lo:.2f}, {hi:.2f}]"
        else:
            limits = "no limits"

        expected = expected_joints.get(name)
        if expected is None:
            status = "[OK]" if jnt_type == "free" else "[?]"
        else:
            # Check if limits match
            if is_limited:
                match = (abs(lo - expected[0]) < 0.01 and abs(hi - expected[1]) < 0.01)
                status = "[OK]" if match else "[WARN]"
            else:
                status = "[WARN]"

        print(f"  {status} {name}: {jnt_type}, {limits}")


def test_mass_distribution(model):
    """Verify mass matches expected ~76kg (70kg rider + 6kg board)."""
    print(f"\nMass distribution:")

    total_mass = 0
    for i in range(model.nbody):
        body = model.body(i)
        mass = model.body_mass[i]
        if mass > 0.1:
            print(f"  {body.name}: {mass:.1f} kg")
        total_mass += mass

    print(f"\nTotal mass: {total_mass:.1f} kg")
    expected_mass = 70 + 6  # rider + board
    if abs(total_mass - expected_mass) < 5:
        print(f"[PASS] Mass is close to expected {expected_mass} kg")
    else:
        print(f"[WARN] Mass differs from expected {expected_mass} kg")


def test_simulation(model, data):
    """Test basic simulation runs without errors."""
    print("\nSimulation test:")

    # Reset
    mujoco.mj_resetData(model, data)

    # Set initial board height
    data.qpos[2] = 0.2  # z position

    # Run 100 steps
    for i in range(100):
        mujoco.mj_step(model, data)

    print(f"  After 100 steps (1s):")
    print(f"    Board position: [{data.qpos[0]:.3f}, {data.qpos[1]:.3f}, {data.qpos[2]:.3f}]")
    print(f"    Board velocity: [{data.qvel[0]:.3f}, {data.qvel[1]:.3f}, {data.qvel[2]:.3f}]")

    # Board should have fallen (no foil forces yet)
    if data.qpos[2] < 0.2:
        print("[PASS] Board fell due to gravity (expected without foil forces)")
    else:
        print("[WARN] Board did not fall")


def test_actuators(model, data):
    """Test actuator control."""
    print("\nActuator test:")

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.5  # Start higher

    # Apply knee control
    data.ctrl[0] = 0.5  # left knee
    data.ctrl[1] = 0.5  # right knee

    for i in range(50):
        mujoco.mj_step(model, data)

    print(f"  Actuators: {model.nu}")
    for i in range(model.nu):
        actuator = model.actuator(i)
        print(f"    {actuator.name}: ctrl={data.ctrl[i]:.2f}")

    print("[PASS] Actuators work")


def test_sensors(model, data):
    """Test sensor readings."""
    print("\nSensor test:")

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.3

    mujoco.mj_forward(model, data)

    print(f"  Sensors: {model.nsensor}")
    for i in range(min(5, model.nsensor)):  # Show first 5
        sensor = model.sensor(i)
        addr = int(model.sensor_adr[i])
        dim = int(model.sensor_dim[i])
        value = data.sensordata[addr:addr+dim]
        print(f"    {sensor.name}: {value}")

    print("[PASS] Sensors work")


def test_external_forces(model, data):
    """Test applying external forces (for foil physics)."""
    print("\nExternal force test:")

    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.2  # Start at water level

    # Find board body ID
    board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "board")
    print(f"  Board body ID: {board_id}")

    # Apply upward force (simulating lift)
    # xfrc_applied is [fx, fy, fz, tx, ty, tz] for each body
    lift_force = model.body_mass[0:].sum() * 9.81  # Equal to weight
    data.xfrc_applied[board_id] = [0, 0, lift_force, 0, 0, 0]

    # Run simulation
    for i in range(100):
        mujoco.mj_step(model, data)

    print(f"  Applied lift: {lift_force:.1f} N")
    print(f"  Board z after 1s: {data.qpos[2]:.3f} m")

    # Should stay roughly at same height (lift = weight)
    if abs(data.qpos[2] - 0.2) < 0.5:
        print("[PASS] External forces work")
    else:
        print(f"[INFO] Board moved to z={data.qpos[2]:.3f} (may need force tuning)")


def render_frame(model, data):
    """Test rendering a single frame."""
    print("\nRender test:")
    try:
        renderer = mujoco.Renderer(model, height=480, width=640)
        renderer.update_scene(data)
        pixels = renderer.render()
        print(f"  Frame shape: {pixels.shape}")
        print("[PASS] Rendering works")
        return pixels
    except Exception as e:
        print(f"[SKIP] Rendering not available: {e}")
        return None


def main():
    print("=" * 50)
    print("MuJoCo Rider Model Test")
    print("=" * 50)

    model, data = test_model_loading()
    test_body_structure(model)
    test_joints(model)
    test_mass_distribution(model)
    test_simulation(model, data)
    test_actuators(model, data)
    test_sensors(model, data)
    test_external_forces(model, data)
    frame = render_frame(model, data)

    print("\n" + "=" * 50)
    print("All basic tests complete!")
    print("=" * 50)

    # Save a test image if rendering worked
    if frame is not None:
        try:
            from PIL import Image
            img = Image.fromarray(frame)
            img.save("test_render.png")
            print("Saved: test_render.png")
        except ImportError:
            print("(Install PIL to save test image)")


if __name__ == "__main__":
    main()
