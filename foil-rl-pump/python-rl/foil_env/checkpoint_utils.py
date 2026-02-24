"""
Checkpoint utilities for pump foil RL models.

Saves/loads config alongside model checkpoints for reproducibility.
"""

import os
import warnings
from pathlib import Path
from typing import Tuple, Optional, Any

from stable_baselines3 import PPO

from foil_env.config import TrainingConfig, DEFAULT_CONFIG


def save_checkpoint(
    model: PPO,
    config: TrainingConfig,
    path: str,
) -> str:
    """
    Save model and config together.

    Args:
        model: Trained PPO model
        config: TrainingConfig used for training
        path: Path for model (e.g., "checkpoints/run/model_100pct")
              Config will be saved as "checkpoints/run/model_100pct_config.json"

    Returns:
        Path to saved model (with .zip extension)
    """
    # Normalize path (remove .zip if present)
    path = str(path)
    if path.endswith('.zip'):
        path = path[:-4]

    # Save model
    model.save(path)
    model_path = f"{path}.zip"

    # Save config alongside
    config_path = f"{path}_config.json"
    config.save(config_path)

    return model_path


def load_checkpoint(
    path: str,
    device: str = "auto",
) -> Tuple[PPO, TrainingConfig]:
    """
    Load model and config from checkpoint.

    Args:
        path: Path to model checkpoint (with or without .zip)
        device: Device for model ("auto", "cpu", "cuda")

    Returns:
        Tuple of (model, config)
        For legacy checkpoints without config, returns default config with warning.
    """
    # Normalize path
    path = str(path)
    if path.endswith('.zip'):
        base_path = path[:-4]
    else:
        base_path = path
        path = f"{path}.zip"

    # Load model
    model = PPO.load(path, device=device)

    # Try to load config
    config_path = f"{base_path}_config.json"
    if os.path.exists(config_path):
        config = TrainingConfig.load(config_path)
    else:
        warnings.warn(
            f"Legacy checkpoint '{path}' has no config file. "
            f"Using default config. Model may have been trained with different settings.",
            UserWarning
        )
        config = DEFAULT_CONFIG

    return model, config


def create_env_from_config(
    config: TrainingConfig,
    render_mode: Optional[str] = None,
) -> Any:
    """
    Create environment from config.

    Args:
        config: TrainingConfig with foil and limb settings
        render_mode: Optional render mode ("human", "rgb_array")

    Returns:
        PumpFoilEnvCurriculum instance
    """
    from foil_env.pump_foil_env_curriculum import PumpFoilEnvCurriculum

    return PumpFoilEnvCurriculum(
        config=config.foil.to_dict(),
        limb_config=config.limbs,
        dt=0.01,
        render_mode=render_mode,
        curriculum_phase=config.curriculum_phase,
        agent_blend=config.agent_blend,
    )


def get_config_path(model_path: str) -> str:
    """Get the config path for a given model path."""
    path = str(model_path)
    if path.endswith('.zip'):
        path = path[:-4]
    return f"{path}_config.json"


def has_config(model_path: str) -> bool:
    """Check if a model checkpoint has an associated config file."""
    return os.path.exists(get_config_path(model_path))
