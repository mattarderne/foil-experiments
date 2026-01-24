# Foil physics module
from foil_env.foil_physics import FoilPhysics, FoilState, ForceOutput, CONFIG
from foil_env.foil_config import (
    FoilGeometry,
    PUMP_FOIL_BEGINNER,
    PUMP_FOIL_INTERMEDIATE,
    PUMP_FOIL_ADVANCED,
    PUMP_FOIL_SMALL,
)

__all__ = [
    'FoilPhysics', 'FoilState', 'ForceOutput', 'CONFIG',
    'FoilGeometry',
    'PUMP_FOIL_BEGINNER', 'PUMP_FOIL_INTERMEDIATE',
    'PUMP_FOIL_ADVANCED', 'PUMP_FOIL_SMALL',
]
