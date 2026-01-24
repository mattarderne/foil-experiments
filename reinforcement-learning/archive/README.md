# Archive - Reference Only

These scripts are archived for reference. They are NOT the standard video format.

## Archived Scripts

- `animate_learning.py` - Old single-panel learning animation
- `animate_4limb.py` - 2-panel GIF animation (body + metrics)
- `animate_foil_proper.py` - Single-panel foil visualization (useful drawing code, but not standard format)
- `create_waist_video.py` - 4-panel but single model only (not comparison)
- `create_standard_video.py` - Attempted standard but wrong layout

## Standard Video Format

Use `create_comparison_video.py` (in main folder) which shows:
- **Top row**: 4 foil visualizations side-by-side (different stages/methods)
- **Bottom row**: Time series plots (altitude, velocity, controls)

This format allows comparing baseline vs manual vs trained models in a single video.

Also see `animate_stance_learning.py` for the original implementation of this format.
