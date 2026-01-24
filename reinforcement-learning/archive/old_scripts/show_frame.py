"""Quick script to save a single frame as PNG for preview."""
import sys
sys.path.insert(0, '.')
from animate_foil_proper import run_episode, draw_frame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

frames, dt = run_episode('checkpoints/model_4limb_modal.zip', max_frames=50)
fig, ax = plt.subplots(figsize=(12, 8))

# Show frame 25 (mid-episode) with trails from previous frames
draw_frame(ax, frames, 25)
plt.savefig('foil_frame_preview.png', dpi=150)
print('Saved: foil_frame_preview.png')
