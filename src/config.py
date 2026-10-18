"""Shared settings for the probe, the split, and the router."""

N_SPREAD_FRAMES = 8
N_BURST_FRAMES = 8
BURST_START_FRAC = 0.20
SHORT_SIDE = 96
# YOLO reads the same eight spread frames at this size. The 96-pixel probe
# is enough for cuts and pans, and too small for a detector.
DETECT_SHORT_SIDE = 320
CUT_SCAN_SECONDS = 30
CUT_JUMP = 0.50

CLASSES = ("gameplay", "egocentric", "dialogue", "shot_scene")

# Below this confidence the caller should use the hybrid chunker.
DEFER_THRESHOLD = 0.45

VIDEOS_PER_CLASS = 8
HOLDOUT_FROM = 6
SYNTHETIC_FPS = 8
SYNTHETIC_FRAMES = 24
SYNTHETIC_SIZE = 96

DATASET_ID = "aintropy-ai/video-mme-sample10"
BOOTSTRAP_ROUNDS = 1000
APPEARANCE_DIMS = 16
