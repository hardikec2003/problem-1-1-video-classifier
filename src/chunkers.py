"""The chunker each class selects, and the damage of picking the wrong one."""

PIPELINES = {
    "gameplay": (
        "Kuration gameplay. Windows of about 4 seconds. "
        "Run pixel_shift, action-state scoring, and collision only when a "
        "key or mouse log is joined to the video. Otherwise keep the windows "
        "and skip those operators."
    ),
    "egocentric": (
        "Kuration SLAM. Windows of about 4 seconds. "
        "Run slam_reconstruct, slam_quality, and slam_motion_diversity. "
        "Drop windows whose track is broken."
    ),
    "dialogue": (
        "Speech turns. Split where silence lasts longer than about half a "
        "second, and keep one frame as the visual anchor of each turn."
    ),
    "shot_scene": (
        "Detect hard cuts, then group neighboring shots that share a place. "
        "This covers film, wildlife packages, and broadcast sports."
    ),
    "uncertain": (
        "Safe hybrid: detect cuts and speech turns, and do not run a Kuration "
        "operator. Used when the classifier is unsure."
    ),
}

# Rows are the true class. A wrong gameplay-to-dialogue route indexes talk
# and drops control-conditioned moments. A wrong film-to-egocentric route
# makes SLAM treat every cut as impossible motion.
ROUTING_COST = {
    "gameplay": {"gameplay": 0.0, "egocentric": 0.6, "dialogue": 1.0, "shot_scene": 0.7},
    "egocentric": {"gameplay": 0.6, "egocentric": 0.0, "dialogue": 0.7, "shot_scene": 0.8},
    "dialogue": {"gameplay": 0.9, "egocentric": 0.6, "dialogue": 0.0, "shot_scene": 0.4},
    "shot_scene": {"gameplay": 0.7, "egocentric": 0.9, "dialogue": 0.4, "shot_scene": 0.0},
}


# The hybrid chunker runs cuts and speech together. It is cheap for dialogue
# and edited video, and only a partial miss for gameplay and egocentric footage.
HYBRID_COST = {
    "gameplay": 0.55,
    "egocentric": 0.55,
    "dialogue": 0.15,
    "shot_scene": 0.15,
}


def chunker_for(label: str) -> str:
    return PIPELINES[label]
