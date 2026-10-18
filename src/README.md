The router is split so each file does one job. `main.py` is the command.

- `probe.py` reads a fixed number of frames, however long the file is.
- `features.py` turns the probe into the cut, pan, leftover-motion, and speech clues.
- `embed.py` runs a frozen MobileNetV3-Small on the 8 spread frames and averages them.
- `detect.py` runs a frozen YOLO11n on those same frames at a clearer size and counts people, screens, animals, and sports gear.
- `model.py` shrinks the MobileNet average to 16 numbers and fits logistic regression. `compare` scores MobileNet, YOLO, and both.
- `chunkers.py` names the pipeline each class should run, and the cost of a wrong choice.
- `labels.py` maps a Video-MME subcategory to a starter chunking label and a holdout split.
- `metrics.py` reports macro-F1, routing cost, and a bootstrap interval for both.
- `synthetic.py` builds the four tiny videos used by the tests.
- `config.py` holds the shared numbers.
