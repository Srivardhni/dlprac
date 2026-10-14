# Installation on the college lab computer

First check the lab's Python version in VS Code terminal:

```bash
python --version
python -m pip --version
```

If the repo has been cloned:
```bash
python -m pip install -e ".[all]"
```

Run one practical:
```bash
python -m dlpracticals 3
```

Run the keyword list by opening `dlpracticals/keywords/writing_guide.md`.

## Before installing TensorFlow
TensorFlow support depends on Python version, operating system, and available hardware. Do not blindly install the newest TensorFlow on a lab machine. Check the lab's Python version and use the TensorFlow version approved by the course/lab.

## Current limitations
- Experiments 4 and 5 expect local image folders organized by class/person.
- Experiments 6, 8, and 9 use tiny toy examples to demonstrate architecture, not meaningful language models or translators.
- Experiment 7 uses only ten sample sentences, so accuracy is not a real-world performance measure.
- `reference/` is currently a scaffold pending comparison against each exact manual program. Do not treat it as a validated copy of all ten manual solutions.
