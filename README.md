# dlpracticals

A study aid for the DS1505-P Fundamentals of Deep Learning lab.

## What's inside
- `dlpracticals/simplified/`: short learning versions of experiments 1-10.
- `dlpracticals/keywords/writing_guide.md`: experiment keywords and common code patterns.
- `dlpracticals/reference/`: reference section currently being built; it is not yet a complete manual-matched set.

## Install directly from GitHub
No Git installation or repository clone is required for this command:

```bash
python -m pip install "dlpracticals[all] @ https://github.com/Srivardhni/dlprac/archive/refs/heads/main.zip"
```

TensorFlow installation depends on the Python version and operating system. First check the lab's Python version and follow the version supported by the lab environment if installation fails.

## Run
Run an experiment by number:

```bash
python -m dlpracticals 3
```

Use a different number from 1 to 10 for another experiment. Some experiments need a local dataset folder and will ask you for its path.

In Python:

```python
from dlpracticals.simplified import exp3
exp3.run()
```

## Important limitations
- Experiments 4 and 5 expect local image folders organised by class/person.
- Experiments 6, 8, and 9 use tiny toy examples to demonstrate architecture, not meaningful language models or translators.
- Experiment 7 uses ten sample sentences, so its accuracy is not a real-world performance measure.
- The reference folder is not yet verified against all programs in the lab manual.
- Training results vary with random initialisation, installed library versions, and available hardware.
