# dlpracticals

A study aid for the DS1505-P Fundamentals of Deep Learning lab.

## Two sections
- `dlpracticals.simplified`: short learning versions designed to make the core logic easier to revise.
- `dlpracticals.reference`: fuller reference programs. Dataset paths and runtime requirements may need adjustment to match the lab machine and manual.

## Install
Python 3.10+ is declared. TensorFlow compatibility varies by Python version and operating system.

```bash
python -m pip install ".[all]"
```

For an editable install while developing:
```bash
python -m pip install -e ".[all]"
```

## Run
```bash
python -m dlpracticals 1
```
Or in Python:
```python
from dlpracticals.simplified import exp3
exp3.run()
```

## Important
- This package does not bypass lab rules.
- Some experiments require datasets or downloads; inspect each module before running.
- Training results vary and toy datasets are not meaningful benchmarks.
- Use the reference version to compare structure, not as a claim that it exactly reproduces every output from the manual.
