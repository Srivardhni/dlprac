# Installation on the college lab computer

## 1. Check Python first
In the VS Code terminal:

```bash
python --version
python -m pip --version
```

If `python` is not recognised on Windows, try `py --version`.

## 2. Install from the GitHub URL
This does not require cloning the repository or installing Git:

```bash
python -m pip install "dlpracticals[all] @ https://github.com/Srivardhni/dlprac/archive/refs/heads/main.zip"
```

The package's `all` extra includes TensorFlow and the optional image/data libraries. TensorFlow wheels depend on Python version and OS, so check compatibility before installing on a shared lab computer.

## 3. Run a practical
For example, Experiment 3:

```bash
python -m dlpracticals 3
```

Replace `3` with a number from 1 to 10. Experiment 4 and 5 ask for a dataset folder path.

## 4. Open the keyword guide
Open `dlpracticals/keywords/writing_guide.md` in the GitHub repository.

## Known limitations
- Experiments 4 and 5 expect local image folders organised by class/person.
- Experiments 6, 8, and 9 are toy architecture demonstrations, not usable language models or a trained translator.
- Experiment 7 has only ten sample sentences, so its accuracy is not meaningful for real-world use.
- `reference/` is still a scaffold and has not been validated against all ten manual programs.
