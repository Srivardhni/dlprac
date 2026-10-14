from dlpracticals import __version__

def test_version_exists():
    assert __version__

def test_experiment_modules_exist():
    from dlpracticals.simplified import exp1, exp2, exp3, exp4, exp5
    from dlpracticals.simplified import exp6, exp7, exp8, exp9, exp10
    assert all(callable(m.run) for m in (exp1, exp2, exp3, exp4, exp5, exp6, exp7, exp8, exp9, exp10))
