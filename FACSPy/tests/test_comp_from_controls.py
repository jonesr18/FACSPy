import numpy as np

from FACSPy.transforms._comp_from_controls import (
    compute_spill_coefficients,
    generate_compensation_from_controls,
)


class SimpleFake:
    def __init__(self, events, channels):
        import pandas as pd
        self.original_events = events
        self.channels = pd.DataFrame(index=channels)


def test_compute_spill_coefficients_median_ratio_simple():
    # Create synthetic data: 3 channels, control for channel 0 with spill to 1 and 2
    n = 1000
    primary = np.random.normal(loc=1000, scale=50, size=n)
    ch1 = 0.1 * primary + np.random.normal(scale=5, size=n)
    ch2 = 0.05 * primary + np.random.normal(scale=2, size=n)
    events = np.vstack([primary, ch1, ch2]).T

    coefs = compute_spill_coefficients(events, primary_idx=0, method='median_ratio')
    assert np.isclose(coefs[0], 1.0)
    assert np.isclose(coefs[1], 0.1, atol=0.01)
    assert np.isclose(coefs[2], 0.05, atol=0.01)


def test_generate_compensation_from_controls_basic_integration():
    # two single-color controls for ch0 and ch1
    n = 500
    p0 = np.random.normal(1000, 30, size=n)
    p1 = np.random.normal(900, 40, size=n)
    ch0 = np.vstack([p0, 0.05 * p0 + np.random.normal(scale=1, size=n), np.random.normal(scale=1, size=n)]).T
    ch1 = np.vstack([np.random.normal(scale=1, size=n), p1, 0.02 * p1 + np.random.normal(scale=0.5, size=n)]).T

    s0 = SimpleFake(ch0, ['A', 'B', 'C'])
    s1 = SimpleFake(ch1, ['A', 'B', 'C'])

    controls = {'FluorA': s0, 'FluorB': s1}
    primary_map = {'FluorA': 'A', 'FluorB': 'B'}

    mat = generate_compensation_from_controls(controls, primary_map, method='median_ratio', min_events=10)
    df = mat.as_dataframe()
    # diagonal should be 1
    assert np.isclose(df.loc['A', 'A'], 1.0)
    assert np.isclose(df.loc['B', 'B'], 1.0)
    # off-diagonals near expected ranges
    assert 0.03 < df.loc['B', 'A'] < 0.07  # B receives ~5% from A
    assert 0.01 < df.loc['C', 'B'] < 0.03  # C receives ~2% from B


def test_generate_compensation_missing_detector_error():
    # Create a control but primary detector name not present in channels
    n = 200
    p = np.random.normal(1000, 20, size=n)
    events = np.vstack([p, 0.1 * p]).T
    s = SimpleFake(events, ['X', 'Y'])
    controls = {'F': s}
    # primary detector 'Z' not in channels
    try:
        _ = generate_compensation_from_controls(controls, {'F': 'Z'}, min_events=10)
        raise AssertionError('expected ValueError')
    except ValueError:
        pass
