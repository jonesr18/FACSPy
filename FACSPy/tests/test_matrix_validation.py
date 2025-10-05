import numpy as np
import pytest

from FACSPy.transforms._matrix import Matrix


class FakeSample:
    """Minimal sample-like object to test Matrix.apply without full I/O."""
    def __init__(self, channels, events):
        # channels: list of channel labels (index)
        # events: numpy array shape (n_events, n_channels)
        import pandas as pd
        self.channels = pd.DataFrame(index=channels)
        # create a 'channel_numbers' column with 1-based indices expected by get_channel_index
        self.channels['channel_numbers'] = list(range(1, len(channels) + 1))
        self.original_events = events
        self.compensated_events = None
        self.compensation_status = 'uncompensated'

    def get_events(self, source='raw'):
        if source == 'raw':
            return self.original_events
        elif source == 'comp':
            return self.compensated_events
        raise ValueError('unknown source')

    def get_channel_index(self, channel_label):
        # emulate FCSFile.get_channel_index behaviour
        return int(self.channels.loc[self.channels.index == channel_label, 'channel_numbers'].iloc[0]) - 1


def test_apply_success_identity():
    channels = ['A', 'B']
    events = np.array([[100., 200.], [10., 20.]])
    sample = FakeSample(channels, events)

    mat = np.eye(2)
    m = Matrix(matrix_id='test_identity', spill_data_or_file=mat, detectors=channels)

    compensated = m.apply(sample)
    # identity matrix should not change values
    assert np.allclose(compensated, events)


def test_apply_missing_detector_raises():
    channels = ['A', 'B']
    events = np.array([[1., 2.]])
    sample = FakeSample(channels, events)

    mat = np.eye(2)
    # detectors list includes a missing channel 'C'
    m = Matrix(matrix_id='test_missing', spill_data_or_file=mat, detectors=['A', 'C'])

    with pytest.raises(ValueError) as exc:
        _ = m.apply(sample)
    assert 'requires detectors not present in the sample' in str(exc.value)


def test_apply_shape_mismatch_raises():
    channels = ['A', 'B', 'C']
    events = np.zeros((5, 3))
    sample = FakeSample(channels, events)

    # 2x2 matrix but 3 detectors resolved -> should raise
    mat = np.eye(2)
    m = Matrix(matrix_id='test_shape', spill_data_or_file=mat, detectors=channels)

    with pytest.raises(ValueError) as exc:
        _ = m.apply(sample)
    assert 'has shape' in str(exc.value) and 'detector indices' in str(exc.value)
