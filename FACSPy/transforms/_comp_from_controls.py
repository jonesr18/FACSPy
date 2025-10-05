"""Generate compensation matrices from single-color control samples.

Implements a simple median-ratio method to compute spill coefficients.
This module is intentionally dependency-light and uses numpy/pandas and the
existing Matrix class.
"""
from __future__ import annotations

import time
from typing import Dict, Iterable, Optional
import numpy as np

from ._matrix import Matrix


def compute_spill_coefficients(control_events: np.ndarray,
                               primary_idx: int,
                               background_events: Optional[np.ndarray] = None,
                               method: str = 'median_ratio',
                               positive_threshold: Optional[float] = None,
                               eps: float = 1e-9) -> np.ndarray:
    """Compute per-channel spill coefficients for a single-color control.

    Parameters
    ----------
    control_events
        Array shape (n_events, n_channels)
    primary_idx
        Index of the primary (stained) detector in the array
    background_events
        Optional background events to use for median subtraction
    method
        Currently only 'median_ratio' is implemented
    positive_threshold
        If provided, only events with primary > threshold are used to avoid dividing by noise
    eps
        Small epsilon to avoid division by zero

    Returns
    -------
    coef: ndarray
        1D array of length n_channels containing coefficients where coef[primary_idx] == 1.0
    """
    if control_events.ndim != 2:
        raise ValueError('control_events must be 2D (n_events, n_channels)')

    n_events, n_channels = control_events.shape
    if n_events == 0:
        raise ValueError('control_events contains no events')

    primary = control_events[:, primary_idx].astype(np.float64)

    if background_events is not None:
        if background_events.shape[1] != n_channels:
            raise ValueError('background_events must have same number of channels as control_events')
        bg = np.median(background_events, axis=0)
    else:
        bg = np.zeros(n_channels, dtype=np.float64)

    # subtract background medians
    numerator = control_events - bg
    denom = primary - bg[primary_idx]

    # choose events used for ratio calculation
    if positive_threshold is None:
        # require denom > small epsilon
        mask = denom > eps
    else:
        mask = denom > positive_threshold

    if np.count_nonzero(mask) == 0:
        raise ValueError('No positive events found for primary detector after thresholding')

    coefs = np.ones(n_channels, dtype=np.float64)
    if method == 'median_ratio':
        # for each channel j, coef_j = median( (signal_j - bg_j) / (signal_primary - bg_primary) )
        ratios = (numerator[mask, :] / denom[mask, None])
        # robustify: use median
        med = np.median(ratios, axis=0)
        coefs = med
        # ensure primary channel coefficient is exactly 1
        coefs[primary_idx] = 1.0
    else:
        raise NotImplementedError(f"Method '{method}' not implemented")

    return coefs


def generate_compensation_from_controls(controls: Dict[str, object],
                                        primary_detector_map: Dict[str, str],
                                        *,
                                        method: str = 'median_ratio',
                                        background: Optional[object] = None,
                                        min_events: int = 100,
                                        subsample: Optional[int] = None,
                                        positive_threshold: Optional[float] = None) -> Matrix:
    """Generate a compensation Matrix from single-color control samples.

    Parameters
    ----------
    controls
        Mapping fluor_name -> control sample. A control sample may be:
        - an object with attributes `.original_events` (ndarray) and `.channels` (pd.DataFrame index)
        - a numpy ndarray (interpreted as events)
    primary_detector_map
        Mapping fluor_name -> detector label (string) indicating the primary detector for that control
    method
        Algorithm to compute coefficients (currently only 'median_ratio')
    background
        Optional unstained control in same formats as controls; used to compute per-channel background medians
    min_events
        Minimum events required per control
    subsample
        If provided, randomly select up to this many events per control for computation
    positive_threshold
        threshold applied to the primary channel to select positive events

    Returns
    -------
    Matrix
        Compensation matrix object
    """
    # Normalize controls into (events, channel_labels)
    parsed_controls = {}
    for fluor, sample in controls.items():
        if isinstance(sample, np.ndarray):
            events = sample
            channels = [f'ch{i}' for i in range(events.shape[1])]
        else:
            # assume sample has original_events and channels
            events = getattr(sample, 'original_events', None)
            channels = None
            if events is None:
                raise ValueError(f'Control {fluor} does not expose original_events ndarray')
            # try to read channel labels
            ch_df = getattr(sample, 'channels', None)
            if ch_df is not None:
                try:
                    channels = list(ch_df.index)
                except Exception:
                    channels = None

        if subsample is not None and events.shape[0] > subsample:
            rng = np.random.default_rng(0)
            idx = rng.choice(events.shape[0], size=subsample, replace=False)
            events = events[idx, :]

        if events.shape[0] < min_events:
            raise ValueError(f'Control {fluor} has only {events.shape[0]} events (min_events={min_events})')

        parsed_controls[fluor] = (events, channels)

    # Ensure a consistent detector ordering across controls
    first_channels = None
    for fluor, (_, chs) in parsed_controls.items():
        if chs is not None:
            first_channels = chs
            break

    if first_channels is None:
        # fallback to numeric channel names from first control
        sample0 = next(iter(parsed_controls.values()))
        n_chan = sample0[0].shape[1]
        first_channels = [f'ch{i}' for i in range(n_chan)]

    # verify all controls share same channel labels if provided
    for fluor, (_, chs) in parsed_controls.items():
        if chs is not None and chs != first_channels:
            raise ValueError(f'Control {fluor} has channel labels {chs} which do not match {first_channels}')

    detectors = first_channels
    n = len(detectors)

    # prepare optional background median
    bg_events = None
    if background is not None:
        if isinstance(background, np.ndarray):
            bg_events = background
        else:
            bg_events = getattr(background, 'original_events', None)
            if bg_events is None:
                raise ValueError('background provided but does not expose original_events ndarray')

    # build matrix
    matrix = np.eye(n, dtype=np.float64)

    # track provenance
    provenance = {'controls': {}, 'method': method}

    for fluor, detector in primary_detector_map.items():
        if fluor not in parsed_controls:
            raise ValueError(f'Primary fluor {fluor} not found in controls')
        events, _ = parsed_controls[fluor]
        # find detector index
        try:
            primary_idx = detectors.index(detector)
        except ValueError:
            raise ValueError(f"Primary detector '{detector}' for fluor '{fluor}' not found in detected channels: {detectors}")

        coefs = compute_spill_coefficients(events,
                                          primary_idx=primary_idx,
                                          background_events=bg_events,
                                          method=method,
                                          positive_threshold=positive_threshold)

        # set into matrix: row = target detector, column = source? For FACSPy Matrix, convention is
        # matrix applied as flowutils.compensate.compensate(events, matrix, indices)
        # where matrix is spill matrix with shape (n,n) where element [i,j] is contribution of j into i.
        # Our coefs represent ratio of each channel to the primary, so we set column=primary_idx to coefs
        matrix[:, primary_idx] = coefs

        provenance['controls'][fluor] = {'detector': detector, 'n_events': events.shape[0]}

    matrix_id = f"generated_from_controls:{method}:{int(time.time())}"
    m = Matrix(matrix_id=matrix_id, spill_data_or_file=matrix, detectors=detectors)
    # attach provenance
    try:
        m.generated_provenance = provenance
    except Exception:
        # best-effort; not critical
        pass

    return m
