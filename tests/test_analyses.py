"""Regression coverage for the refactored execution boundaries."""

from pathlib import Path

import numpy as np
import pandas as pd

from layered_dielectrics import config, phantoms


def test_sensitivity_preserves_baseline(monkeypatch, tmp_path):
    from layered_dielectrics.sensitivity import run
    monkeypatch.setattr(config, 'OUTPUT_TABLES_DIR', tmp_path / 'tables')
    # The legacy angular grid includes the singular grazing angle; its NaNs
    # are handled by the established nanmean/nanmax aggregation.
    with np.errstate(divide='ignore', invalid='ignore'):
        actual = run(phantoms.PHA10_18G())
    expected = pd.read_json(Path(__file__).parent / 'fixtures/sensitivity_PHA10_18G.json')
    assert actual['Contributor'].tolist() == expected['Contributor'].tolist()
    np.testing.assert_allclose(actual[['TE', 'TM', 'Unc. |dB']],
                               expected[['TE', 'TM', 'Unc. |dB']], atol=1e-10, rtol=1e-8)
    assert (tmp_path / 'tables/PHA10-18G.xlsx').is_file()


def test_summary_returns_results_and_saves_default_formats(monkeypatch, tmp_path):
    from layered_dielectrics.broadband_phantom_summary import run
    monkeypatch.setattr(config, 'OUTPUT_PLOTS_DIR', tmp_path)
    result = run(phantoms.PHA24_30G_V2())
    assert result['reflection_te'].shape == (25, 200)
    assert np.isfinite(result['reflection_te']).all()
    np.testing.assert_allclose(np.abs(result['reflection_te'][:, 0]),
                               np.abs(result['reflection_tm'][:, 0]))
    assert list(tmp_path.glob('*.svg')) and list(tmp_path.glob('*.pdf'))
