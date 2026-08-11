from dataclasses import dataclass

import numpy as np

from asterofit.src.output_results import summarize_star_results


@dataclass
class FakeConfig:
    estimators_to_summary: list
    estimators_to_plot: list
    if_classical: bool = True
    if_seismic: bool = False


def make_star_results(n, extra=None):
    rng = np.random.default_rng(0)
    d = {
        'star_mass': rng.normal(1.0, 0.05, n),
        'radius': rng.normal(1.0, 0.05, n),
        'Teff': rng.normal(5700, 50, n),
        'luminosity': rng.normal(1.0, 0.1, n),
        'FeH': rng.normal(0, 0.1, n),
        'chi2_classical': rng.chisquare(3, n),
        'chi2': rng.chisquare(3, n),
    }
    if extra:
        d.update(extra)
    return d


def test_too_few_models_status():
    config = FakeConfig(estimators_to_summary=['star_mass', 'radius'], estimators_to_plot=['star_mass', 'radius'])
    result = summarize_star_results(make_star_results(4), config)
    assert result == {'status': 'too_few_models'}


def test_too_few_samples_status():
    # 6 surviving models (> 5, so it clears the 'too_few_models' floor) but 6
    # estimators-to-summary columns -> samples.shape[0] <= samples.shape[1]
    columns = ['star_mass', 'radius', 'Teff', 'luminosity', 'FeH', 'chi2_classical']
    config = FakeConfig(estimators_to_summary=columns, estimators_to_plot=['star_mass'])
    result = summarize_star_results(make_star_results(6), config)
    assert result['status'] == 'too_few_samples'
    assert 'chi2' in result
    assert 'quantile_prob' not in result


def test_ok_status_has_quantiles_and_best_models():
    config = FakeConfig(estimators_to_summary=['star_mass', 'radius'], estimators_to_plot=['star_mass'])
    result = summarize_star_results(make_star_results(20), config)
    assert result['status'] == 'ok'
    assert result['quantile_prob'].shape == (3, 2)  # (16/50/84 percentile) x (2 estimators)
    assert result['quantile_prob_classical'].shape == (3, 2)
    # best_models_table: rows = ['chi2', 'chi2_classical'], cols = [ranked_by, chi2, star_mass, radius]
    assert result['best_models_table'].shape == (2, 4)


def test_best_model_selection_picks_max_probability():
    config = FakeConfig(estimators_to_summary=['star_mass'], estimators_to_plot=['star_mass'])
    star_results = make_star_results(20)
    star_results['chi2'][7] = np.min(star_results['chi2']) - 1.0  # force model 7 to be the clear best
    result = summarize_star_results(star_results, config)
    assert result['best_models_indices'][0] == 7
