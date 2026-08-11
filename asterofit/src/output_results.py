import numpy as np

try:
    from .tools import quantile
except ImportError:
    from tools import quantile

__all__ = ['summarize_star_results']


def summarize_star_results(star_results, config) -> dict:
    '''
    Compute the sample matrix, best-model rankings, and weighted-quantile summaries
    for one star's finished StarResults -- everything output_results needs to know
    before it writes plots/files. Pure function: no file I/O, no plotting, no
    dependence on if_plot (that's an output-writing decision, not a summarization one).

    ----------
    Input:
    star_results: StarResults (or any object supporting the same string-keyed
    indexing, e.g. a plain dict)
        One star's collapsed, chi2-populated results -- i.e.
        `grid.star_results[istar]` after `process_star_results()` has run.
        Must have every key in `config.estimators_to_summary`/
        `estimators_to_plot`, plus 'chi2' (and 'chi2_classical'/
        'chi2_seismic' if the corresponding config flag is set).
    config: ScanConfig
        `estimators_to_summary`/`estimators_to_plot` select which columns go
        into `samples`/`samples_to_plot`; `if_classical`/`if_seismic` select
        which extra chi2/prob/quantile fields get computed.

    ----------
    Output:
    result: dict with a 'status' key:
      'too_few_models'  -- fewer than 5 models were selected; nothing else is computed.
      'too_few_samples' -- enough models, but not enough to weight-quantile
                            ('samples' has more columns than rows); only the
                            chi2/prob/best-model fields below are present.
      'ok'              -- everything is present, including 'quantile_prob'/
                            'best_models_table'.

    Common fields (status in {'too_few_samples', 'ok'}):
      'samples', 'samples_to_plot', 'chi2', 'prob',
      'chi2_classical'/'prob_classical' (if config.if_classical),
      'chi2_seismic'/'prob_seismic' (if config.if_seismic),
      'best_models_ranked_by', 'best_models_indices', 'best_models_chi2s'

    Additional fields when status == 'ok':
      'quantile_prob', 'quantile_prob_classical' (if config.if_classical),
      'quantile_prob_seismic' (if config.if_seismic), 'best_models_table'
    '''
    samples = np.transpose(np.array([star_results[param_name] for param_name in config.estimators_to_summary]))
    samples_to_plot = np.transpose(np.array([star_results[param_name] for param_name in config.estimators_to_plot]))

    chi2 = star_results['chi2']
    if len(chi2) <= 5:
        return {'status': 'too_few_models'}

    prob = np.exp(-chi2/2.)
    best_models_ranked_by = ['chi2']
    best_models_indices = [np.nanargmax(prob)]
    best_models_chi2s = [chi2[np.nanargmax(prob)]]

    result = dict(samples=samples, samples_to_plot=samples_to_plot, chi2=chi2, prob=prob)

    if config.if_classical:
        chi2_classical = np.array(star_results['chi2_classical'], dtype=float)
        prob_classical = np.exp(-(chi2_classical)/2.)
        best_models_ranked_by.append('chi2_classical')
        best_models_indices.append(np.nanargmax(prob_classical))
        best_models_chi2s.append(chi2_classical[np.nanargmax(prob_classical)])
        result['chi2_classical'] = chi2_classical
        result['prob_classical'] = prob_classical

    if config.if_seismic:
        chi2_seismic = np.array(star_results['chi2_seismic'], dtype=float)
        prob_seismic = np.exp(-(chi2_seismic)/2.)
        best_models_ranked_by.append('chi2_seismic')
        best_models_indices.append(np.nanargmax(prob_seismic))
        best_models_chi2s.append(chi2_seismic[np.nanargmax(prob_seismic)])
        result['chi2_seismic'] = chi2_seismic
        result['prob_seismic'] = prob_seismic

    result['best_models_ranked_by'] = best_models_ranked_by
    result['best_models_indices'] = best_models_indices
    result['best_models_chi2s'] = best_models_chi2s

    if samples.shape[0] <= samples.shape[1]:
        result['status'] = 'too_few_samples'
        return result

    if config.if_classical:
        result['quantile_prob_classical'] = quantile(samples, (0.16, 0.5, 0.84), weights=prob_classical)
    if config.if_seismic:
        result['quantile_prob_seismic'] = quantile(samples, (0.16, 0.5, 0.84), weights=prob_seismic)
    result['quantile_prob'] = quantile(samples, (0.16, 0.5, 0.84), weights=prob)

    Nchi2 = len(best_models_chi2s)
    result['best_models_table'] = np.concatenate([
        np.array(best_models_ranked_by).reshape(Nchi2, 1),
        np.array(best_models_chi2s).reshape(Nchi2, 1),
        samples[best_models_indices, :],
    ], axis=1)

    result['status'] = 'ok'
    return result


if __name__ == "__main__":
    # a tiny, self-contained smoke test -- no grid instance, no file I/O.
    from dataclasses import dataclass, field

    @dataclass
    class FakeConfig:
        estimators_to_summary: list
        estimators_to_plot: list
        if_classical: bool = True
        if_seismic: bool = False

    rng = np.random.default_rng(0)
    N = 20
    star_results = {
        'star_mass': rng.normal(1.0, 0.05, N),
        'radius': rng.normal(1.0, 0.05, N),
        'chi2_classical': rng.chisquare(3, N),
        'chi2': rng.chisquare(3, N),
    }
    config = FakeConfig(estimators_to_summary=['star_mass', 'radius'], estimators_to_plot=['star_mass', 'radius'])

    result = summarize_star_results(star_results, config)
    print('status:', result['status'])
    print('best_models_ranked_by:', result['best_models_ranked_by'])
    print('quantile_prob shape:', result['quantile_prob'].shape)
