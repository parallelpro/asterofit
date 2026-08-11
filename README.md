# asterofit

An asteroseismic grid modelling analysis toolkit. `asterofit` estimates
stellar parameters (age, mass, radius, ...) by comparing a grid of stellar
evolutionary tracks against observed classical constraints (Teff, [Fe/H],
luminosity, ...) and/or seismic constraints (individual oscillation mode
frequencies, Dnu), for one or more stars at once.

## How it works

The core `grid` class drives a three-stage pipeline, each stage optionally
split across worker processes/threads:

1. **scan_tracks** -- score every model in every track against every star's
   observations (classical chi2, seismic mode matching and Dnu, optional
   surface correction); keep only the models that pass loose classical/seismic
   cuts.
2. **process_star_results** -- collapse the surviving per-track models into
   flat per-star arrays and compute the final seismic and combined chi2.
3. **output_results** -- write corner plots, summary tables, echelle
   diagrams, and/or an HDF5 dump per star under `filepath_output`.

After `run()` returns, `self.star_results` holds one `StarResults` object per
star with every estimated parameter, ready for further analysis.

## Installation

Requires Python 3 with `pip`.

```bash
git clone https://github.com/parallelpro/asterofit.git
cd asterofit
pip install -e .
```

This installs `asterofit` in editable mode along with its dependencies
(`numpy`, `scipy`, `pandas`, `matplotlib`). To also run the test suite:

```bash
pip install -e ".[test]"
pytest asterofit/tests
```

`demo.py` additionally requires `astropy`, which is not installed by default.

## Usage

```python
from asterofit import grid, user_setup_params

# read_models: a function that loads one evolutionary track file into a table
# tracks: a list of filepaths to evolutionary track files
# user_setup_params: a dict of configuration (input data paths, columns,
#   which constraints to use, output options, ...) -- see asterofit/params.py
# for the full set of options

g = grid(read_models, tracks, user_setup_params)
g.run()
```

See `demo.py` and `demo.ipynb` for a complete worked example, and
`asterofit/params.py` for all configurable options.

## Under construction
- Priors on age and mass
- Implement regularized chi2
- Implement reduced chi2
- Improve frequency matching
- Documentation

## Future roadmap
- Improve the sampling method
- Fit with multiple stars (binaries/clusters)
- Mixed mode surface correction
- Add customizable priors (useful for distance, IMF, etc.)
- Support for extinction

## License

MIT License, see [LICENSE](LICENSE).
