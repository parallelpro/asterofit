import numpy as np
import pytest

from asterofit.src.star_results import StarResults, concat


def test_two_character_key_does_not_break_indexing():
    # regression test: __getitem__/__setitem__ used to check len(indices)==2
    # before checking whether indices was a string, so a 2-character key like
    # 'Zi' got misparsed as a (key, iTrack) tuple ('Z', 'i').
    s = StarResults(3, ['Teff', 'Zi', 'FeH'])
    s['Zi', 1] = np.arange(5)
    assert np.array_equal(s['Zi'][1], np.arange(5))
    assert np.array_equal(s['Zi', 1], np.arange(5))


def test_key_named_like_bookkeeping_attribute_does_not_clobber_it():
    # regression test: storage used to be setattr(self, key, ...), so a data
    # column named 'keys'/'Ntrack'/'NKeys' would silently overwrite the
    # container's own bookkeeping attributes of the same name.
    s = StarResults(2, ['Teff', 'keys', 'Ntrack'])
    s['keys', 0] = np.array([1, 2, 3])
    s['Ntrack', 0] = np.array([4, 5])

    assert np.array_equal(s['keys'][0], np.array([1, 2, 3]))
    assert np.array_equal(s['Ntrack'][0], np.array([4, 5]))
    # the container's own bookkeeping must be untouched
    assert s.Ntrack == 2
    assert list(s.keys) == ['Teff', 'keys', 'Ntrack']


def test_collapse_concatenates_across_tracks():
    s = StarResults(3, ['Teff'])
    s['Teff', 0] = np.array([1., 2.])
    s['Teff', 1] = np.array([3.])
    s['Teff', 2] = np.array([4., 5., 6.])
    s.collapse(copy=False)
    assert np.array_equal(s['Teff'], np.array([1., 2., 3., 4., 5., 6.]))
    assert s.Ntrack == 0


def test_concat_merges_multiple_star_results_objects():
    # concat() merges at the track level (pre-collapse) -- this is how main.py
    # combines per-thread results before process_star_results() later collapses
    # away the track dimension. So the merged container should have Ntrack =
    # sum of the inputs' Ntrack, still ragged per-track, until collapsed.
    keys = ['Teff', 'lum']
    star0 = StarResults(2, keys)
    star0['Teff', :] = [np.array([1., 2.]), np.array([3., 4.])]
    star0['lum', :] = [np.array([10., 20.]), np.array([30., 40.])]

    star1 = StarResults(2, keys)
    star1['Teff', :] = [np.array([5.]), np.array([6., 7.])]
    star1['lum', :] = [np.array([50.]), np.array([60., 70.])]

    merged = concat([star0, star1])
    assert len(merged['Teff']) == 4  # 2 tracks from star0 + 2 from star1, not yet collapsed

    merged.collapse(copy=False)
    assert np.array_equal(merged['Teff'], np.array([1., 2., 3., 4., 5., 6., 7.]))
    assert np.array_equal(merged['lum'], np.array([10., 20., 30., 40., 50., 60., 70.]))
