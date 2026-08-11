import numpy as np

def concat(list_of_small_star_results):
    '''
    Merge several StarResults (e.g. one per multiprocessing worker, each
    holding a different subset of tracks for the same star) into one, by
    concatenating each key's per-track columns together end to end -- the
    result still has one entry per track (from every input, back to back),
    it is not `collapse()`'d. Used by `grid.run()` to reassemble
    `scan_tracks()`'s per-worker results before `process_star_results()`.

    All inputs must share the same `keys` (as constructed with).

    >>> Ntrack = 2
    >>> keys = ['Teff', 'radius', 'lum', 'delta_nu', '[M/H]']
    >>> star0 = StarResults(Ntrack, keys)
    >>> star1 = StarResults(Ntrack, keys)
    >>> star0['Teff', :] = [np.linspace(1000,1200, 5) for i in range(2)]
    >>> star0['lum', :] = [np.linspace(1,12, 5) for i in range(2)]
    >>> star1['Teff', :] = [np.linspace(1000,1200, 5) for i in range(2)]
    >>> star1['lum', :] = [np.linspace(1,12, 5) for i in range(2)]
    >>> star = concat([star0, star1])

    ----------
    Input:
    list_of_small_star_results: list[StarResults]
        Several StarResults, all constructed with the same `keys`.

    ----------
    Output:
    merged_star_results: StarResults
        `Ntrack` = 1 (per `StarResults.__init__`'s own bookkeeping -- the
        actual per-key data is the concatenation of every input's Ntrack
        entries, still one ragged sub-array per original track).

    '''

    keys = list_of_small_star_results[0].keys
    NKeys = list_of_small_star_results[0].NKeys
    n_star_results = len(list_of_small_star_results)
    merged_star_results = StarResults(1, keys)
    for key in keys:
        merged_star_results[key] = np.concatenate([ small_star_results[key] for small_star_results in list_of_small_star_results])
    return merged_star_results


class StarResults():
    def __init__(self, Ntrack, keys):
        '''

        A container for one star's per-track scan results: each `key` (a
        column name like 'Teff' or 'chi2_seismic_l0') maps to a length-Ntrack
        object array, one ragged sub-array of surviving models per track.
        Every element must be a numpy array. This is the ragged, "one entry
        per track" stage; `collapse()` flattens it into a single array once
        all tracks have been scanned.

        Initialize a StarResults container to hold 1000 tracks and
        2 stellar properties:
        Ntrack = 1000
        keys = ['Teff', 'radius']
        >>> star = StarResults(Ntrack, keys)

        Store values:
        >>> star['Teff', 0] = np.linspace(5300, 5400, 100)
        >>> star['radius', 1] = np.linspace(1, 10, 100)
        >>> star['lum', :] = [np.linspace(1, 10, 100) for itrack in range(Ntrack)]

        ----------
        Input:
        Ntrack: int
            Number of tracks this instance will hold one entry per (until
            `collapse()`'d, at which point `Ntrack` becomes 0 -- see below).
        keys: array-like[str]
            The column names this instance will store; fixed at construction
            -- `__setitem__`/`__getitem__` raise `ValueError` for any other key.

        ----------
        Sets:
        self.keys: array-like[str]
        self.NKeys: int
            `keys` as given, and its length.
        self.Ntrack: int
            `Ntrack` as given (mutated to 0 by `collapse()`).
        self._data: dict[str, array-like[Ntrack] of object]
            Internal storage backing `__getitem__`/`__setitem__` -- not part
            of the public interface, access data through indexing instead.

        ----------
        Methods:
        collapse: concatenate every track's sub-array together per key,
            removing the notion of "Ntrack" (see its own docstring).

        '''

        self.keys = np.array(keys)
        self.Ntrack = Ntrack
        self.NKeys = len(keys)

        # initialize with an empty array
        empty = np.empty((Ntrack), dtype=object)
        empty.fill(np.array([]))
        self._data = {key: np.copy(empty) for key in keys}


    def __getitem__(self, indices):
        '''
        Two forms: `star[key]` returns the whole column (length Ntrack before
        `collapse()`, a flat array after); `star[key, itrack]` returns just
        that track's sub-array.

        ----------
        Input:
        indices: str, or (str, int or slice)
            A key name, or a (key, itrack) pair.

        ----------
        Output:
        array-like
            `self._data[key]` (whole column) or `self._data[key][itrack]`
            (one track's sub-array), depending on which form was used.

        '''
        # check string-ness first: a 2-character key (e.g. 'Zi') also has len()==2,
        # and would otherwise get misparsed as a (key, itrack) tuple below.
        if isinstance(indices, (str, np.str_)):
            key = indices
            return self._data[key]
        elif len(indices)==2:
            key, itrack = indices
            return self._data[key][itrack]
        else:
            raise IndexError('StarResults object does not support more than 2 indices.')

    def __setitem__(self, indices, value):
        '''
        The `__getitem__` forms, as assignment targets: `star[key] = value`
        replaces the whole column (used by `collapse()`, and to write the
        final chi2 columns after collapsing); `star[key, itrack] = value`
        sets just that track's sub-array (used while scanning, once per
        track). `key` must already exist (from the `keys` given at
        construction) -- this container's schema is fixed, not extensible.

        ----------
        Input:
        indices: str, or (str, int or slice)
            A key name, or a (key, itrack) pair.
        value: array-like
            The whole column, or one track's sub-array, matching `indices`.

        ----------
        Output: none (mutates self._data in place).

        '''
        if isinstance(indices, (str, np.str_)):
            key = indices
            if key not in self._data: raise ValueError('StarResults object does not have a key named {:s}.'.format(key))
            self._data[key] = value

        elif len(indices)==2:
            key, itrack = indices
            if key not in self._data: raise ValueError('StarResults object does not have a key named {:s}.'.format(key))
            self._data[key][itrack] = value

        else:
            raise IndexError('StarResults object only supports StarResults[key, itrack] indexing syntax.')

    # def remove_none(self, copy=True):
    #     '''

    #         Only keep the non-empty entries.

    #     '''

    #     idx = np.all(np.array([np.array(self.cidx[key], dtype=bool) for key in self.keys]), axis=0)
    #     # initialize with an empty array

    #     if copy:
    #         Ntrack = np.sum(idx)
    #         newself = StarResults(Ntrack, self.keys)
    #         for key in self.keys:
    #             newself[key, :] = getattr(self, key)[idx]
    #         return newself
    #     else:
    #         self.Ntrack = np.sum(idx)
    #         for key in self.keys:
    #             setattr(self, key, getattr(self, key)[idx])
    #         return self

    def collapse(self, copy=True):
        '''
        Concatenate every track's per-key sub-array together into one flat
        array per key, discarding the notion of which track each model came
        from -- this is what turns `scan_tracks()`'s per-track, ragged
        `StarResults` into the flat form `process_star_results()`/
        `output_results()` actually read (e.g. `star['chi2']` becomes a
        single 1D array of every surviving model across every track, instead
        of one ragged sub-array per track).

        ----------
        Optional input:
        copy: bool, default True
            If True, leave `self` untouched and return a new, already-
            collapsed `StarResults` (used by `concat()`, via this default).
            If False, collapse `self` in place and return it (used by
            `process_star_results()`, since the per-star `StarResults`
            objects already live on `grid.star_results` and don't need a copy).

        ----------
        Output:
        StarResults, with `Ntrack == 0` and every key's column flattened to a
        single 1D array (2D for the per-mode 'diff_freq'/'mod_freq'-style
        keys) -- `self` if `copy=False`, otherwise a new instance.

        '''
        if copy:
            collapsed_star_results = StarResults(1, self.keys)
            for key in self.keys:
                collapsed_star_results[key, :] = np.concatenate(self._data[key], axis=0)
            collapsed_star_results.Ntrack = 0
            return collapsed_star_results
        else:
            for key in self.keys:
                self._data[key] = np.concatenate(self._data[key], axis=0)
            self.Ntrack = 0
            return self



if __name__ == "__main__":
    # test 1 - construct keys and assign values

    print('--- Test 1 ---')
    Ntrack = 1000
    keys = ['Teff', 'radius', 'lum', 'delta_nu', '[M/H]']
    test_star = StarResults(Ntrack, keys)

    print('star.NKeys: ', test_star.NKeys)
    print('star.keys: ', test_star.keys)
    print('star.Ntrack: ', test_star.Ntrack)

    print('star["Teff"][0]: ', test_star['Teff'][0])
    print('star["Teff"][1]: ', test_star['Teff'][1])
    test_star['Teff', 1] = np.arange(5000,5010,1)
    print('star["Teff", 1]: ', test_star['Teff', 1])
    print('star["Teff"][1]: ', test_star['Teff'][1])

    test_star.collapse(copy=False)
    print('Collapse')
    print('star.Ntrack: ', test_star.Ntrack)
    print('star["Teff"]: ', test_star['Teff'])
    test_star['lum'] = np.linspace(1,10,10)
    print('star["lum"]: ', test_star['lum'])



    # # # test 2 - concatenate star results
    # print('--- Test 2 ---')
    # Ntrack = 2
    # keys = ['Teff',  'lum', 'radius']
    # star0 = StarResults(Ntrack, keys)
    # star1 = StarResults(Ntrack, keys)
    # star0['Teff', :] = [np.linspace(1000, 1200, 5) for i in range(Ntrack)]
    # star0['lum', :] = [np.linspace(1, 12, 5) for i in range(Ntrack)]
    # star0['radius', 0] = np.linspace(0.1, 0.12, 5)

    # print(star0['Teff'])
    # print(star0['lum'])
    # print(star0['radius'])

    # star1['Teff', :] = [np.linspace(2000, 2200, 5) for i in range(Ntrack)]
    # star1['lum', :] = [np.linspace(21, 22, 5) for i in range(Ntrack)]
    # star1['radius', 0]  = np.linspace(0.1, 0.12, 5)
    # star = concat([star0, star1])
    # print(star['Teff'])
    # print(star['lum'])
    # print(star['radius'])
