import numpy as np 

def concat(list_of_small_stardata):
    '''
    Construct a stardata container from existing containers:
    >>> Ntrack = 2
    >>> keys = ['Teff', 'radius', 'lum', 'delta_nu', '[M/H]']
    >>> star0 = stardata(Ntrack, keys)
    >>> star1 = stardata(Ntrack, keys)
    >>> star0[:, 'Teff'] = [np.linspace(1000,1200, 5) for i in range(2)]
    >>> star0[:, 'lum'] = [np.linspace(1,12, 5) for i in range(2)]
    >>> star1[:, 'Teff'] = [np.linspace(1000,1200, 5) for i in range(2)]
    >>> star1[:, 'lum'] = [np.linspace(1,12, 5) for i in range(2)]
    >>> star = concat([star0, star1])
    '''

    keys = list_of_small_stardata[0].keys 
    NKeys = list_of_small_stardata[0].NKeys
    NStardata = len(list_of_small_stardata)
    big_stardata = stardata(1, keys)
    for key in keys:
        big_stardata[key] = np.concatenate([ s[key] for s in list_of_small_stardata])
    return big_stardata


class stardata():
    def __init__(self, Ntrack, keys):
        '''

        This is a container for one dimensional star data.
        Each element must be a numpy array.
        
        Initialize a stardata container to hold 1000 tracks and 
        2 stellar properties:
        Ntrack = 1000
        keys = ['Teff', 'radius']
        >>> star = stardata(Ntrack, keys)

        Store values:
        >>> star['Teff', 0] = np.linspace(5300, 5400, 100)
        >>> star['radius', 1] = np.linspace(1, 10, 100)
        >>> star['lum', :] = [np.linspace(1, 10, 100) for iTrack in range(Ntrack)]

        Attributes:
        - NKeys  # number of keys
        - keys  # list of keys
        - Ntrack  # number of tracks
        - cidx[key]  # a (Ntrack,) boolean array indicating whether data have been written

        Methods:
        - remove_none: remove the track entries with at least one empty key
        - collapse: concatenate all tracks for each variable. remove the notion of "Ntrack"

        '''

        self.keys = np.array(keys)
        self.Ntrack = Ntrack 
        self.NKeys = len(keys)
        # self.cidx = {key: np.zeros(self.Ntrack, dtype=bool) for key in keys}

        # initialize with an empty array
        empty = np.empty((Ntrack), dtype=object)
        empty.fill(np.array([]))
        for key in keys:
            setattr(self, key, np.copy(empty))


    def __getitem__(self, indices):
        if len(indices)==2:
            key, iTrack = indices
            return getattr(self, key)[iTrack]
        elif type(indices) in [str, np.str_]:
            key = indices
            return getattr(self, key)
        else:
            raise IndexError('stardata object does not support more than 2 indices.')

    def __setitem__(self, indices, value):
        if len(indices)==2:
            key, iTrack = indices
            if not hasattr(self, key): raise ValueError('stardata object does not have a key named {:s}.'.format(key))
            # self.cidx[key][iTrack] = True
            getattr(self, key)[iTrack] = value
            
        elif type(indices) in [str, np.str_]:
            key = indices
            if not hasattr(self, key): raise ValueError('stardata object does not have a key named {:s}.'.format(key))
            # self.cidx[key][:] = True
            setattr(self, key, value)

        else:
            raise IndexError('stardata object only supports stardata[key, itrack] indexing syntax.')

    # def remove_none(self, copy=True):
    #     '''

    #         Only keep the non-empty entries.

    #     '''

    #     idx = np.all(np.array([np.array(self.cidx[key], dtype=bool) for key in self.keys]), axis=0)
    #     # initialize with an empty array

    #     if copy:
    #         Ntrack = np.sum(idx)
    #         newself = stardata(Ntrack, self.keys)
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

            Concatenate all tracks for each variable. Remove the notion of "Ntrack".

        '''
        if copy:
            newself = stardata(1, self.keys)
            for key in self.keys:
                newself[key, :] = np.concatenate(getattr(self, key), axis=0)
            newself.Ntrack = 0 
            # newself.cidx = True
            return newself
        else:
            for key in self.keys:
                # print(key, getattr(self, key))
                setattr(self, key, np.concatenate(getattr(self, key), axis=0))
            self.Ntrack = 0 
            # self.cidx = True
            return self

        

if __name__ == "__main__":
    # test 1 - construct keys and assign values

    print('--- Test 1 ---')
    Ntrack = 1000
    keys = ['Teff', 'radius', 'lum', 'delta_nu', '[M/H]']
    t1 = stardata(Ntrack, keys)

    print('star.NKeys: ', t1.NKeys)
    print('star.keys: ', t1.keys)
    print('star.Ntrack: ', t1.Ntrack)

    print('star["Teff"][0]: ', t1['Teff'][0])
    print('star["Teff"][1]: ', t1['Teff'][1])
    t1['Teff', 1] = np.arange(5000,5010,1)
    print('star["Teff", 1]: ', t1['Teff', 1])
    print('star["Teff"][1]: ', t1['Teff'][1])

    t1.collapse(copy=False) 
    print('Collapse')
    print('star.Ntrack: ', t1.Ntrack)
    print('star["Teff"]: ', t1['Teff'])
    t1['lum'] = np.linspace(1,10,10)
    print('star["lum"]: ', t1['lum'])



    # # # test 2 - concatenate stardata
    # print('--- Test 2 ---')
    # Ntrack = 2
    # keys = ['Teff',  'lum', 'radius']
    # star0 = stardata(Ntrack, keys)
    # star1 = stardata(Ntrack, keys)
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
