'''
Script to process all the K2 cluster data with Appaloosa

Run on WWU workstation (iMac)
Run like this:
$ ipython ~/python/appaloosa/misc/runk2clust.py


run all the data on the workstation with this, but do actual analysis/plots later

'''

import numpy as np
from appaloosa.appaloosa import appaloosa as ap
from os.path import expanduser


# home = expanduser("~")
# dir = home + '/research/k2_cluster_flares/'

# do both types of LC's
lctype = ['Everest', 'Vanderburg']
dbmode = ['everest', 'vdb']

# datadir = dir + 'k2clusters.ipac.caltech.edu/'
# clusters = ['hyades','pleiades']

for i in range(len(lctype)):
    dir = '/Volumes/CoveyData/K2_Clusters/' + lctype[i]

    lis = dir + '/' + 'all.lis'

    print('>> running ' + lis)

    files = np.loadtxt(lis, dtype='str', unpack=True, usecols=(0,))

    for f in files:
        print(dir + f[1:])
        ap.RunLC(file= dir + f[1:], dbmode=dbmode[i], verbosefake=True, nfake=1000, maxgap=2,
                 debug=False, display=False)
