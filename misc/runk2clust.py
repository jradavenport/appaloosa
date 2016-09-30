'''
Script to process all the K2 cluster data with Appaloosa

Run on WWU workstation (iMac)
Run like this:
$ ipython ~/python/appaloosa/runk2clus.py


run all the data on the workstation with this, but do actual analysis/plots later

'''

import numpy as np
import appaloosa.appaloosa as ap
from os.path import expanduser


# home = expanduser("~")
# dir = home + '/research/k2_cluster_flares/'

# do both types of LC's
lctype = ['Vanderburg', 'Everest']
dbmode = ['vdb', 'everest']

# datadir = dir + 'k2clusters.ipac.caltech.edu/'
# clusters = ['hyades','pleiades']

for i in range(len(lctype)):
    dir = '/Volumes/CoveyData/K2_Clusters/' + lctype[i]

    lis = dir + '/' + 'all.lis'

    files = np.loadtxt(lis, dtype='str')

    for f in files:
        print(dir + f)
        ap.RunLC(file= dir + f[1:], dbmode=dbmode[i], debug=False, display=False, verbosefake=True, nfake=1000)
