'''
Use this script to take a bunch of aprun/*.flare files,
 and put them in to aprun/XXX/*.flare

Shouldn't be needed with future runs of appaloosa.

NOTE: run above aprun dir, not within it!
'''

import os
import numpy as np


# make list of existing aprun/*.flare files
fname =  'flarefiles_to_sort.lis'
os.system('ls aprun/*.flare > ' + fname)

files = np.loadtxt(fname, dtype='str',)

# get objectid list, truncate to 3 digits
objid3 = map(lambda x: x[6:9], files)

# uniq the 3 digit list

u_id = np.unique(objid3)

# make sub-dirs
for f in u_id:
    if not os.path.isdir('aprun/' + f):
        try:
            os.makedirs('aprun/' + f)
        except OSError:
            pass
    # move files in to sub-dirs
    os.system('mv aprun/' + f + '*.flare  aprun/' + f + '/.')
