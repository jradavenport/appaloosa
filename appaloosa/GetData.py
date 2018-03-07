import os
import numpy as np
import pandas as pd


'''
A function to GET data from the appaloosa directory mess... 

Only meant to be run from MY computer @ WWU

Warning: is obscenely slow
'''


def CopyData(kic, outdir='./'):
    datadir = '/Users/davenpj3/research/kepler-flares/'
    fakelis = '0x56ff1094_fakes.lis'
    home = os.getenv("HOME")

    fakes = pd.read_table(datadir + fakelis, names=['file'], delim_whitespace=True, usecols=(0,))

    # find the files
    star = np.where(fakes['file'].str.contains(str(kic)))[0]

    # BRUTE FORCE
    # for each fake file, copy fake & flare (if it exists) to new directory
    for k in range(len(star)):
        os.system('cp ' + datadir + fakes['file'].values[star][k][0:-5] + '* ' + outdir)


def GetStars(kics, outdir='./'):
    for k in range(len(kics)):
        CopyData(kics[k], outdir=outdir)


if __name__ == "__main__":
    import sys
    file = pd.read_csv(sys.argv[1], names=['id', 'kics'])
    GetStars(file['kics'].values, outdir='data/')