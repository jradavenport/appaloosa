'''
Script(s) to prep jobs for processing all light curves with CONDOR
'''

import numpy as np
import os
from os.path import expanduser
import time


def HexTime():
    '''
    Generate a unique string of HEX related to the timestamp. Useful for generating runs

    Examples
    --------
    >>> HexTime()
        '0x56538a65'

    '''
    t = int(time.time())
    return str(hex(t))


def PrepUW(prefix=''):
    '''
    Generate the Condor config file needed to script the running of Appaloosa,
    and the little helper shell script. Intuition gathered from working on UW
    Astro's Condor network with STSP.

    Parameters
    ----------
    prefix : str, optional
        What prefix to call this run. By default a unique string of HEX code is used,
        based on the timestamp.

    Returns
    -------
    Generates files in the working directory for running Condor.

    '''
    if (prefix==''):
        prefix = HexTime()

    # where is the list of KeplerObjectID's stored for this machine?
    # dir = '/astro/users/jrad/Dropbox/research_projects/nsf_flare_code/'
    home = expanduser("~")
    dir = home + '/Dropbox/research_projects/nsf_flare_code/'
    file = 'get_objects.out'

    kid = np.loadtxt(dir + file, dtype='str',
                     unpack=True, skiprows=1, usecols=(0,))

    # this is UW-internal specific also
    workdir = '/astro/store/scratch/tmp/jrad/nsf_flares/' + prefix + '/'

    if not os.path.isdir(workdir):
        try:
            os.makedirs(workdir)
        except OSError:
            pass

    # put the database authentication file in the working dir
    os.system('cp ' + dir + 'auth.txt ' + workdir + '.')

    condor_file = workdir + prefix + '.cfg'
    shellscript = workdir + prefix + '.sh'
    pyversion = home + "/anaconda/bin/python"

    # the path to the actual science code
    python_code = home + '/python/appaloosa/appaloosa.py'

    # 2222222222222222 create CONDOR .cfg file
    f2 = open(condor_file, 'w')
    f2.write('Notification = never \n')
    f2.write('Executable = '+ shellscript +' \n')
    f2.write('Initialdir = ' + workdir + '\n')
    f2.write('Universe = vanilla \n')
    f2.write(' \n')
    f2.write('Log = ' + workdir + prefix + '_log.txt \n')
    f2.write('Error = ' + workdir + prefix + '_err.txt \n')
    f2.write('Output = ' + workdir + prefix + '_out.txt \n')
    f2.write(' \n')

    for k in kid:
        # put entry in to CONDOR .cfg file for this window
        f2.write('Arguments = ' + k + ' \n')
        f2.write('Queue \n')

    f2.write(' \n')
    f2.close()

    # 33333333333333333
    # create the very simple PYTHON-launching shell script
    f3 = open(shellscript,'w')
    f3.write("#!/bin/bash \n")
    f3.write(pyversion + " " + python_code + " $1 \n")
    f3.close()

    # fix permissions
    os.system("chmod 777 " + shellscript)
    os.system("chmod 777 " + condor_file)

    print('')
    print('UW Condor prep is complete.')
    print('To launch: on the Condor head machine, do this')
    print('$ condor_submit ' + condor_file)
    print('')

    return


def PrepWWU(prefix=''):
    '''
    Generate the Condor config file needed to script the running of Appaloosa,
    and the little helper shell script. Built for running on the WWU CS Compute Cluster.

    Parameters
    ----------
    prefix : str, optional
        What prefix to call this run. By default a unique string of HEX code is used,
        based on the timestamp.

    Returns
    -------
    Generates files in the working directory for running Condor.

    '''
    if (prefix==''):
        prefix = HexTime()

    # where is the list of KeplerObjectID's stored for this machine?
    home = expanduser("~")
    dir = home + '/data/kepler/'

    file = 'all_fits.lis' # created manually, path to every FITS file
    # because so many files, had to create list like this:
    # $ find Q*_public/ -type f -name '*.fits' > all_fits.lis

    kid = np.loadtxt(dir + file, dtype='str',
                     unpack=True, usecols=(0,))

    # Put all the run outputs in this data directory as well
    workdir = home + '/data/' + prefix + '/'
    if not os.path.isdir(workdir):
        try:
            os.makedirs(workdir)
        except OSError:
            pass

    condor_file = workdir + prefix + '.cfg'
    shellscript = workdir + prefix + '.sh'

    pyversion = home + "/anaconda2/bin/python"

    # the path to the actual science code
    python_code = home + '/python/appaloosa/appaloosa.py'

    # 2222222222222222 create CONDOR .cfg file
    f2 = open(condor_file, 'w')
    f2.write('Notification = never \n')
    f2.write('Executable = '+ shellscript +' \n')
    f2.write('Initialdir = ' + workdir + '\n')
    f2.write('Universe = vanilla \n')
    f2.write(' \n')
    f2.write('Log = ' + workdir + prefix + '_log.txt \n')
    f2.write('Error = ' + workdir + prefix + '_err.txt \n')
    f2.write('Output = ' + workdir + prefix + '_out.txt \n')
    f2.write(' \n')

    for k in kid:
        # put entry in to CONDOR .cfg file for this window
        f2.write('Arguments = ' + k + ' \n')
        f2.write('Queue \n')

    f2.write(' \n')
    f2.close()

    # 33333333333333333
    # create the very simple PYTHON-launching shell script
    f3 = open(shellscript,'w')
    f3.write("#!/bin/bash \n")
    f3.write(pyversion + " " + python_code + " $1 \n")
    f3.close()

    # fix permissions
    os.system("chmod 777 " + shellscript)
    os.system("chmod 777 " + condor_file)

    print('')
    print('WWU Condor prep is complete.')
    print('To launch: on the CS Compute Cluser, do this:')
    print('$ condor_submit ' + condor_file)
    print('')

    return


# let this file be called from the terminal directly. e.g.:
# $python conda.py
if __name__ == "__main__":
    # for now just bulid the UW condor prep in
    PrepWWU()
    # PrepWWU()
