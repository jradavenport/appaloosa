"""
script to carry out flare finding in Kepler LC's

"""

import numpy as np
import MySQLdb
import sys
import time
from aflare import aflare
import detrend
import matplotlib.pyplot as plt
# from detrend import polysmooth,


def _getLC(objectid, type=''):

    # this holds the keys to the db... don't put on github!
    auth = np.loadtxt('auth.txt',dtype='string')

    isok = 0 # a flag to check if database returned sensible answer
    ntry = 0

    while isok<1:
        # connect to the db
        db = MySQLdb.connect(passwd=auth[2], db="Kepler",
                             user=auth[1], host=auth[0])

        query = 'SELECT QUARTER, TIME, SAP_FLUX, SAP_FLUX_ERR, SAP_QUALITY, LCFLAG ' + \
                'FROM Kepler.source WHERE KEPLERID=' + str(objectid)

        # only get SLC or LLC data if requested
        if type=='slc':
            query = query + ' AND LCFLAG=0 '
        if type=='llc':
            query = query + ' AND LCFLAG=1 '


        query = query + ' ORDER BY TIME;'

        # make a cursor to the db
        cur = db.cursor()
        cur.execute(query)

        # get all the data
        rows = cur.fetchall()

        # convert to numpy data array
        data = np.asarray(rows)

        # close the cursor to the db
        cur.close()

        # make sure the database returned the actual light curve
        if len(data[:,0] > 1000):
            isok = 10
        # only try 10 times... shouldn't ever need this limit
        if ntry > 9:
            isok = 2
        ntry = ntry + 1
        time.sleep(10) # give the database a breather

    return data


def runLC():
    # read the objectID from the CONDOR job...
    # objectid = sys.argv[1]

    objectid = '9726699'  # GJ 1243
    # objectid = '8226697' # a random star

    # get the data from the MYSQL db
    data = _getLC(objectid)
    # data columns are:
    # QUARTER, TIME, SAP_FLUX, SAP_FLUX_ERR, SAP_QUALITY, LCFLAG

    time = data[1,:]
    flux_raw = data[2,:]
    qtr = data[0,:]
    flux_qtr = detrend.QtrFlat(time, flux_raw, qtr)

    flux_sin = detrend.FitSin(time, flux_qtr)
#
#     # now on to the smoothing, flare finding, flare fitting, and results!
#     smo = detrend.rolling_poly(data[1,:], flux_q, data[3,:], data[0,:])
#
# ediff = (data[1,:] - smo) / data[2,:] # simple error weighted outlier finding

