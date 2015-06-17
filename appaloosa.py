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


def getLC(objectid, type=''):

    # this holds the keys to the db... don't put on github!
    auth = np.loadtxt('auth.txt',dtype='string')

    isok = 0 # a flag to check if database returned sensible answer
    ntry = 0

    while isok<1:
        # connect to the db
        db = MySQLdb.connect(passwd=auth[2], db="Kepler",
                             user=auth[1], host=auth[0])

        query = 'SELECT QUARTER, TIME, PDCSAP_FLUX, PDCSAP_FLUX_ERR, ' +\
                'SAP_QUALITY, LCFLAG, SAP_FLUX, SAP_FLUX_ERR ' +\
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
        if len(data[:,0] > 99):
            isok = 10
        # only try 10 times... shouldn't ever need this limit
        if ntry > 9:
            isok = 2
        ntry = ntry + 1
        time.sleep(10) # give the database a breather

    return data


def onecadence(data):
    # get the unique quarters
    qtr = data[:,0]
    cadence = data[:,5]
    uQtr = np.unique(qtr)

    indx = []

    # for each quarter, is there more than one cadence?
    for q in uQtr:
        x = np.where( (np.abs(qtr-q) < 0.1) )

        etimes = np.unique(cadence[x])
        y = np.where( (cadence[x] == min(etimes)) )

        indx = indx.append(x[0][y])

    data_out = data[indx,:]
    return data_out


# objectid = '9726699'  # GJ 1243
def runLC(objectid='9726699', ftype='sap'):
    # read the objectID from the CONDOR job...
    # objectid = sys.argv[1]

    # pick and process a totally random LC.
    # important for reality checking!
    if (objectid is 'random'):
        obj, num = np.loadtxt('get_objects.out', skiprows=1, unpack=True, dtype='str')
        rand_id = int(np.random.random() * len(obj))
        objectid = obj[rand_id]
        print('Random ObjectID Selected: ' + objectid)

    # get the data from the MYSQL db
    data_raw = getLC(objectid)
    data = onecadence(data_raw)

    # data columns are:
    # QUARTER, TIME, FLUX, FLUX_ERR, SAP_QUALITY, LCFLAG
    qtr = data[:,0]
    time = data[:,1]
    flux_raw = data[:,2]
    error = data[:,3]

    flux_qtr = detrend.QtrFlat(time, flux_raw, qtr)

    flux_sin = detrend.FitSin(time, flux_qtr, error)

    plt.figure()
    plt.plot(time, flux_raw, 'b')
    plt.plot(time, flux_qtr, 'g')
    plt.plot(time, flux_sin, 'r')
    plt.show()

#     # now on to the smoothing, flare finding, flare fitting, and results!
#     smo = detrend.rolling_poly(data[1,:], flux_q, data[3,:], data[0,:])
#
# ediff = (data[1,:] - smo) / data[2,:] # simple error weighted outlier finding
