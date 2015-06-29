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


def GetLC(objectid, type=''):

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


def OneCadence(data):
    '''
    Within each quarter of data from the database, pick the data with the
    fastest cadence. We want to study 1-min if available. Don't want
    multiple cadence observations in the same quarter, bad for detrending.

    Parameters
    ----------
    data : numpy array
        the result from MySQL database query, using the getLC() function

    Returns
    -------
    Data array

    '''
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

        indx = np.append(indx, x[0][y])

    indx = np.array(indx, dtype='int')

    data_out = data[indx,:]
    return data_out


def DetectCand(time, flux, error, model, error_cut=3, gapwindow = 0.1, nptsmin=2):
    '''
    detect flare candidates
    '''

    chi = (flux - model) / error

    # find points above sigma threshold
    cand1 = np.where((chi >= error_cut))

    # find consecutive points above threshold
    # cand2 = np.where((cand1[0][1:]-cand1[0][:-1] < 2))


    _, dl, dr = detrend.FindGaps(time) # find edges of time windows
    for i in range(0, len(dl)):
        x1 = np.where((np.abs(time[cand1]-time[dr[i]-1]) < gapwindow))
        x2 = np.where((np.abs(time[cand1]-time[dl[i]]) < gapwindow))
        cand1 = np.delete(cand1, x1)
        cand1 = np.delete(cand1, x2)

    # for now just return indx of candidates
    return cand1


def FlagCuts(flags):

    '''
    return the indexes that pass flag cuts

    Ethan says cut on 16, 128, 2048. Can add more later.
    '''

    flags_int = np.array(flags, dtype='int')
    bad_flgs = [16, 128, 2048]

    bad = np.zeros_like(flags)

    for k in bad_flgs:
        bad = bad + np.bitwise_and(flags_int, k)

    good = np.where((bad < 1))[0]
    return good


# objectid = '9726699'  # GJ 1243
def RunLC(objectid='9726699', ftype='sap', display=True):
    '''
    Main wrapper to obtain and process a light curve
    '''

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
    data_raw = GetLC(objectid)
    data = OneCadence(data_raw)

    # data columns are:
    # QUARTER, TIME, PDCFLUX, PDCFLUX_ERR, SAP_QUALITY, LCFLAG, SAPFLUX, SAPFLUX_ERR

    qtr = data[:,0]
    time = data[:,1]

    if ftype == 'sap':
        flux_raw = data[:,6]
        error = data[:,7]
    else:
        flux_raw = data[:,2]
        error = data[:,3]

    _,lg,rg = detrend.FindGaps(time)
    uQtr = np.unique(qtr)

    flux_qtr = detrend.QtrFlat(time, flux_raw, qtr)

    flux_gap = detrend.GapFlat(time, flux_qtr)

    flux_sin = detrend.FitSin(time, flux_gap, error)

    flux_smo = detrend.MultiBoxcar(time, flux_gap - flux_sin, error)

    flux_model = flux_sin + flux_smo

    cand = DetectCand(time, flux_gap, error, flux_model)

    if display is True:
        plt.figure()
        plt.plot(time, flux_gap, 'k')
        plt.plot(time, flux_model, 'green')
        for g in lg:
            plt.scatter(time[g], flux_gap[g], color='blue', marker='v',s=40)

        plt.scatter(time[cand], flux_gap[cand], color='red', marker='o',s=40)
        plt.show()

