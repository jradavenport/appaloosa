"""
script to carry out flare finding in Kepler LC's

"""

import numpy as np
import MySQLdb
import sys
from aflare import aflare
import detrend
# from detrend import polysmooth,


def _getLC(objectid):

    # this holds the keys to the db... don't put on github!
    auth = np.loadtxt('auth.txt',dtype='string')

    # connect to the db
    db = MySQLdb.connect(passwd=auth[2], db="Kepler",
                         user=auth[1], host=auth[0])

    query = 'SELECT QUARTER, TIME, SAP_FLUX, SAP_FLUX_ERR, SAP_QUALITY, LCFLAG ' + \
            'FROM Kepler.source '+\
            'WHERE KEPLERID='+str(objectid)+\
            'ORDER BY TIME;'

    # make a cursor to the db
    cur = db.cursor()
    cur.execute(query)

    # get all the data
    rows = cur.fetchall()

    # convert to numpy data array
    data = np.asarray(rows)

    # close the cursor to the db
    cur.close()

    return data



# read the objectID from the CONDOR job...
# objectid = sys.argv[1]

# objectid = '9726699'  # GJ 1243

objectid = '8226697' # a random star

# get the data from the MYSQL db
data = _getLC(objectid)
# data columns are:
# QUARTER, TIME, SAP_FLUX, SAP_FLUX_ERR, SAP_QUALITY, LCFLAG

if (len(data[0,:]) > 100):
    flux_q = detrend.QtrFlat(data[1,:], data[2,:], data[0,:])

    # now on to the smoothing, flare finding, flare fitting, and results!
    smo = detrend.polysmooth(data[1,:], flux_q, data[3,:], data[0,:])

ediff = (data[1,:] - smo) / data[2,:] # simple error weighted outlier finding

