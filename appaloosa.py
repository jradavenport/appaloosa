"""
script to carry out flare finding in Kepler LC's

"""

import numpy as np
import MySQLdb
import sys



def aflare(t, p):
    """
    This is the Analytic Flare Model from the flare-morphology paper
    Please reference Davenport (2014) http://arxiv.org/abs/1411.3723

    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    p : 1-d array
        p == [tpeak, fwhm (units of time), amplitude (units of flux)] x N

    Returns
    -------
    flare : 1-d array
        The fluxes of the flare model
    """
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    Nflare = np.floor( (len(p)/3.0) )

    flare = np.zeros_like(t)
    # compute the flare model for each flare
    for i in range(Nflare):
        outm = np.piecewise(t, [(t<= p[0+i*3]) & (t-p[0+i*3])/p[1+i*3] > -1.,
                                (t > p[0+i*3])],
                            [lambda x: (_fr[0]+                             # 0th order
                                        _fr[1]*((t-p[0+i*3])/p[1+i*3])+     # 1st order
                                        _fr[2]*((t-p[0+i*3])/p[1+i*3])^2.+  # 2nd order
                                        _fr[3]*((t-p[0+i*3])/p[1+i*3])^3.+  # 3rd order
                                        _fr[4]*((t-p[0+i*3])/p[1+i*3])^4. ),# 4th order
                             lambda x: (_fd[0]*np.exp( ((t-p[0+i*3])/p[1+i*3])*_fd[1] ) +
                                        _fd[2]*np.exp( ((t-p[0+i*3])/p[1+i*3])*_fd[3] ))]
                            ) * p[2+i*3] # amplitude
        flare = flare + outm

    return flare


# this holds the keys to the db... don't put in github.
# but, this is a horrible  clearly not super-duper encrypted either
auth = np.loadtxt('auth.txt',dtype='string')

# connect to the db
db = MySQLdb.connect(passwd=auth[2], db="Kepler",
                     user=auth[1], host=auth[0])


# read the objectID from the CONDOR job...
# objectid = sys.argv[1]
objectid = '9726699'  # test LC

# make a cursor
cur = db.cursor()

cur.execute('SELECT QUARTER, TIME, SAP_FLUX, SAP_FLUX_ERR,SAP_QUALITY, LCFLAG FROM Kepler.source WHERE KEPLERID='+objectid+';')

# get all the data
rows = cur.fetchall()

# convert to numpy data array
data = np.asarray(rows)

# close the cursor to the db
cur.close()


# now on to the smoothing, flare finding, flare fitting, and results!
