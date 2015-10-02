"""
script to carry out flare finding in Kepler LC's

"""

import numpy as np
import os.path
import time
import datetime
from aflare import aflare1
import detrend
from version import __version__
import warnings
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
try:
    import MySQLdb
    haz_mysql = True
except ImportError:
    haz_mysql = False


def _chisq(data, error, model):
    '''
    Compute the normalized chi square statistic
    '''
    return np.sum( ((data - model) / error)**2.0 ) / data.size


def GetLC(objectid, type='', readfile=False, savefile=False, onecadence=False):

    exten = '.lc.gz'

    isok = 0 # a flag to check if database returned sensible answer
    ntry = 0

    if readfile is True:
        # attempt to find file in working dir
        if os.path.isfile(str(objectid) + exten):
            data = np.loadtxt(str(objectid) + exten)
            isok = 101

    # this holds the keys to the db... don't put on github!
    auth = np.loadtxt('auth.txt', dtype='string')

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

    if onecadence is True:
        data_raw = data.copy()
        data = OneCadence(data_raw)

    if savefile is True:
        # output a file in working directory
        np.savetxt(str(objectid) + exten, data)

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


def DetectCandidate(time, flux, error, flags, model,
                    error_cut=2, gapwindow=0.1, minsep=3,
                    returnall=False):
    '''
    detect flare candidates using sigma threshold, toss out bad flags

    Parameters
    ----------
    time :
    flux :
    error :
    flags :
    model :
    error_cut : int, optional
        the sigma threshold to select outliers (default is 2)
    gapwindow : float, optional
        The duration of time around data gaps to ignore flare candidates
        (default is 0.1 days)
    minsep : int, optional
        The number of datapoints required between individual flare events
        (default is 3)

    Returns
    -------
    (flare start index, flare stop index)

    if returnall=True, returns
    (flare start index, flare stop index, candidate indicies)

    '''

    bad = FlagCuts(flags, returngood=False)

    chi = (flux - model) / error

    # find points above sigma threshold, and passing flag cuts
    cand1 = np.where((chi >= error_cut) & (bad < 1))[0]

    _, dl, dr = detrend.FindGaps(time) # find edges of time windows
    for i in range(0, len(dl)):
        x1 = np.where((np.abs(time[cand1]-time[dr[i]-1]) < gapwindow))
        x2 = np.where((np.abs(time[cand1]-time[dl[i]]) < gapwindow))
        cand1 = np.delete(cand1, x1)
        cand1 = np.delete(cand1, x2)

    # find start and stop index, combine neighboring candidates in to same events
    cstart = cand1[np.append([0], np.where((cand1[1:]-cand1[:-1] > minsep))[0]+1)]
    cstop = cand1[np.append(np.where((cand1[1:]-cand1[:-1] > minsep))[0], [len(cand1)-1])]

    # for now just return index of candidates
    if returnall is True:
        return cstart, cstop, cand1
    else:
        return cstart, cstop

'''
# start working on a rolling cumulative distribution function, where
# flares are selected as positive outliers from rolling windows
def RollingCDF(time, flux, minwindow=0.06, maxwindow=1, numwindows=10):
    windows = 10.**np.logspace(np.log10(minwindow),
                               np.log10(maxwindow), numwindows)

    # now step thru the light curve...
'''


def FlagCuts(flags, bad_flags = (16, 128, 2048), returngood=True):

    '''
    return the indexes that pass flag cuts

    Ethan says cut out [16, 128, 2048], can add more later.
    '''

    # convert flag array to type int, just in case
    flags_int = np.array(flags, dtype='int')
    # empty array to hold places where bad flags exist
    bad = np.zeros_like(flags)

    # these are the specific flags to reject on
    # NOTE: == 2**[4, 7, 11]
    # bad_flgs = [16, 128, 2048]

    # step thru each of the bitwise flags, find where exist
    for k in bad_flags:
        bad = bad + np.bitwise_and(flags_int, k)

    # find places in array where NO bad flags are set
    if returngood is True:
        good = np.where((bad < 1))[0]
        return good
    else:
        return bad


def EquivDur(time, flux):
    '''
    Compute the Equivalent Duration of an event. This is simply the area
    under the flare, in relative flux units.

    Flux must be array in units of zero-centered RELATIVE FLUX

    Time must be array in units of DAYS

    Output has units of SECONDS
    '''
    p = np.trapz(flux, x=(time * 60.0 * 60.0 * 24.0))
    return p


def FlareStats(time, flux, error, model, istart=-1, istop=-1,
               c1=(-1,-1), c2=(-1,-1), cpoly=2, ReturnHeader=False):
    '''
    Compute properties of a flare event. Assumes flux is in relative flux units,
    i.e. rel_flux = (flux - median) / median

    Parameters
    ----------
    time : 1d numpy array
    flux : 1d numpy array
    error : 1d numpy array
    model : 1d numpy array
        These 4 arrays must have the same number of elements
    istart : int, optional
        The index in the input arrays (time,flux,error,model) that the
        flare starts at. If not used, defaults to the first data point.
    istop : int, optional
        The index in the input arrays (time,flux,error,model) that the
        flare ends at. If not used, defaults to the last data point.

    '''

    # if FLARE indicies are not stated by user, use start/stop of data
    if (istart < 0):
        istart = 0
    if (istop < 0):
        istop = len(flux)

    # can't have flare start/stop at same point
    if (istart == istop):
        istop = istop + 1
        istart = istart - 1

    # need to have flare at least 3 datapoints long
    if (istop-istart < 2):
        istop = istop + 1

    # print(istart, istop) # % ;

    tstart = time[istart]
    tstop = time[istop]
    dur0 = tstop - tstart

    # define continuum regions around the flare, same duration as
    # the flare, but spaced by half a duration on either side
    if (c1[0]==-1):
        t0 = time[istart] - dur0
        t1 = time[istart] - dur0/2.
        c1 = np.where((time >= t0) & (time <= t1))
    if (c2[0]==-1):
        t0 = time[istop] + dur0/2.
        t1 = time[istop] + dur0
        c2 = np.where((time >= t0) & (time <= t1))

    flareflux = flux[istart:istop+1]
    flaretime = time[istart:istop+1]
    modelflux = model[istart:istop+1]
    flareerror = error[istart:istop+1]

    contindx = np.concatenate((c1[0], c2[0]))
    if (len(contindx) == 0):
        # if NO continuum regions are found, then just use 1st/last point of flare
        contindx = np.array([istart, istop])
        cpoly = 1
    contflux = flux[contindx] # flux IN cont. regions
    conttime = time[contindx]
    contfit = np.polyfit(conttime, contflux, cpoly)
    contline = np.polyval(contfit, flaretime) # poly fit to cont. regions

    medflux = np.median(model)

    # measure flare amplitude
    ampl = np.max(flareflux-contline) / medflux
    tpeak = flaretime[np.argmax(flareflux-contline)]

    p05 = np.where((flareflux-contline <= ampl*0.5))
    if len(p05[0]) == 0:
        fwhm = dur0 * 0.25
        # print('> warning') # % ;
    else:
        fwhm = np.max(flaretime[p05]) - np.min(flaretime[p05])

    # fit flare with single aflare model
    pguess = (tpeak, fwhm, ampl)
    # print(pguess) # % ;
    # print(len(flaretime)) # % ;

    try:
        popt1, pcov = curve_fit(aflare1, flaretime, (flareflux-contline) / medflux, p0=pguess)
    except ValueError:
        # tried to fit bad data, so just fill in with NaN's
        # shouldn't happen often
        popt1 = np.array([np.nan, np.nan, np.nan])
    except RuntimeError:
        # could not converge on a fit with aflare
        # fill with bad flag values
        popt1 = np.array([-99., -99., -99.])

    # flare_chisq = total( flareflux - modelflux)**2.  / total(error)**2
    flare_chisq = _chisq(flareflux, flareerror, modelflux)

    # measure KS stats of flare versus model
    ks_d, ks_p = stats.ks_2samp(flareflux, modelflux)

    # measure KS stats of flare versus continuum regions
    ks_dc, ks_pc = stats.ks_2samp(flareflux-contline, contflux-np.polyval(contfit, conttime))

    # put flux in relative units, remove dependence on brightness of stars
    # rel_flux = (flux_gap - flux_model) / np.median(flux_model)
    # rel_error = error / np.median(flux_model)

    # measure flare ED
    ed = EquivDur(flaretime, (flareflux-contline)/medflux)

    # output a dict or array?
    params = np.array((tstart, tstop, tpeak, ampl, fwhm, dur0,
                       popt1[0], popt1[1], popt1[2],
                       flare_chisq, ks_d, ks_p, ks_dc, ks_pc, ed), dtype='float')
    # the parameter names for later reference
    header = 't_start, t_stop, t_peak, amplitude, FWHM, duration, '+\
             't_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1, '+\
             'flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur'

    if ReturnHeader is True:
        return header
    else:
        return params


def MeasureS2N(flux, error, model, istart=-1, istop=-1):
    '''
    this MAY not be something i want....
    '''
    if (istart < 0):
        istart = 0
    if (istop < 0):
        istop = len(flux)

    flareflux = flux[istart:istop+1]
    modelflux = model[istart:istop+1]

    s2n = np.sum(np.sqrt((flareflux) / (flareflux + modelflux)))
    return s2n


# objectid = '9726699'  # GJ 1243
def RunLC(objectid='9726699', ftype='sap', lctype='', display=False, readfile=False):
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
    data_raw = GetLC(objectid, readfile=readfile, type=lctype)
    data = OneCadence(data_raw)

    # data columns are:
    # QUARTER, TIME, PDCFLUX, PDCFLUX_ERR, SAP_QUALITY, LCFLAG, SAPFLUX, SAPFLUX_ERR

    qtr = data[:,0]
    time = data[:,1]
    lcflag = data[:,5]

    if ftype == 'sap':
        flux_raw = data[:,6]
        error = data[:,7]
    else: # for PDC data
        flux_raw = data[:,2]
        error = data[:,3]

    _,lg,rg = detrend.FindGaps(time)
    uQtr = np.unique(qtr)

    # flatten quarters with polymonial
    flux_qtr = detrend.QtrFlat(time, flux_raw, qtr)

    # then flatten between gaps
    flux_gap = detrend.GapFlat(time, flux_qtr)

    # fit sin curves
    flux_sin = detrend.FitSin(time, flux_gap, error)

    # run iterative boxcar over data
    flux_smo = detrend.MultiBoxcar(time, flux_gap - flux_sin, error)

    flux_model = flux_sin + flux_smo
    # flux_model = flux_sin + detrend.MultiBoxcar(time, flux_smo, error) # double detrend?

    istart,istop = DetectCandidate(time, flux_gap, error, lcflag, flux_model)
    print(str(len(istart))+' flare candidates found')

    if display is True:
        plt.figure()
        plt.plot(time, flux_gap, 'k')
        plt.plot(time, flux_model, 'green')
        for g in lg:
            plt.scatter(time[g], flux_gap[g], color='blue', marker='v',s=40)

        plt.scatter(time[istart], flux_gap[istart], color='red', marker='o',s=40)
        plt.scatter(time[istop], flux_gap[istop], color='orange', marker='p',s=60)

        plt.show()

    '''
    #-- IF YOU WANT TO PLAY WITH THE WAVELET STUFF MORE, WORK HERE
    test_model = detrend.WaveletSmooth(time, flux_gap)
    test_cand = DetectCandidate(time, flux_gap, error, test_model)

    print(len(cand))
    print(len(test_cand))

    if display is True:
        plt.figure()
        plt.plot(time, flux_gap, 'k')
        plt.plot(time, test_model, 'green')

        plt.scatter(time[test_cand], flux_gap[test_cand], color='orange', marker='p',s=60, alpha=0.8)
        plt.show()
    '''

    # set this to silence bad fit warnings from polyfit
    warnings.simplefilter('ignore', np.RankWarning)


    outstring = ''
    outstring = outstring + '# Kepler-ObjectID = ' + objectid + '\n'
    now = datetime.datetime.now()
    outstring = outstring + '# Date-Run = ' + str(now) + '\n'
    outstring = outstring + '# Appaloosa-Version = ' + __version__ + '\n'

    outstring = outstring + '# Columns: '

    header = FlareStats(time, flux_gap, error, flux_model,
                        istart=istart[0], istop=istop[0],
                        ReturnHeader=True)
    outstring = outstring + '# ' + header + '\n'
    # loop over EACH FLARE, compute stats
    for i in range(0,len(istart)):
        stats_i = FlareStats(time, flux_gap, error, flux_model,
                             istart=istart[i], istop=istop[i])

        outstring = outstring + str(stats_i[0])
        for k in range(1,len(stats_i)):
            outstring = outstring + ', ' + str(stats_i[k])
        outstring = outstring + '\n'

    # put flare output in to a subdirectory. keep things clean!
    outdir = 'aprun/'
    if not os.path.isdir(outdir):
        try:
            os.makedirs(outdir)
        except OSError:
            pass

    # open the output file to store data on every flare recovered
    fout = open(outdir + objectid + '.flare', 'w')
    fout.write(outstring)
    fout.close()

# let this file be called from the terminal directly. e.g.:
# $python appaloosa.py 12345678
if __name__ == "__main__":
    import sys
    RunLC(objectid=str(sys.argv[1]), display=False)

