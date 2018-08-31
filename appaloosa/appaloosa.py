"""
script to carry out flare finding in Kepler LC's

"""

import numpy as np
import os.path

import time as clock
import time
import datetime
from version import __version__
from aflare import aflare1
from helper import EquivDur
import detrend
from gatspy.periodic import LombScargleFast
import warnings
import matplotlib.pyplot as plt
import pandas as pd

from scipy import signal

import matplotlib
import glob
import json
from fake import FlareStats, ed6890, chisq, FakeFlaresDist, FakeCompleteness
from get import Get, GetLCdb

matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams.update({'font.family':'sansserif'})
from scipy.signal import savgol_filter

# from rayleigh import RayleighPowerSpectrum
try:
    import MySQLdb
    haz_mysql = True
except ImportError:
    haz_mysql = False

def ED(start,stop,time,flux_model,flux, error):

    '''
    Returns the equivalend duratio of a flare event,
    found within indices [start, stop],
    calculated as the area under the residual (flux-flux_model)
    Returns also the error on ED following (Davenport 2016)

    Parameters:
    --------------
    start - start time index of a flare event
    stop - end time index of a flare event
    time - time array
    flux_model - model quiescent flux
    flux - long-term trend removed raw light curve
    error - rolling std error to raw flux

    Returns:
    --------------
    ed - equivalent duration in seconds
    ederr - uncertainty in seconds
    '''

    start, stop = int(start),int(stop)
    time = np.asarray(time)[start:stop+1]
    model = np.asarray(flux_model)[start:stop+1]
    flux = np.asarray(flux)[start:stop+1]
    error = np.asarray(error)[start:stop+1]
    residual = (flux - model)
    ed = np.trapz(residual,time*60.*60.*24.)
    #measure error on ED

    flare_chisq = chisq(flux, error, model)
    ederr = np.sqrt(ed**2 / (stop-start) / flare_chisq)
    return ed, ederr, flare_chisq






#def DetectCandidate(time, flux, error, flags, model,
#                    error_cut=2, gapwindow=0.1, minsep=3,
#                    returnall=False):
#    '''
#    Detect flare candidates using sigma threshold, toss out bad flags.
#    Uses very simple sigma cut to find significant points.

#    Parameters
#    ----------
#    time :
#    flux :
#    error :
#    flags :
#    model :
#    error_cut : int, optional
#        the sigma threshold to select outliers (default is 2)
#    gapwindow : float, optional
#        The duration of time around data gaps to ignore flare candidates
#        (default is 0.1 days)
#    minsep : int, optional
#        The number of datapoints required between individual flare events
#        (default is 3)

#    Returns
#    -------
#    (flare start index, flare stop index)

#    if returnall=True, returns
#    (flare start index, flare stop index, candidate indicies)

#    '''

#    bad = FlagCuts(flags, returngood=False)

#    chi = (flux - model) / error

#    # find points above sigma threshold, and passing flag cuts
#    cand1 = np.where((chi >= error_cut) & (bad < 1))[0]

#    _, dl, dr = detrend.FindGaps(time) # find edges of time windows
#    for i in range(0, len(dl)):
#        x1 = np.where((np.abs(time[cand1]-time[dr[i]-1]) < gapwindow))
#        x2 = np.where((np.abs(time[cand1]-time[dl[i]]) < gapwindow))
#        cand1 = np.delete(cand1, x1)
#        cand1 = np.delete(cand1, x2)

#    # find start and stop index, combine neighboring candidates in to same events
#    cstart = cand1[np.append([0], np.where((cand1[1:]-cand1[:-1] > minsep))[0]+1)]
#    cstop = cand1[np.append(np.where((cand1[1:]-cand1[:-1] > minsep))[0], [len(cand1)-1])]

#    # for now just return index of candidates
#    if returnall is True:
#        return cstart, cstop, cand1
#    else:
#        return cstart, cstop


def FINDflare(flux, error, N1=3, N2=1, N3=3,
              avg_std=False, std_window=7,
              returnbinary=False, debug=False):
    '''
    The algorithm for local changes due to flares defined by
    S. W. Chang et al. (2015), Eqn. 3a-d
    http://arxiv.org/abs/1510.01005

    Note: these equations originally in magnitude units, i.e. smaller
    values are increases in brightness. The signs have been changed, but
    coefficients have not been adjusted to change from log(flux) to flux.

    Note: this algorithm originally ran over sections without "changes" as
    defined by Change Point Analysis. May have serious problems for data
    with dramatic starspot activity. If possible, remove starspot first!

    Parameters
    ----------
    flux : numpy array
        data to search over
    error : numpy array
        errors corresponding to data.
    N1 : int, optional
        Coefficient from original paper (Default is 3)
        How many times above the stddev is required.
    N2 : int, optional
        Coefficient from original paper (Default is 1)
        How many times above the stddev and uncertainty is required
    N3 : int, optional
        Coefficient from original paper (Default is 3)
        The number of consecutive points required to flag as a flare
    avg_std : bool, optional
        Should the "sigma" in this data be computed by the median of
        the rolling().std()? (Default is False)
        (Not part of original algorithm)
    std_window : float, optional
        If avg_std=True, how big of a window should it use?
        (Default is 25 data points)
        (Not part of original algorithm)
    returnbinary : bool, optional
        Should code return the start and stop indicies of flares (default,
        set to False) or a binary array where 1=flares (set to True)
        (Not part of original algorithm)
    '''

    med_i = np.nanmedian(flux)

    if debug is True:
        print("DEBUG: med_i = " + str(med_i))

    if avg_std is False:
        sig_i = np.nanstd(flux) # just the stddev of the window
    else:
        # take the average of the rolling stddev in the window.
        # better for windows w/ significant starspots being removed
        sig_i = np.nanmedian(pd.Series(flux).rolling(std_window, center=True).std())
    if debug is True:
        print("DEBUG: sig_i = " + str(sig_i))

    ca = flux - med_i
    cb = np.abs(flux - med_i) / sig_i
    cc = np.abs(flux - med_i - error) / sig_i

    if debug is True:
        print("DEBUG: N0={}, N1={}, N2={}".format(sum(ca>0),sum(cb>N1),sum(cc>N2)))

    # pass cuts from Eqns 3a,b,c
    ctmp = np.where((ca > 0) & (cb > N1) & (cc > N2))

    cindx = np.zeros_like(flux)
    cindx[ctmp] = 1

    # Need to find cumulative number of points that pass "ctmp"
    # Count in reverse!
    ConM = np.zeros_like(flux)
    # this requires a full pass thru the data -> bottleneck
    for k in range(2, len(flux)):
        ConM[-k] = cindx[-k] * (ConM[-(k-1)] + cindx[-k])

    # these only defined between dl[i] and dr[i]
    # find flare start where values in ConM switch from 0 to >=N3
    istart_i = np.where((ConM[1:] >= N3) &
                        (ConM[0:-1] - ConM[1:] < 0))[0] + 1

    # use the value of ConM to determine how many points away stop is
    istop_i = istart_i + (ConM[istart_i] - 1)

    istart_i = np.array(istart_i, dtype='int')
    istop_i = np.array(istop_i, dtype='int')

    if returnbinary is False:
        return istart_i, istop_i
    else:
        bin_out = np.zeros_like(flux, dtype='int')
        for k in range(len(istart_i)):
            bin_out[istart_i[k]:istop_i[k]+1] = 1
        return bin_out


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








#def MeasureS2N(flux, error, model, istart=-1, istop=-1):
#    '''
#    this MAY NOT be something i want....
#    '''
#    if (istart < 0):
#        istart = 0
#    if (istop < 0):
#        istop = len(flux)

#    flareflux = flux[istart:istop+1]
#    modelflux = model[istart:istop+1]

#    s2n = np.sum(np.sqrt((flareflux) / (flareflux + modelflux)))
#    return s2n


#def FlarePer(time, minper=0.1, maxper=30.0, nper=20000):
#    '''
#    Look for periodicity in the flare occurrence times. Could be due to:
#    a) mis-identified periodic things (e.g. heartbeat stars)
#    b) crazy binary star flaring things
#    c) flares on a rotating star
#    d) bugs in code
#    e) aliens

#    '''

#    # use energy = 1 for flare times.
#    # This will create something like the window function
#    energy = np.ones_like(time)

#    # Use Jake Vanderplas faster version!
#    pgram = LombScargleFast(fit_offset=False)
#    pgram.optimizer.set(period_range=(minper,maxper))

#    pgram = pgram.fit(time, energy - np.nanmedian(energy))

#    df = (1./minper - 1./maxper) / nper
#    f0 = 1./maxper
#    pwr = pgram.score_frequency_grid(f0, df, nper)

#    freq = f0 + df * np.arange(nper)
#    per = 1./freq

#    pk = per[np.argmax(pwr)] # peak period
#    pp = np.max(pwr) # peak period power

#    return pk, pp


def MultiFind(lc, dl,dr,mode,
              gapwindow=0.1, minsep=3, debug=False):
    '''
    this needs to be either
    1. made in to simple multi-pass cleaner,
    2. made in to "run till no signif change" cleaner, or
    3. folded back in to main code
    '''

    lc['flux_model'] = 0.
    istart = np.array([], dtype='int')
    istop = np.array([], dtype='int')
    flux_model = lc.flux_model.copy().values

    for le,ri in list(zip(dl,dr)):
        lct = lc.iloc[le:ri].copy()
        time, flux, error, flags = lct.time.values, lct.flux.values, lct.error.values, lct.flags.values
        # the bad data points (search where bad < 1)
        bad = FlagCuts(flags, returngood=False)
        flux_i = np.copy(flux)

        if (mode == 0):
            # only for fully preprocessed LCs like K2SC
            flux_model_i = np.nanmedian(flux) * np.ones_like(flux)
            flux_diff = flux - flux_model_i

        if (mode == 1):
            # just use the multi-pass boxcar and average. Simple. Too simple...
            flux_model1 = detrend.MultiBoxcar(time, flux, error, kernel=0.1)
            flux_model2 = detrend.MultiBoxcar(time, flux, error, kernel=1.0)
            flux_model3 = detrend.MultiBoxcar(time, flux, error, kernel=10.0)

            flux_model_i = (flux_model1 + flux_model2 + flux_model3) / 3.
            flux_diff = flux - flux_model_i

        if (mode == 2):
            # first do a pass thru w/ largebox to get obvious flares
            box1 = detrend.MultiBoxcar(time, flux_i, error, kernel=2.0, numpass=2)
            sin1 = detrend.FitSin(time, box1, error, maxnum=2, maxper=(max(time)-min(time)))

            box2 = detrend.MultiBoxcar(time, flux_i - sin1, error, kernel=0.25)
            flux_model_i = (box2 + sin1)
            flux_diff = flux - flux_model_i


        if (mode == 3):
            # do iterative rejection and spline fit - like FBEYE did
            # also like DFM & Hogg suggest w/ BART
            box1 = detrend.MultiBoxcar(time, flux_i, error, kernel=2.0, numpass=2)

            sin1 = detrend.FitSin(time, box1, error, maxnum=5, maxper=(max(time)-min(time)),
                                  per2=False, debug=debug)
            box3 = detrend.MultiBoxcar(time, flux_i - sin1, error, kernel=0.3)
            t = np.array(time)
            dt = np.nanmedian(t[1:] - t[0:-1])
            #print(dt)
            exptime_m = (np.nanmax(time) - np.nanmin(time)) / len(time)
            # ksep used to = 0.07...
            flux_model_i = detrend.IRLSSpline(time, box3, error, numpass=20, debug=debug, ksep=exptime_m*10.) + sin1
            signalfwhm = dt * 2
            ftime = np.arange(0, 2, dt)
            modelfilter = aflare1(ftime, 1, signalfwhm, 1)
            flux_diff = signal.correlate(flux_i - flux_model_i, modelfilter, mode='same')
        if (mode == 4):
            # fit data with a SAVGOL filter
            dt = np.nanmedian(time[1:] - time[0:-1])
            Nsmo = np.floor(0.2 / dt)
            if Nsmo % 2 == 0:
                Nsmo = Nsmo + 1
            flux_model_i = savgol_filter(flux, Nsmo, 2, mode='nearest')
            flux_diff = flux - flux_model_i


        # run final flare-find on DATA - MODEL
        isflare = FINDflare(flux_diff, error, N1=3, N2=4, N3=3,
                            returnbinary=True, avg_std=True)

        # now pick out final flare candidate points from above
        cand1 = np.where((bad < 1) & (isflare > 0))[0]

        x1 = np.where((np.abs(time[cand1]-time[-1]) < gapwindow))
        x2 = np.where((np.abs(time[cand1]-time[0]) < gapwindow))
        cand1 = np.delete(cand1, x1)
        cand1 = np.delete(cand1, x2)
        if (len(cand1) < 1):
            istart_i = np.array([])
            istop_i = np.array([])
        else:
            # find start and stop index, combine neighboring candidates in to same events
            istart_i = cand1[np.append([0], np.where((cand1[1:]-cand1[:-1] > minsep))[0]+1)]
            istop_i = cand1[np.append(np.where((cand1[1:]-cand1[:-1] > minsep))[0], [len(cand1)-1])]
        # if start & stop times are the same, add 1 more datum on the end
        to1 = np.where((istart_i-istop_i == 0))
        if len(to1[0])>0:
            istop_i[to1] += 1

        if debug is True:
            plt.figure()
            plt.title('debugging plot')
            plt.scatter(time, flux_i, alpha=0.5,label='flux')
            plt.plot(time,flux_model_i, c='black',label='flux model')
            plt.scatter(time[cand1], flux_i[cand1], c='red',label='flare candidates')
            plt.legend()
            plt.show()
            plt.close()

        #chi2 = chisq(lc.flux.values[le:ri], flux_model_i, error[le:ri])
        istart = np.array(np.append(istart, istart_i + le), dtype='int')
        istop = np.array(np.append(istop, istop_i + le), dtype='int')
        flux_model[le:ri] = flux_model_i
    return istart, istop, flux_model


# sketching some baby step ideas w/a matched filter
#'''
#def MatchedFilterFind(time, flux, signalfwhm=0.01):
#    dt = np.nanmedian(time[1:] - time[0:-1])
#    ftime = np.arange(0, 2, dt)
#    modelfilter = aflare1(ftime, 1, signalfwhm, 1)

#    corr = signal.correlate(flux-np.nanmedian(flux), modelfilter, mode='same')

#    plt.figure()
#    plt.plot(time, flux - np.nanmedian(flux))
#    plt.plot(time, corr,'red')
#    plt.show()

#    return
#'''


def FakeFlares(df1, lc, dl, dr, mode, gapwindow=0.1, nfake=10, debug=False, npass=1,
                savefile=False, outfile='', display=False, verboseout = False):

    '''
    Create nfake number of events, inject them in to data
    Use grid of amplitudes and durations, keep ampl in relative flux units
    Keep track of energy in Equiv Dur.

    Parameters:
    -------------
    time
    flux
    error
    flags
    tstart
    tstop

    nfake =10
    npass =1
    outfile =''
    savefile =False
    gapwindow =0.1
    verboseout =False
    display =False
    debug =False

    Returns:
    ------------
    ed_fake
    rec_fake
    ed_rec
    ed_rec_err
    istart_rec
    istop_rec

    duration defined in minutes
    amplitude defined multiples of the median error
    still need to implement npass, to re-do whole thing and average results
    '''
    fakeres = pd.DataFrame()
    new_flux = np.array(lc.flux)
    for l,r in list(zip(dl,dr)):
    	df2t= lc.iloc[l:r]

    	df1t = df1[(df1.istart >= l) & (df1.istop <= r)]
    	medflux = df2t.flux_model.median()# flux needs to be normalized
    	tstart = df2t.time[df2t.index.isin(df1t.istart)].values
    	tstop = df2t.time[df2t.index.isin(df1t.istop)].values
    	print(tstart, tstop)
    	flags = df2t.flags.values
    	error = df2t.error.values / medflux
    	flux = df2t.flux.values / medflux - 1.
    	time = df2t.time.values
    	std = np.nanmedian(error)

    	dur_fake, ampl_fake = FakeFlaresDist(std, nfake, mode='hawley2014', debug=debug)
    	t0_fake = np.zeros(nfake, dtype='float')
    	ed_fake = np.zeros(nfake, dtype='float')


    	for k in range(nfake):
    	    # generate random peak time, avoid known flares
    	    isok = False
    	    while isok is False:
    	        # choose a random peak time
    	        t0 =  np.random.choice(time)
    	        if len(tstart)>0:
    	            print('EHEH', t0, tstart, tstop)
    	            if ~(np.any(t0 >= tstart) & np.any(t0 <= tstop)):
                        isok = True
    	           # x = np.where((t0 >= tstart) & (t0 <= tstop))
    	           # if (len(x[0]) < 1):
    	           #     isok = True
    	        else: isok = True
    	    t0_fake[k] = t0
    	    # generate the fake flare
    	    fl_flux = aflare1(time, t0, dur_fake[k], ampl_fake[k])
    	    ed_fake[k] = EquivDur(time, fl_flux)
    	    # inject flare in to light curve
    	    new_flux[l:r] = new_flux[l:r]+ fl_flux
    '''
    Re-run flare finding for data + fake flares
    Figure out: which flares were recovered?
    '''
    # all the hard decision making should go herehere
    new_lc = pd.DataFrame({'flux':new_flux,'time':lc.time, 'error':lc.error, 'flags':lc.flags})
    istart, istop, flux_model = MultiFind(new_lc,dl,dr, mode, gapwindow=gapwindow, debug=debug)
    header = ['ed_fake','rec_fake','ed_rec','ed_rec_err','istart_rec','istop_rec']
    rec_fake, ed_rec, ed_rec_err = np.zeros(nfake), np.zeros(nfake), np.zeros(nfake)
    istart_rec, istop_rec = np.zeros(nfake), np.zeros(nfake)
    if len(istart)>0: # in case no flares are recovered, even after injection
        for k in range(nfake): # go thru all recovered flares
            # do any injected flares overlap recovered flares?
            rec = np.where((t0_fake[k] >= lc.time[istart]) & (t0_fake[k] <= lc.time[istop]))
            if (len(rec[0]) > 0):
                rec_fake[k] = 1
                ed_rec[k], ed_rec_err[k], _ = ED(istart[rec[0]], istop[rec[0]],lc.time, lc.flux_model, new_flux, lc.error)
                istart_rec[k], istop_rec[k] = istart[rec[0]],istop[rec[0]]
                istart = np.delete(istart,rec[0])
                istop = np.delete(istop,rec[0])
    cols=[]
    for name in header:
        cols.append(vars()[name])
    df = pd.DataFrame(dict(zip(header,cols)))
    fakeres = fakeres.append(df, ignore_index=True)

    if savefile is True:

        header = ['min_time','max_time','std_dev','nfake',
                  'min_amplitude','max_amplitude',
                  'min_duration','max_duration',
                  'tstamp',
                 ]

        if glob.glob(outfile)==[]:
            dfout = pd.DataFrame()
        else:
            dfout = pd.read_json(outfile)

        tstamp = clock.asctime(clock.localtime(clock.time()))
        outrow = [[item] for item in [min(time), max(time), std, nfake, min(ampl_fake),
                                      max(ampl_fake),min(dur_fake),max(dur_fake),tstamp]]
        dfout = dfout.append(pd.DataFrame(dict(zip(header,outrow))),
                                 ignore_index=True)
        dfout.to_json(outfile)

    #centers of bins, fraction of recovered fake flares per bin, EDs of generated fake flares,
    return fakeres



# objectid = '9726699'  # GJ 1243
def RunLC(file='', objectid='', ftype='sap', lctype='',
          display=False, readfile=False, debug=False, dofake=True,
          dbmode='fits', gapwindow=0.1, maxgap=0.125,
          nfake=100, mode=3,iterations=10):
    '''
    Main wrapper to obtain and process a light curve
    '''


    # pick and process a totally random LC.
    # important for reality checking!
    if (objectid is 'random'):
        obj, num = np.loadtxt('get_objects.out', skiprows=1, unpack=True, dtype='str')
        rand_id = int(np.random.random() * len(obj))
        objectid = obj[rand_id]
        print('Random ObjectID Selected: ' + objectid)

    # get the data
    if debug is True:
        print(str(datetime.datetime.now()) + ' GetLC started')
        print(file, objectid)

    #---------------------------------------------------
    if dbmode is 'mysql':
        data_raw = GetLCdb(objectid, readfile=readfile, type=lctype, onecadence=False)
        data = OneCadence(data_raw)

        # data columns are:
        # QUARTER, TIME, PDCFLUX, PDCFLUX_ERR, SAP_QUALITY, LCFLAG, SAPFLUX, SAPFLUX_ERR

        qtr = data[:,0]
        time = data[:,1]
        lcflag = data[:,4] # actual SAP_QUALITY

        exptime = data[:,5] # actually the LCFLAG
        exptime[np.where((exptime < 1))] = 54.2 / 60. / 60. / 24.
        exptime[np.where((exptime > 0))] = 30 * 54.2 / 60. / 60. / 24.

        if ftype == 'sap':
            flux_raw = data[:,6]
            error = data[:,7]
        else: # for PDC data
            flux_raw = data[:,2]
            error = data[:,3]

        # put flare output in to a set of subdirectories.
        # use first 3 digits to help keep directories to ~1k files
        fldr = objectid[0:3]
        outdir = 'aprun/' + fldr + '/'
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        # open the output file to store data on every flare recovered
        outfile = outdir + objectid

    elif dbmode in ('txt','ktwo','everest','vdb','csv','kplr','k2sc'):
        #outfile, objectid, qtr, time, lcflag, exptime, flux_raw, error = Get(dbmode,file, objectid)
        outfile, objectid, lc = Get(dbmode,file, objectid)

    #-----------------------------------------------

    if debug is True:
        print('outfile = ' + outfile)

    ### Basic flattening
    # flatten quarters with polymonial
    flux_qtr = detrend.QtrFlat(lc.time.values, lc.flux_raw.values, lc.qtr.iloc[0])

    # then flatten between gaps
    lc['flux'] = detrend.GapFlat(lc.time.values, flux_qtr, maxgap=maxgap)

    #find continuous observing periods
    _, dl, dr = detrend.FindGaps(lc.time.values, maxgap=maxgap)
    if debug is True:
        print("dl: {}, dr: {}".format(dl,dr))

    # uQtr = np.unique(qtr)
    if debug is True:
        print(str(datetime.datetime.now()) + ' MultiFind started')
    istart, istop, lc['flux_model'] = MultiFind(lc,dl,dr,gapwindow=gapwindow, debug=debug, mode=mode)

    df1 = pd.DataFrame({'istart':istart,
                        'istop':istop,
                        'ed68':np.full_like(istart,-99),
                        'ed90':np.full_like(istart,-99)})
    allfakes = pd.DataFrame()
    # run artificial flare test in this gap
    if debug is True:
        print(str(datetime.datetime.now()) + ' FakeFlares started')

    if dofake is True:
        dffake = pd.DataFrame()
        for k in range(iterations):
            fakeres = FakeFlares(df1, lc, dl, dr, mode, savefile=True,gapwindow=gapwindow,outfile=outfile[:outfile.find('.')]+'_fake.json',display=display, nfake=nfake, debug=debug)
            dffake = dffake.append(fakeres, ignore_index=True)

        dffake.to_csv('{}_all_fakes.csv'.format(outfile))

        df1['ed68'], df1['ed90'] = FakeCompleteness(dffake,nfake,iterations, display=display,file=objectid)



#    '''
#    ### MY FIRST ATTEMPT AT FLARE FINDING
#    # fit sin curves
#    flux_sin = detrend.FitSin(time, flux, error)

#    # run iterative boxcar over data
#    flux_smo = detrend.MultiBoxcar(time, flux - flux_sin, error)

#    flux_model = flux_sin + flux_smo
#    # flux_model = flux_sin + detrend.MultiBoxcar(time, flux_smo, error) # double detrend?

#    istart, istop = DetectCandidate(time, flux, error, lcflag, flux_model)
#    '''


    if display is True:
        print(str(len(istart))+' flare candidates found.')

        plt.figure(figsize=(8,4))
        plt.scatter(lc.time, lc.flux, c='k', alpha=0.7,)

        for g in range(len(istart)):
            lct = lc.iloc[istart[g]:istop[g]+1]
            plt.scatter(lct.time,lct.flux,color='red')

        plt.plot(lc.time, lc.flux_model, 'blue', lw=3, alpha=0.7)
        for i in range(1,4):
            plt.fill_between(lc.time, lc.flux_model+lc.error*i,
                             y2=lc.flux_model-lc.error*i,
                             color='green',alpha=0.25)

        plt.xlabel('Time (BJD - 2454833 days)')
        plt.ylabel(r'Flux (e- sec$^{-1}$)')

        xdur = np.nanmax(lc.time) - np.nanmin(lc.time)
        xdur0 = np.nanmin(lc.time) + xdur/2.
        xdur1 = np.nanmin(lc.time) + xdur/2. + 5.
        plt.xlim(xdur0, xdur1) # only plot a chunk of the data


#        xdurok = np.where((lc.time >= xdur0) & (lc.time <= xdur1))
        lct = lc[(lc.time >= xdur0) & (lc.time <= xdur1)]
        if lct.shape[0] != 0:
            yminmax = [np.nanmin(lct.flux), np.nanmax(lct.flux)]
            if sum(np.isfinite(yminmax)) > 1:
                plt.ylim(yminmax[0], yminmax[1])

        plt.savefig(file + '_lightcurve.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.show()
    '''
    #-- IF YOU WANT TO PLAY WITH THE WAVELET STUFF MORE, WORK HERE
    test_model = detrend.WaveletSmooth(time, flux)
    test_cand = DetectCandidate(time, flux, error, test_model)

    print(len(cand))
    print(len(test_cand))

    if display is True:
        plt.figure()
        plt.plot(time, flux, 'k')
        plt.plot(time, test_model, 'green')

        plt.scatter(time[test_cand], flux[test_cand], color='orange', marker='p',s=60, alpha=0.8)
        plt.show()
    '''

    # set this to silence bad fit warnings from polyfit
    warnings.simplefilter('ignore', np.RankWarning)

    metadata = {'ObjectID' : objectid,
                 'File' : file,
                 'Date-Run' : str(datetime.datetime.now()),
                 'Appaloosa-Version': __version__,
                 'N_epoch in LC' : str(len(lc.time)),
                 'Total exp time of LC' : str(np.sum(lc.exptime)),
                 }

    if debug is True:
        print(str(datetime.datetime.now()) + 'Getting output header')

    header = FlareStats(lc, ReturnHeader=True)
    header = header + ['ED68i','ED90i']
    dfout = pd.DataFrame()

    if debug is True:
        print(str(datetime.datetime.now()) + 'Getting FlareStats')

    # loop over EACH FLARE, compute stats
    df2 = pd.DataFrame({'istart':istart,'istop':istop})
    stats_i = []

    for i in range(0,len(istart)):
        stats_i = FlareStats(lc, istart=istart[i], istop=istop[i])

        stats_i = np.append(stats_i,[df1.ed68.iloc[i],df1.ed90.iloc[i]])
        stats_i = [[item] for item in stats_i]
        dfout = dfout.append(pd.DataFrame(dict(zip(header,stats_i))),
                             ignore_index=True)
    if not dfout.empty:
        dfout.to_csv(outfile + '_flare_stats.csv')
        with open(outfile + '_flare_stats_meta.json','w') as f:
            j = json.dumps(metadata)
            f.write(j)

    #Add the number of flares from this LC to the list
    flist = 'flarelist.csv'.format(outfile)
    header = ['Object ID',' Date of Run','Number of Flares','Filename',
                        'Total Exposure Time of LC in Days','BJD-2454833 days']

    if glob.glob(flist)==[]:
        dfout = pd.DataFrame()
    else:
        dfout = pd.read_csv(flist)

    line=[objectid, datetime.datetime.now(), len(istart),file,np.sum(lc.exptime),lc.time[0]]
    line = [[item] for item in line]
    dfout = dfout.append(pd.DataFrame(dict(zip(header,line))),
                             ignore_index=True)
    dfout.to_csv(flist)
    return
# let this file be called from the terminal directly. e.g.:
# $python appaloosa.py 12345678
if __name__ == "__main__":
    import sys
    RunLC(file=str(sys.argv[1]), dbmode='fits', display=True, debug=True, nfake=100)
