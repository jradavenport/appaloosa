import numpy as np
import matplotlib.pyplot as plt

def chisq(data, error, model):
    '''
    Compute the normalized chi square statistic:
    chisq =  1 / N * SUM(i) ( (data(i) - model(i))/error(i) )^2
    '''
    return np.sum( ((data - model) / error)**2.0 ) / np.size(data)

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

def Plot(lc, ax, istart=None,istop=None,onlybit=None):

    '''
    Plot the lightcurve with standard deviation to a given instance of
    matplotlib.pyplot.Axes
    '''

    ax.scatter(lc.time, lc.flux, c='k', alpha=0.7,)
    ax.plot(lc.time, lc.flux_model, 'blue', lw=3, alpha=0.7)
    for i in range(1,4):
        ax.fill_between(lc.time, lc.flux_model+lc.error*i,
                         y2=lc.flux_model-lc.error*i,
                         color='green',alpha=0.25)
    ax.set_xlabel('Time (BJD - 2454833 days)')
    ax.set_ylabel(r'Flux (e- sec$^{-1}$)')

    if  np.all(istart != None) & np.all(istop != None):
        for (l,r) in list(zip(istart,istop)):
            lct = lc.iloc[l:r+1]
            ax.scatter(lct.time, lct.flux, color='red')
    if onlybit != None :
        xdur0 = (lc.time.min() + lc.time.max()) / 2.
        ax.set_xlim(xdur0, xdur0 + onlybit) # only plot a chunk of the data
    else:
        ax.set_ylim(lc.flux.min(), lc.flux.max())

    return ax

#UNUSED
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
