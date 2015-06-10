'''
Hold the detrending method(s) to use.

1) Write a sliding local polynomial smoother
2) translate softserve (?)

'''
import numpy as np
from pandas import rolling_median

def polysmooth(time, flux, error, qtr):
    uQtr = np.unique(qtr)

    smo = flux
    return smo


def QtrFlat(time, flux, qtr):
    '''
    step thru each unique qtr
    fit 2nd order poly to smoothed version of qtr
    return flat lc

    ignore long/short cadence, deal with on front end
    '''

    uQtr = np.unique(qtr)

    tot_med = np.median(flux) # the total from all quarters

    flux_flat = flux

    for q in uQtr:
        x = np.where( (np.abs(qtr-q) < 0.1) ) # find all epochs within each Qtr, but careful w/ floats

        krnl = float(len(x[0])) / 100.0
        if krnl < 15.0:
            krnl = 15.0

        flux_sm = rolling_median(flux[x], krnl)

        indx = np.isfinite(flux_sm) # get rid of NaN's put in by rolling_median.

        fit = np.polyfit(time[x][indx], flux_sm[indx], 1)

        flux_flat[x] = flux[x] - np.polyval(fit, time[x]) + tot_med

    return flux_flat