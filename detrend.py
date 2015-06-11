'''
Hold the detrending method(s) to use.

1) Write a sliding local polynomial smoother
2) translate softserve (?)

'''
import numpy as np
from pandas import rolling_median


def rolling_poly(time, flux, error, order=3, window=0.5):
    smo = flux

    w1 = np.where((time >= time[0] + window / 2.0) &
                  (time <= time[-1] + window / 2.0 ))[0]

    for i in range(0,len(w1)):
        x = np.where((time[w1] >= time[w1][i] - window / 2.0) &
                     (time[w1] <= time[w1][i] + window / 2.0))

        fit = np.polyfit(time[w1][x], flux[w1][x], order,
                          w = (1. / error[w1][x]) )

        smo[w1[i]] = np.polyval(fit, time[w1][i])

    return smo


def QtrFlat(time, flux, qtr, order=3):
    '''
    step thru each unique qtr
    fit 2nd order poly to smoothed version of qtr
    return flat lc

    ignore long/short cadence, deal with on front end
    '''

    uQtr = np.unique(qtr)

    tot_med = np.median(flux) # the total from all quarters

    flux_flat = np.ones_like(flux) * tot_med

    for q in uQtr:
        # find all epochs within each Qtr, but careful w/ floats
        x = np.where( (np.abs(qtr-q) < 0.1) )

        krnl = float(len(x[0])) / 100.0
        if (krnl < 10.0):
            krnl = 10.0

        flux_sm = rolling_median(flux[x], krnl)

        indx = np.isfinite(flux_sm) # get rid of NaN's put in by rolling_median.

        fit = np.polyfit(time[x][indx], flux_sm[indx], order)

        flux_flat[x] = flux[x] - np.polyval(fit, time[x]) + tot_med

    return flux_flat


