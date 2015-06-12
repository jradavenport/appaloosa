'''
Hold the detrending method(s) to use.

1) Write a sliding local polynomial smoother
2) translate softserve (?)

'''
import numpy as np
from pandas import rolling_median, rolling_mean
from scipy import signal
from scipy.optimize import curve_fit


def rolling_poly(time, flux, error, order=3, window=0.5):
    # this is SUPER slow... maybe useful in some places (LLC only?)

    smo = np.zeros_like(flux)

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


def FindGaps(time, min=0.125):
    # assumes data is already sorted!
    dt = time[1:] - time[:-1]
    gap1 = np.where((dt >= min))[0]
    # add start/end of LC to loop over easily
    gap = np.append(0, np.append(gap1, len(time)-1))
    return gap


def _sinfunc(t, per, amp, t0, yoff):
    return np.sin((t - t0) * 2.0 * np.pi / per) * amp  + yoff


def FitSin(time, flux, maxnum = 3):
    gap = FindGaps(time) # finds right edge of time windows

    minper = 0.1 # days
    maxper = 30. # days
    nper = 2000
    periods = np.linspace(minper, maxper, nper)

    flux_out = np.array(flux, copy=True)
    sin_out = np.zeros_like(flux) # return the sin function!

    for i in range(0, len(gap)-1):
        ioff = 0
        if i > 0:
            ioff = 1
        rng = [gap[i]+ioff, gap[i+1]]

        # total baseline of time window
        dt = time[rng[1]] - time[rng[0]]

        medflux = np.median(flux[rng[0]:rng[1]])
        ti = time[rng[0]:rng[1]]

        freq = 2.0 * np.pi / periods[np.where((periods < dt))]

        for k in range(0, maxnum):
            pgram = signal.lombscargle(ti, flux_out[rng[0]:rng[1]] - medflux,
                                       freq)

            pwr = np.sqrt(4. * (pgram / time.shape[0]))

            # find the period with the peak power
            pk = periods[np.where((periods < dt))][np.argmax(pwr)]

            # fit sin curve to window and subtract
            p0 = [pk, 3.0 * np.nanstd(flux_out[rng[0]:rng[1]]-medflux), 0.0, 0.0]
            pfit = curve_fit(_sinfunc, ti,
                             flux_out[rng[0]:rng[1]]-medflux, p0=p0)

            flux_out[rng[0]:rng[1]] = flux_out[rng[0]:rng[1]] - _sinfunc(ti, *pfit[0])
            sin_out[rng[0]:rng[1]] = sin_out[rng[0]:rng[1]] + _sinfunc(ti, *pfit[0])

    return sin_out

