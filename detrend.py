'''
Hold the detrending method(s) to use.

1) Write a sliding local polynomial smoother
2) translate softserve (?)

'''
import numpy as np
from pandas import rolling_median, rolling_mean
from scipy.optimize import curve_fit
from gatspy.periodic import LombScargleFast
# from scipy import signal



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


def FindGaps(time, maxgap=0.125, minspan=3.0):
    # assumes data is already sorted!
    dt = time[1:] - time[:-1]
    gap = np.where((dt >= maxgap))[0]

    # remove gaps that are too close together

    # add start/end of LC to loop over easily
    gap_out = np.append(0, np.append(gap, len(time)))

    right = np.append(gap + 1, len(time)) # right end of data
    left = np.append(0, gap + 1) # left start of data

    return gap_out, left, right


def _sinfunc(t, per, amp, t0, yoff):
    return np.sin((t - t0) * 2.0 * np.pi / per) * amp  + yoff


def FitSin(time, flux, error, maxnum = 5, nper=2000):
    _, dl, dr = FindGaps(time) # finds right edge of time windows

    minper = 0.1 # days
    maxper = 30. # days

    periods = np.linspace(minper, maxper, nper)

    flux_out = np.array(flux, copy=True)
    sin_out = np.zeros_like(flux) # return the sin function!

    # now loop over every chunk of data and fit N periods
    for i in range(0, len(dl)):
        # total baseline of time window
        dt = max(time[dl[i]:dr[i]]) - min(time[dl[i]:dr[i]])

        medflux = np.median(flux[dl[i]:dr[i]])
        ti = time[dl[i]:dr[i]]

        freq = 2.0 * np.pi / periods[np.where((periods < dt))]

        for k in range(0, maxnum):
            # pgram = signal.lombscargle(ti, flux_out[dl[i]:dr[i]] - medflux, freq)
            # pwr = np.sqrt(4. * (pgram / time.shape[0]))
            # find the period with the peak power
            # pk = periods[np.where((periods < dt))][np.argmax(pwr)]

            # try Jake Vanderplas faster version!
            pgram = LombScargleFast().fit(time, flux - medflux, error)
            pwr, per = pgram.periodogram_auto()
            pk = per[np.where((per < dt))][np.argmax(pwr)]


            # fit sin curve to window and subtract
            p0 = [pk, 3.0 * np.nanstd(flux_out[dl[i]:dr[i]]-medflux), 0.0, 0.0]
            pfit = curve_fit(_sinfunc, ti,
                             flux_out[dl[i]:dr[i]]-medflux, p0=p0)

            flux_out[dl[i]:dr[i]] = flux_out[dl[i]:dr[i]] - _sinfunc(ti, *pfit[0])
            sin_out[dl[i]:dr[i]] = sin_out[dl[i]:dr[i]] + _sinfunc(ti, *pfit[0])

        # add the median flux for this window BACK in
        sin_out[dl[i]:dr[i]] = sin_out[dl[i]:dr[i]] + medflux

    return sin_out

