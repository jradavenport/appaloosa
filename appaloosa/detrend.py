'''
Use this file to keep various detrending methods

'''
import numpy as np
from pandas import rolling_median #, rolling_mean, rolling_std, rolling_skew
from scipy.optimize import curve_fit
from gatspy.periodic import LombScargleFast
from gatspy.periodic import SuperSmoother
# import pywt
from scipy import signal
from scipy.interpolate import LSQUnivariateSpline, UnivariateSpline
import matplotlib.pyplot as plt


def rolling_poly(time, flux, error, order=3, window=0.5):
    '''
    Fit polynomials in a sliding window
    Name convention meant to match the pandas rolling_ stats

    Parameters
    ----------
    time : 1-d numpy array
    flux : 1-d numpy array
    error : 1-d numpy array
    order : int, optional
    window : float, optional

    Returns
    -------
    '''

    # This is SUPER slow... maybe useful in some places (LLC only?).
    # Can't be sped up much w/ indexing, because needs to move fixed
    # windows of time...

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


def GapFlat(time, flux, order=3, maxgap=0.125):
    '''

    Parameters
    ----------

    Returns
    -------
    Data with polymonials removed
    '''
    _, dl, dr = FindGaps(time, maxgap=maxgap) # finds right edge of time windows

    tot_med = np.nanmedian(flux) # the total from all quarters

    flux_flat = np.array(flux, copy=True)

    for i in range(0, len(dl)):
        krnl = int(float(dl[i]-dr[i]) / 100.0)
        if (krnl < 10):
            krnl = 10
        flux_sm = rolling_median(flux[dl[i]:dr[i]], krnl)

        indx = np.isfinite(flux_sm)

        fit = np.polyfit(time[dl[i]:dr[i]][indx], flux_sm[indx], order)

        flux_flat[dl[i]:dr[i]] = flux[dl[i]:dr[i]] - \
                                 np.polyval(fit, time[dl[i]:dr[i]]) + \
                                 tot_med
    return flux_flat


def QtrFlat(time, flux, qtr, order=3):
    '''
    step thru each unique qtr
    fit 2nd order poly to smoothed version of qtr

    Returns
    -------
    Data with polymonials removed

    ignore long/short cadence, deal with on front end
    '''

    uQtr = np.unique(qtr)

    tot_med = np.nanmedian(flux) # the total from all quarters

    flux_flat = np.ones_like(flux) * tot_med

    for q in uQtr:
        # find all epochs within each Qtr, but careful w/ floats
        x = np.where( (np.abs(qtr-q) < 0.1) )

        krnl = int(float(len(x[0])) / 100.0)
        if (krnl < 10):
            krnl = 10

        flux_sm = rolling_median(np.array(flux[x], dtype='float'), krnl)

        indx = np.isfinite(flux_sm) # get rid of NaN's put in by rolling_median.

        fit = np.polyfit(time[x][indx], flux_sm[indx], order)

        flux_flat[x] = flux[x] - np.polyval(fit, time[x]) + tot_med

    return flux_flat


def FindGaps(time, maxgap=0.125, minspan=2.0):
    '''

    Parameters
    ----------

    Returns
    -------
    outer edges of gap, left edges, right edges
    (all are indicies)
    '''
    # assumes data is already sorted!
    dt = time[1:] - time[:-1]
    gap = np.where((dt >= maxgap))[0]

    # add start/end of LC to loop over easily
    gap_out = np.append(0, np.append(gap, len(time)))

    right = np.append(gap + 1, len(time)) # right end of data
    left = np.append(0, gap + 1) # left start of data

    # remove gaps that are too close together

    # ok = np.where((time[right]-time[left] >= minspan))[0]
    # bad = np.where((time[right]-time[left] < minspan))[0]
    # for k in range(1,len(bad)-1):
        # for each bad span of data, figure out if it can be tacked on

    return gap_out, left, right


def _sinfunc(t, per, amp, t0, yoff):
    return np.sin((t - t0) * 2.0 * np.pi / per) * amp  + yoff

def _sinfunc2(t, per1, amp1, t01, per2, amp2, t02, yoff):
    output = np.sin((t - t01) * 2.0 * np.pi / per1) * amp1 + \
             np.sin((t - t02) * 2.0 * np.pi / per2) * amp2 + yoff
    return output


def FitSin(time, flux, error, maxnum=5, nper=20000,
           minper=0.1, maxper=30.0, plim=0.25,
           returnmodel=True, debug=False, per2=False):
    '''
    Use Lomb Scargle to find periods, fit sins, remove, repeat.

    Parameters
    ----------
    time:
    flux:
    error:
    maxnum:
    nper: int, optional
        number of periods to search over with Lomb Scargle
    minper:
    maxper:
    plim:
    debug:

    Returns
    -------
    '''
    # periods = np.linspace(minper, maxper, nper)

    flux_out = np.array(flux, copy=True)
    sin_out = np.zeros_like(flux) # return the sin function!

    # total baseline of time window
    dt = np.nanmax(time) - np.nanmin(time)

    medflux = np.nanmedian(flux)
    # ti = time[dl[i]:dr[i]]

    for k in range(0, maxnum):
        # Use Jake Vanderplas faster version!
        pgram = LombScargleFast(fit_offset=False)
        pgram.optimizer.set(period_range=(minper,maxper))
        pgram = pgram.fit(time,
                          flux_out - medflux,
                          error)

        df = (1./minper - 1./maxper) / nper
        f0 = 1./maxper
        pwr = pgram.score_frequency_grid(f0, df, nper)

        freq = f0 + df * np.arange(nper)
        per = 1./freq

        pok = np.where((per < dt) & (per > minper))
        pk = per[pok][np.argmax(pwr[pok])]
        pp = np.max(pwr)

        if debug is True:
            print('trial (k): '+str(k)+'.  peak period (pk):'+str(pk)+
                  '.  peak power (pp):'+str(pp))

        # if a period w/ enough power is detected
        if (pp > plim):
            # fit sin curve to window and subtract
            if per2 is True:
                p0 = [pk, 3.0 * np.nanstd(flux_out-medflux), 0.0,
                      pk/2., 1.5 * np.nanstd(flux_out-medflux), 0.1, 0.0]
                try:
                    pfit, pcov = curve_fit(_sinfunc2, time, flux_out-medflux, p0=p0)
                    if debug is True:
                        print('>>', pfit)
                except RuntimeError:
                    pfit = [pk, 0., 0., 0., 0., 0., 0.]
                    if debug is True:
                        print('Curve_Fit2 no good')

                flux_out = flux_out - _sinfunc2(time, *pfit)
                sin_out = sin_out + _sinfunc2(time, *pfit)

            else:
                p0 = [pk, 3.0 * np.nanstd(flux_out-medflux), 0.0, 0.0]
                try:
                    pfit, pcov = curve_fit(_sinfunc, time, flux_out-medflux, p0=p0)
                except RuntimeError:
                    pfit = [pk, 0., 0., 0.]
                    if debug is True:
                        print('Curve_Fit no good')

                flux_out = flux_out - _sinfunc(time, *pfit)
                sin_out = sin_out + _sinfunc(time, *pfit)

        # add the median flux for this window BACK in
        sin_out = sin_out + medflux

    # if debug is True:
    #     plt.figure()
    #     plt.plot(time, flux)
    #     plt.plot(time, flux_out, c='red')
    #     plt.show()

    if returnmodel is True:
        return sin_out
    else:
        return flux_out


'''
def FitMedSin(time, flux, error, nper=20000,
              minper=0.1, maxper=30.0, plim=0.25,
              returnmodel=True ):

    flux_out = np.array(flux, copy=True)
    sin_out = np.zeros_like(flux) # return the sin function!

    # total baseline of time window
    dt = np.nanmax(time) - np.nanmin(time)

    medflux = np.nanmedian(flux)
    # ti = time[dl[i]:dr[i]]

    if np.nanmax(time) - np.nanmin(time) < maxper*2.:
        maxper = (np.nanmax(time) - np.nanmin(time))/2.

    
    # Use Jake Vanderplas supersmoother version
    pgram = SuperSmoother()
    pgram.optimizer.period_range=(minper,maxper)
    pgram = pgram.fit(time,
                      flux_out - medflux,
                      error)

    # Predict on a regular phase grid
    period = pgram.best_period

    phz = np.mod(time, period) / period
    ss = np.argsort(phz)
    magfit = np.zeros_like(phz)

    magfit[ss] = pgram.predict(phz[ss])

    if returnmodel is True:
        return magfit + medflux
    else:
        return flux - magfit
'''


def MultiBoxcar(time, flux, error, numpass=3, kernel=2.0,
                sigclip=5, pcentclip=5, returnindx=False,
                debug=False):
    '''
    Boxcar smoothing with multi-pass outlier rejection. Uses both errors
    and local scatter for rejection

    Parameters
    ----------
    time : numpy array
        assumes units are days
    flux : numpy array
    error : numpy array
    numpass : int, optional
        the number of passes to make over the data. (Default is 3)
    kernel : float, optional
        the boxcar size in hours. (Default is 2.0)
    sigclip : int, optional
        Number of times the standard deviation to clip points at
        (Default is 5)
    pcentclip : int, optional
        % to clip for outliers, i.e. 5= keep 5th-95th percentile
        (Default is 5)

    Returns
    -------
    The smoothed light curve model
    '''

    # flux_sm = np.array(flux, copy=True)
    # time_sm = np.array(time, copy=True)
    # error_sm = np.array(error, copy=True)
    #
    # for returnindx = True
    # indx_out = []

    # the data within each gap range
    time_i = time
    flux_i = flux
    error_i = error
    indx_i = np.arange(len(time)) # for tracking final indx used

    exptime = np.nanmedian(time_i[1:]-time_i[:-1])
    nptsmooth = int(kernel/24.0 / exptime)

    if (nptsmooth < 4):
        nptsmooth = 4

    if debug is True:
        print('# of smoothing points: '+str(nptsmooth))

    # now take N passes of rejection on it
    for k in range(0, numpass):
        # rolling median in this data span with the kernel size
        flux_i_sm = rolling_median(flux_i, nptsmooth, center=True)
        indx = np.isfinite(flux_i_sm)

        if (sum(indx) > 1):
            diff_k = (flux_i[indx] - flux_i_sm[indx])
            lims = np.nanpercentile(diff_k, (pcentclip, 100-pcentclip))

            # iteratively reject points
            # keep points within sigclip (for phot errors), or
            # within percentile clip (for scatter)
            ok = np.logical_or((np.abs(diff_k / error_i[indx]) < sigclip),
                               (lims[0] < diff_k) * (diff_k < lims[1]))

            if debug is True:
                print('k = '+str(k))
                print('number of accepted points: '+str(len(ok[0])))

            time_i = time_i[indx][ok]
            flux_i = flux_i[indx][ok]
            error_i = error_i[indx][ok]
            indx_i = indx_i[indx][ok]

    flux_sm = np.interp(time, time_i, flux_i)

    indx_out = indx_i

    if returnindx is False:
        return flux_sm
    else:
        return np.array(indx_out, dtype='int')


def IRLSSpline(time, flux, error, Q=400.0, ksep=0.07, numpass=5, order=3, debug=False):
    '''
    IRLS = Iterative Re-weight Least Squares

    Parameters
    ----------
    time
    flux
    error
    Q
    ksep
    numpass
    order

    Returns
    -------

    '''
    weight = 1. / (error**2.0)

    knots = np.arange(np.nanmin(time) + ksep, np.nanmax(time) - ksep, ksep)

    if debug is True:
        print('IRLSSpline: knots: ', np.shape(knots))
        print('IRLSSpline: time: ', np.shape(time), np.nanmin(time), time[0], np.nanmax(time), time[-1])
        print('IRLSSpline: <weight> = ', np.mean(weight))
        print(np.where((time[1:] - time[:-1] < 0))[0])
        print(flux)

        # plt.figure()
        # plt.errorbar(time, flux, error)
        # plt.scatter(knots, knots*0. + np.median(flux))
        # plt.show()

    for k in range(numpass):
        spl = LSQUnivariateSpline(time, flux, knots, k=order, check_finite=True, w=weight)
        # spl = UnivariateSpline(time, flux, w=weight, k=order, s=1)

        chisq = ((flux - spl(time))**2.) / (error**2.0)

        weight = Q / ((error**2.0) * (chisq + Q))

    return spl(time)



# def WaveletSmooth(time, flux, threshold=1, wavelet='db6', all=False):
#     '''
#     WORK IN PROGRESS - DO NOT USE
#
#     Generate a wavelet transform of the data, clip on some noise level,
#     then do inverse wavelet to generate model.
#
#     Requires uniformly sampled data!
#     If your data has gaps, watch out....
#     '''
#     _, dl, dr = FindGaps(time)
#
#     if all is True:
#         dl = [0]
#         dr = [len(time)]
#
#     model = np.zeros_like(flux)
#     # now loop over every chunk of data and fit wavelet
#     for i in range(0, len(dl)):
#
#         flux_i = flux[dl[i]:dr[i]]
#
#         # Do basic wavelet decontruct
#         WC = pywt.wavedec(flux_i, wavelet)
#
#         # now do thresholding
#         # got some tips here:
#         # https://blancosilva.wordpress.com/teaching/mathematical-imaging/denoising-wavelet-thresholding/
#         # and:
#         # https://pywavelets.readthedocs.org/en/latest/ref/idwt-inverse-discrete-wavelet-transform.html
#
#         NWC = map(lambda x: pywt.thresholding.hard(x,threshold * np.sqrt(2.*np.log(len(flux_i))) * np.std(flux_i)), WC)
#
#         model_i = pywt.waverec(NWC, wavelet)
#         # print(len(flux_i), len(model_i), len(model[dl[i]:dr[i]]))
#         #
#         # print(model_i[0:10])
#         # print(model_i[-10:])
#         if len(model_i) != len(model[dl[i]:dr[i]]):
#             print("length error on gap ",i)
#             print(len(flux_i), len(model_i), len(model[dl[i]:dr[i]]))
#             model_i = model_i[1:]
#
#         model[dl[i]:dr[i]] = model_i
#
#     # WC = pywt.wavedec(flux, wavelet)
#     # NWC = map(lambda x: pywt.thresholding.soft(x,threshold * np.sqrt(2.*np.log(len(flux))) * np.std(flux)), WC)
#     # model = pywt.waverec(NWC, wavelet)
#
#     print(len(model), len(time))
#     return model
#
#
# def Wavelet_Peaks(time, flux):
#     '''
#     Peak detection via continuous wavelets in scipy... doesnt work very well
#     '''
#     _, dl, dr = FindGaps(time)
#     indx = []
#     for i in range(0, len(dl)):
#         flux_i = flux[dl[i]:dr[i]]
#         time_i = time[dl[i]:dr[i]]
#         exptime = np.nanmedian(time_i[1:]-time_i[:-1])
#         if (exptime*24.*60. < 5):
#             widths = np.arange(1,100)
#         else:
#             widths = np.arange(1,10)
#
#         indx_i = signal.find_peaks_cwt(flux_i, widths)
#         indx = np.append(indx, indx_i)
#     return np.array(indx, dtype='int')
