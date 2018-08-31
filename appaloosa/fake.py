import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import wiener
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from aflare import aflare1
from helper import EquivDur

def chisq(data, error, model):
    '''
    Compute the normalized chi square statistic:
    chisq =  1 / N * SUM(i) ( (data(i) - model(i))/error(i) )^2
    '''
    return np.sum( ((data - model) / error)**2.0 ) / np.size(data)


def ed6890(bins, a):

    try:
        ed68_i = min(bins[a >= 0.68])
    except ValueError:
        ed68_i = -99

    try:
        ed90_i = min(bins[a >= 0.90])
    except ValueError:
        ed90_i = -99

    return ed68_i, ed90_i

def FlareStats(lc, istart=-1, istop=-1,
               c1=(-1,-1), c2=(-1,-1), cpoly=2, ReturnHeader=False):
    '''
    Compute properties of a flare event.

    Parameters
    ----------
    lc : lightcurve with time,flux,error and flux_model
    istart : int, optional
        The index in the input arrays (time,flux,error,model) that the
        flare starts at. If not used, defaults to the first data point.
    istop : int, optional
        The index in the input arrays (time,flux,error,model) that the
        flare ends at. If not used, defaults to the last data point.

    '''
    #Assumes flux is in relative flux units, i.e. rel_flux = (flux - median) / median
    med = lc.flux.median()
    lc.flux = (lc.flux - med) / med
    # if FLARE indicies are not stated by user, use start/stop of data
    if (istart < 0):
        istart = 0
    if (istop < 0):
        istop = len(lc.flux)-1

    # can't have flare start/stop at same point
    if (istart == istop):
        istop = istop + 1
        istart = istart - 1

    # need to have flare at least 3 datapoints long
    if (istop-istart < 2):
        istop = istop + 1

    tstart = lc.time.values[istart]
    tstop = lc.time.values[istop]
    dur0 = tstop - tstart

    # define continuum regions around the flare, same duration as
    # the flare, but spaced by half a duration on either side
    if (c1[0]==-1):
        t0 = tstart - dur0
        t1 = tstart - dur0/2.
        c1 = np.where((lc.time >= t0) & (lc.time <= t1))
    if (c2[0]==-1):
        t0 = tstop + dur0/2.
        t1 = tstop + dur0
        c2 = np.where((lc.time >= t0) & (lc.time <= t1))

    lct = lc.iloc[istart:istop+1]
    flareflux = lct.flux.values
    flaretime = lct.time.values
    modelflux = lct.flux_model.values
    flareerror = lct.error.values

    contindx = np.concatenate((c1[0], c2[0]))
    if (len(contindx) == 0):
        # if NO continuum regions are found, then just use 1st/last point of flare
        contindx = np.array([istart, istop])
        cpoly = 1
    contflux = lc.flux.values[contindx] # flux IN cont. regions
    conttime = lc.time.values[contindx]
    contfit = np.polyfit(conttime, contflux, cpoly)
    contline = np.polyval(contfit, flaretime) # poly fit to cont. regions

    medflux = np.nanmedian(lc.flux_model)

    # measure flare amplitude
    ampl = np.max(flareflux-contline) / medflux
    tpeak = flaretime[np.argmax(flareflux-contline)]

    p05 = np.where((flareflux-contline <= ampl*0.5))
    if len(p05[0]) == 0:
        fwhm = dur0 * 0.25
        # print('> warning') # % ;
    else:
        #print(p05)
        fwhm = np.max(flaretime[p05]) - np.min(flaretime[p05])

    # fit flare with single aflare model
    pguess = (tpeak, fwhm, ampl)
    # print(pguess) # % ;
    # print(len(flaretime)) # % ;

    try:
        popt1, pcov = curve_fit(aflare1, np.array(flaretime), (flareflux-contline) / medflux, p0=pguess)
    except ValueError:
        # tried to fit bad data, so just fill in with NaN's
        # shouldn't happen often
        popt1 = np.array([np.nan, np.nan, np.nan])
    except RuntimeError:
        # could not converge on a fit with aflare
        # fill with bad flag values
        popt1 = np.array([-99., -99., -99.])

    # flare_chisq = total( flareflux - modelflux)**2.  / total(error)**2
    flare_chisq = chisq(flareflux, flareerror, modelflux)

    # measure KS stats of flare versus model
    ks_d, ks_p = stats.ks_2samp(flareflux, modelflux)

    # measure KS stats of flare versus continuum regions
    ks_dc, ks_pc = stats.ks_2samp(flareflux-contline, contflux-np.polyval(contfit, conttime))

    # put flux in relative units, remove dependence on brightness of stars
    # rel_flux = (flux - flux_model) / np.median(flux_model)
    # rel_error = error / np.median(flux_model)

    # measure flare ED
    ed = EquivDur(np.array(flaretime), (flareflux-contline)/medflux)

    # output a dict or array?
    params = np.array((tstart, tstop, tpeak, ampl, fwhm, dur0,
                       popt1[0], popt1[1], popt1[2],
                       flare_chisq, ks_d, ks_p, ks_dc, ks_pc, ed), dtype='float')
    # the parameter names for later reference
    header = ['t_start', 't_stop', 't_peak', 'amplitude', 'FWHM', 'duration',
             't_peak_aflare1', 't_FWHM_aflare1', 'amplitude_aflare1',
             'flare_chisq', 'KS_d_model', 'KS_p_model', 'KS_d_cont', 'KS_p_cont', 'Equiv_Dur']

    if ReturnHeader is True:
        return header
    else:
        return params
def FakeFlaresDist(std, nfake, ampl=(5e-1,5e2), dur=(5e-1,2e2),
                   mode='hawley2014', scatter=False, debug=False):

    '''
    Creates different distributions of fake flares to be injected into light curves.

    rand: Flares are distibuted evenly in dur and ampl
    hawley2014: Flares are distributed in a strip around to a power law with exponent alpha, see fig. 10 in Hawley et al. 2014

    Parameters:
    -----------

    std: standard deviation of quiescent light curve
    nfake: number of fake flares to be created
    ampl: amplitude range in relative (only for 'rand' mode) flux units,
          default=(5e-1,5e3) for consistency with 'hawley2014'
    dur: duration range (only for 'rand' mode) in minutes,
         default=(5e-1,2e2) for consistency with 'hawley2014'
    mode: distribution of fake flares in (duration,amplitude) space,
          default='hawley2014'
    scatter: saves a scatter plot of the distribution for the injected sample,
             default='False'

    Returns:
    -------

    dur_fake: durations of generated fake flares in days
    ampl_fake: amplitudes of generated fake flares in relative flux units


    '''

    if mode=='rand':

        dur_fake =  (np.random.random(nfake) * (dur[1] - dur[0]) + dur[0])
        ampl_fake = (np.random.random(nfake) * (ampl[1] - ampl[0]) + ampl[0])*std
        lndur_fake = np.log10(dur_fake)
        lnampl_fake = np.log10(ampl_fake)
        dur_fake = dur_fake / 60. / 24.

        legend='Distribution random across ' + str(ampl) + ' rel. fl. u. and ' + str(dur) + ' min.'


    elif mode=='hawley2014':

        c_range=np.array([np.log10(5)-6.,np.log10(5)-4.]) #estimated from fig. 10 in Hawley et al. 2014
        alpha=2.                                        #estimated from fig. 10 in Hawley et al. 2014
        ampl=(np.log10(2.*std),np.log10(10000.*std))

        lnampl_fake = (np.random.random(nfake) * (ampl[1] - ampl[0]) + ampl[0])
        lndur_fake=np.zeros(nfake, dtype='float')
        rand=np.random.random(nfake)
        dur_max = (1./alpha) * (lnampl_fake-c_range[0]) #log(tmax)
        dur_min = (1./alpha) * (lnampl_fake-c_range[1]) # log(tmin)

        for a in range(0,nfake):
            lndur_fake[a]= (rand[a] * (dur_max[a] - dur_min[a]) + dur_min[a])

        ampl_fake = np.power(np.full(nfake,10), lnampl_fake)
        dur_fake=np.power(np.full(nfake,10), lndur_fake) / 60. / 24.

        legend='Distribution consistent with Hawley et al. (2014) observations.'

    if debug == True:
        print('Fake flares durations (1,2) and amplitudes (3,4):')
        print(min(lndur_fake),max(lndur_fake),min(lnampl_fake),max(lnampl_fake))

    if scatter == True:

        fig, ax = plt.subplots()
        ax.scatter(lndur_fake[:-1], lnampl_fake[:-1])
        ax.set_xlabel(r'log duration (in min)', fontsize=15)
        ax.set_ylabel(r'log amplitude (in rel. flux units)', fontsize=15)
        ax.set_title('Fake flares generated by Appaloosa\n'+legend, fontsize=12)
        ax.grid(color='0.5', linestyle='-', linewidth=1)
        #ax.axis([0.,2.5,-3.5,-0.5])
        plt.show()
        plt.close()
        #plt.savefig(mode+'_fake_scatter.pdf')


    return dur_fake, ampl_fake

def FakeCompleteness(dffake,nfake,iterations,display=False,file=''):
    '''
    Construct a completeness curve for the fake injections.
    Parameters:
    -------------
    dffake
    nfake
    iterations
    display =False
    file =''

    Returns:
    --------------
    ed68
    ed90

    '''
    nbins = nfake * iterations // 20
    if nbins < 10:
        print('Warning: Few injections, completeness of recovery unclear.\nTry increasing iterations.')
        return -199,-199
    bins = np.linspace(0, dffake.ed_fake.max() + 1, nbins)
    binmids = np.concatenate(([0],(bins[1:]+bins[:-1])/2))
    frac_recovered = dffake.rec_fake.groupby(np.digitize(dffake.ed_rec, bins)).mean()
    frac_recovered.iloc[0] = 0 #add a zero intercept for aesthetics
    frac_recovered.sort_index(inplace=True) #helps plotting
    binmids = np.concatenate(([0],(bins[1:]+bins[:-1])/2)) #add a zero intercept for aesthetics

    try:
        f = pd.DataFrame({'ed_bins': binmids[frac_recovered.index.values[:-1]],
                           'frac_recovered': frac_recovered.iloc[:-1],
                           'frac_rec_sm': wiener(frac_recovered.iloc[:-1],3)})
        ed68, ed90 = ed6890(f.ed_bins,f.frac_rec_sm)
    except (IndexError, ValueError):
        print("Something went wrong with dffake indexing. Try using even number of iterations.")
        return -199, -199
    # use frac_rec_sm completeness curve to estimate 68%/90% complete

    if display is True:
        fig, (ax1,ax2) = plt.subplots(ncols=2, nrows=1,figsize=(8,5))
        # look at the completeness curve
        ax1.plot(f.ed_bins, f.frac_recovered, c='k')
        ax1.vlines([ed68, ed90], ymin=0, ymax=1, colors='b',alpha=0.75, lw=5)
        ax1.set_xlabel('Flare Equivalent Duration (seconds)')
        ax1.set_ylabel('Fraction of Recovered Flares')
        ax1.set_xscale('log')
        ax1.set_title('Artificial Flare injection, N = {}'.format(nfake*iterations))
        ax1.set_xlim(0,)

        #histogram of fake flares
        ax2.hist(dffake.ed_fake[dffake.rec_fake==1], color='r',
                bins=bins, label='recovered', histtype='step')
        ax2.hist(dffake.ed_fake[dffake.rec_fake==0], color='b',
                bins=bins, label='not recovered', histtype='step')
        ax2.set_xlabel('Equiv Dur of simulated flares')
        ax2.set_yscale('log')

        ax2.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('{}_fake_recovered.pdf'.format(file),dpi=300, bbox_inches='tight', pad_inches=0.5)

        plt.close()
    return ed68, ed90
