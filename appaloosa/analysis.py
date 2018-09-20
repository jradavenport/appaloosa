'''
Code to make the analysis and figures used in the paper:
"The Kepler Catalog of Stellar Flares", J. R. A. Davenport (2016)
'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib
import os
import sys
import appaloosa
import pandas as pd
import datetime
import warnings
from scipy.optimize import curve_fit, minimize
from astropy.stats import funcs
import emcee
import corner
# from scipy.stats import binned_statistic_2d
# from os.path import expanduser

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})


def _ABmag2flux(mag, zeropt=48.60,
                wave0=6400.0, fwhm=4000.0):
    '''
    Replicate the IDL procedure:
    http://idlastro.gsfc.nasa.gov/ftp/pro/astro/mag2flux.pro
    flux = 10**(-0.4*(mag +2.406 + 4*np.log10(wave0)))

    Parameters set for Kepler band specifically
    e.g. see http://stev.oapd.inaf.it/~lgirardi/cmd_2.7/photsys.html
    '''

    c = 2.99792458e18 # speed of light, in [A/s]

    # standard equation from Oke & Gunn (1883)
    # has units: [erg/s/cm2/Hz]
    f_nu = 10.0 ** ( (mag + zeropt) / (-2.5) )

    # has units of [erg/s/cm2/A]
    f_lambda = f_nu * c / (wave0**2.0)

    # Finally: units of [erg/s/cm2]
    flux = f_lambda * fwhm
    # now all we'll need downstream is the distance to get L [erg/s]

    return flux


def _tau(mass):
    '''
    Write up the Eqn 11 from Wright (2011) that gives the
    convective turnover timescale, used in Rossby number calculation

    (Ro = Prot / tau)

    Parameters
    ----------
    mass : float
        relative to solar

    Returns
    -------
    tau (in days)

    '''

    log_tau = 1.16 - 1.49 * np.log10(mass) - 0.54 * np.log10(mass)**2.
    return  10.**log_tau


def RoFlare(r,a,b,s):
    '''
    The piecewise function that has a saturated and decaying regime

    Parameters
    ----------
    r : the log Ro value
    a : the amplitude
    b : the break Ro
    s : the slope

    Returns
    -------

    '''
    f = np.piecewise(r, [(r <= b), (r > b)],
                     [a, # before the break, it is flat
                      lambda x: (s * (x-b) + a)])

    return f


def _Perror(n, full=False, down=False):
    '''
    Calculate the asymmetric Poisson error, using Eqn 7
    and Eqn 12 in Gehrels 1986 ApJ, 3030, 336

    Parameters
    ----------
    n
    full

    Returns
    -------

    '''

    err_up = err_dn = np.sqrt(n + 0.75) + 1.0 # this is the default behavior for N=0

    xn = np.where((n > 0))[0]
    if np.size(xn) > 0:
        err_dn[xn] = np.abs(n[xn] * (1.-1./(9. * n[xn])-1./(3.*np.sqrt(n[xn])))**3.-n[xn])
        err_up[xn] = n[xn] + np.sqrt(n[xn] + 0.75) + 1.0 - n[xn]
    # else:
    #     err_up = np.sqrt(n + 0.75) + 1.0
    #     err_dn = err_up
    #     # err_up = err_dn = np.nan

    if full is True:
        return err_dn, err_up
    else:
        if down is True:
            return err_dn
        else:
            return err_up


def _DistModulus(m_app, M_abs):
    '''
    Trivial wrapper to invert the classic equation:
    m - M = 5 log(d) - 5

    Parameters
    ----------
    m_app
        apparent magnitude
    M_abs
        absolute magnitude

    Returns
    -------
    distance, in pc
    '''
    mu = m_app - M_abs
    dist = 10.0**(mu/5.0 + 1.0)
    return dist


def _linfunc(x, m, b):
    '''
    A simple linear function to fit with curve_fit
    '''
    return m * x + b


def _plaw(x, m, b):
    '''
    a powerlaw function
    '''
    x2 = 10.**x
    return b * (x2**m)


def Angus2015(B_V, age):
    '''
    Compute the rotation period expected for a star of a given color (temp) and age

    NOTE: - input Age is in MYr
          - output Period is in days

    Eqn 15 from Angus+2015
    http://adsabs.harvard.edu/abs/2015MNRAS.450.1787A

    '''
    P = (age ** 0.55) * 0.4 * ((B_V - 0.45) ** 0.31)

    return P


def Angus2015_age(B_V, P):
    '''
    Compute the rotation period expected for a star of a given color (temp) and age

    NOTE: - output Age is in MYr
          - input Period is in days

    Eqn 15 from Angus+2015
    http://adsabs.harvard.edu/abs/2015MNRAS.450.1787A

    '''
    # P = (age ** 0.55) * 0.4 * ((B_V - 0.45) ** 0.31)
    age = np.power(P / (0.4 * ((B_V - 0.45) ** 0.31)), 1. / 0.55)
    return age


def MH2008(B_V, age):
    '''
    Equations 12,13,14 from Mamajek & Hillenbrand (2008)
    http://adsabs.harvard.edu/abs/2008ApJ...687.1264M

    Coefficients from Table 10
    
    Parameters
    ----------
    B_V (B-V) color
    age in Myr

    Returns
    -------
    period in color

    '''
    a = 0.407
    b = 0.325
    c = 0.495
    n = 0.566

    f = a * np.power(B_V - c, b)
    g = np.power(age, n)

    P = f * g

    return P


def MH2008_age(B_V, P):
    '''
    inverse of other function. Input color and P, output age

    '''
    a = 0.407
    b = 0.325
    c = 0.495
    n = 0.566

    f = a * np.power(B_V - c, b)
    # g = np.power(age, n)
    # P = f * g
    age = np.power(P / f, 1. / n)

    return age


def getBV(mass, isochrone='1.0gyr.dat'):
    try:
        __file__
    except NameError:
        __file__ = os.getenv("HOME") +  '/python/appaloosa/appaloosa/analysis.py'

    dir = os.path.dirname(os.path.realpath(__file__)) + '/../misc/'
    file = dir + isochrone

    df = pd.read_table(file, delim_whitespace=True, comment='#',
                       names=('Z', 'log_age', 'M_ini', 'M_act', 'logL/Lo', 'logTe', 'logG',
                              'mbol', 'Kepler', 'g', 'r', 'i', 'z', 'DDO51_finf','int_IMF',
                              'stage', 'J', 'H', 'Ks', 'U', 'B', 'V', 'R', 'I'))

    mass_iso = df['M_ini'].values
    ss = np.argsort(mass_iso)  # needs to be sorted for interpolation

    BV_iso = df['B'].values - df['V'].values
    BV = np.interp((mass), mass_iso[ss], BV_iso[ss])

    return BV


def FlareEqn0(X, a1, a2, b1, b2):
    '''
    this is the simple FFD evolution for bins of mass,
    i.e. only produce FFD model as a function of energy and age (drop mass dependence)

    run on each of the (g-i) sample bins shown in paper

    Parameters
    ----------
    X = (logE, logt)
        age in log Myr
        E in log erg

    Returns
    -------
    log Rate of flares

    '''
    logE, logt = X

    a = a1 * logt + a2
    b = b1 * logt + b2
    logR = logE * a + b

    return logR


def FlareEqn(X, a1, a2, a3, b1, b2, b3):
    '''
    The big FFD fitting equation, fititng both powerlaw slope and intercept as functions of mass and age

    THIS is the original version from Paper2 draft v1

    Parameters
    ----------
    X = (logE, logt, m)
        age in log Myr
        mass in Solar
        E in log erg

    Returns
    -------
    log Rate of flares

    '''
    logE, logt, m = X

    a = a1 * logt + a2 * m + a3
    b = b1 * logt + b2 * m + b3
    logR = logE * a + b

    return logR

def FlareEqnNew(X, a1, a2, a3, b1, b2, b3):
    '''
    The big FFD fitting equation, fititng ONLY the powerlaw intercept as functions of mass and age
    The powerlaw slope is fixed to a=-1

    warning: currently requires a1,a2,a3 for backwards compatibility with the "FlareEqn" function above...
    Parameters
    ----------
    X = (logE, logt, m)
        age in log Myr
        mass in Solar
        E in log erg

    Returns
    -------
    log Rate of flares

    '''
    logE, logt, m = X

    a = -1.
    b = b1 * logt + b2 * m + b3
    logR = logE * a + b

    return logR


def flare_lnprob(p, x, y, yerr):
    N = np.size(x)
    model = FlareEqn(x, *p)
    return -0.5 * appaloosa.chisq(y, yerr, model)


def FlareEqn_nolog(X, a1, a2, a3, b1, b2, b3):
    '''

    Parameters
    ----------
    X = (logE, logt, m)
        age in log Myr
        mass in Solar
        E in log erg

    Returns
    -------
    Rate of flares (NOTE: not the log rate)

    '''
    logE, logt, m = X

    a = a1 * logt + a2 * m + a3
    b = b1 * logt + b2 * m + b3
    logR = logE * a + b

    return 10.0 ** logR


def FlareEqn2(X, a1, a2, a3, a4, a5, a6, b1, b2):
    '''

    Parameters
    ----------
    X = (logE, logt, m)
        age in log Myr
        mass in Solar
        E in log erg

    Returns
    -------
    log Rate of flares

    '''
    logE, logt, m = X

    a = (a1 * logt) + (a2 * m) + (a3 * logt * m) + (a4 * logt**2) + (a5 * m**2) + a6
    b = b1 * m + b2
    logR = logE * a + b

    return logR


def FlareEqn2_nolog(X, a1, a2, a3, a4, a5, a6, b1, b2):
    '''

    Parameters
    ----------
    X = (logE, logt, m)
        age in log Myr
        mass in Solar
        E in log erg

    Returns
    -------
    log Rate of flares

    '''
    logE, logt, m = X

    a = (a1 * logt) + (a2 * m) + (a3 * logt * m) + (a4 * logt**2) + (a5 * m**2) + a6
    b = b1 * m + b2
    logR = logE * a + b

    return 10.0 ** logR



def Chi_fl(giclr):
    '''
    Compute the Chi_fl parameter, defined as Flux(Kepler band) / Flux (Bol)

    Used to convert L_fl/L_kp to L_fl/L_bol

    NOTE: only defined between 0 <= g-i <= 5,
    or approximately 1.5 >= M_sun >= 0.15

    Parameters
    ----------
    giclr: float or numpy float array of the g-i stellar color

    Returns
    -------
    Chi_fl values

    '''
    fit = np.array([-0.00129193,  0.02105752, -0.14589187,  0.10493256,  0.00440871])

    return 10.0**np.polyval(fit, giclr)


def massL(m1=0.2, m2=1.3, dm=0.01, isochrone='1.0gyr.dat'):
    try:
        __file__
    except NameError:
        __file__ = os.getenv("HOME") +  '/python/appaloosa/appaloosa/analysis.py'

    dir = os.path.dirname(os.path.realpath(__file__)) + '/../misc/'
    file = dir + isochrone

    df = pd.read_table(file, delim_whitespace=True, comment='#',
                       names=('Z', 'log_age', 'M_ini', 'M_act', 'logL/Lo', 'logTe', 'logG',
                              'mbol', 'Kepler', 'g', 'r', 'i', 'z', 'DDO51_finf','int_IMF',
                              'stage', 'J', 'H', 'Ks', 'U', 'B', 'V', 'R', 'I'))

    masses = np.arange(m1, m2, dm)

    mass_iso = df['M_ini'].values
    ss = np.argsort(mass_iso)  # needs to be sorted for interpolation

    Mkp_iso = df['Kepler'].values
    # BV = np.interp((mass), mass_iso[ss], BV_iso[ss])

    pc2cm = 3.08568025e18
    F_kp = _ABmag2flux(Mkp_iso)
    L_kp = np.array(F_kp * (4.0 * np.pi * (10. * pc2cm)**2.0), dtype='float')

    logLs = np.interp(masses, mass_iso[ss], np.log10(L_kp[ss]))

    return masses, logLs






def energies(gmag, kmag, isochrone='1.0gyr.dat', return_all=False):
    '''
    Compute the quiescent energy for every star. Use the KIC (g-i) color,
    with an isochrone, get the absolute Kepler mag for each star, and thus
    the distance & luminosity.

    Isochrone is a 1.0 Gyr track from the Padova CMD v2.7
    http://stev.oapd.inaf.it/cgi-bin/cmd_2.7

    Kepler and Sloan phot system both in AB mags.

    Returns
    -------
    Quiescent Luminosities in the Kepler band
    '''

    # read in Padova isochrone file
    # note, I've cheated and clipped this isochrone to only have the
    # Main Sequence, up to the blue Turn-Off limit.

    try:
        __file__
    except NameError:
        __file__ = os.getenv("HOME") +  '/python/appaloosa/appaloosa/analysis.py'

    dir = os.path.dirname(os.path.realpath(__file__)) + '/../misc/'

    '''
    Mkp, Mg, Mr, Mi = np.loadtxt(dir + isochrone, comments='#',
                                 unpack=True, usecols=(8,9,10,11))

    # To match observed data to the isochrone, cheat:
    # e.g. Find interpolated g, given g-i. Same for Kp

    # do this 3 times, each color combo. Average result for M_kp
    Mgi = (Mg-Mi)
    ss = np.argsort(Mgi) # needs to be sorted for interpolation
    Mkp_go = np.interp((gmag-imag), Mgi[ss], Mkp[ss])
    Mg_o = np.interp((gmag-imag), Mgi[ss], Mg[ss])

    Mgr = (Mg-Mr)
    ss = np.argsort(Mgr)
    Mkp_ro = np.interp((gmag-rmag), Mgr[ss], Mkp[ss])
    Mr_o = np.interp((gmag-rmag), Mgr[ss], Mr[ss])

    Mri = (Mr-Mi)
    ss = np.argsort(Mri)
    Mkp_io = np.interp((rmag-imag), Mri[ss], Mkp[ss])
    Mi_o = np.interp((rmag-imag), Mri[ss], Mi[ss])

    Mkp_o = (Mkp_go + Mkp_ro + Mkp_io) / 3.0

    dist_g = np.array(_DistModulus(gmag, Mg_o), dtype='float')
    dist_r = np.array(_DistModulus(rmag, Mr_o), dtype='float')
    dist_i = np.array(_DistModulus(imag, Mi_o), dtype='float')
    dist = (dist_g + dist_r + dist_i) / 3.0

    dm_g = (gmag - Mg_o)
    dm_r = (rmag - Mr_o)
    dm_i = (imag - Mi_o)
    dm = (dm_g + dm_r + dm_i) / 3.0
    '''


    mass, Mkp, Mg, Mk = np.loadtxt(dir + isochrone, comments='#',
                                   unpack=True, usecols=(2,8,9,18))

    Mgk = (Mg-Mk)
    ss = np.argsort(Mgk) # needs to be sorted for interpolation
    Mkp_o = np.interp((gmag-kmag), Mgk[ss], Mkp[ss])
    Mk_o = np.interp((gmag-kmag), Mgk[ss], Mk[ss])
    mass_o = np.interp((gmag-kmag), Mgk[ss], mass[ss])

    dist = np.array(_DistModulus(kmag, Mk_o), dtype='float')
    dm = (kmag - Mk_o)

    pc2cm = 3.08568025e18

    # returns Flux [erg/s/cm^2]
    F_kp = _ABmag2flux(Mkp_o + dm)

    # again, classic bread/butter right here,
    # change Flux to Luminosity [erg/s]
    L_kp = np.array(F_kp * (4.0 * np.pi * (dist * pc2cm)**2.0), dtype='float')

    # !! Should be able to include errors on (g-i), propogate to
    #    errors on Distance, and thus lower error limit on L_kp !!

    # !! Should have some confidence about the interpolation,
    #    e.g. if beyond g-i isochrone range !!

    if return_all is True:
        return np.log10(L_kp), dist, np.array(mass_o, dtype='float')
    else:
        return np.log10(L_kp)


def paper2_plots(condorfile='condorout.dat.gz', debug=False,
                 kicfile='kic.txt.gz', statsfile='stats.txt',
                 figdir='figures2/', figtype='.pdf', rerun=False, oldplot=True):
    '''
    Paper 2: flares vs ages

    Run on WWU workstation in dir: ~/research/kepler-flares/
    '''

    # if doing the re-run (make FFD for all objects) then do all the old extra plots too
    if rerun:
        oldplot = True

    # read in KIC file
    # http://archive.stsci.edu/pub/kepler/catalogs/ <- data source
    # http://archive.stsci.edu/kepler/kic10/help/quickcol.html <- info
    print('RUNNING PAPER2_PLOTS')

    print('reading in ',datetime.datetime.now())
    kicdata = pd.read_csv(kicfile, delimiter='|')

    fdata = pd.read_table(condorfile, delimiter=',', skiprows=1, header=None)
    ''' KICnumber, lsflag (0=llc,1=slc), dur [days], log(ed68), tot Nflares, sum ED, sum ED err, [ Flares/Day (logEDbin) ] '''

    # need KICnumber, Flare Freq data in units of ED
    kicnum_c = fdata.iloc[:,0].unique()
    # num_fl_tot = fdata.groupby([0])[4].sum()

    bigdata = kicdata[kicdata['kic_kepler_id'].isin(kicnum_c)]

    # compute the distances and luminosities of all stars
    Lkp_uniq, dist_uniq, mass_uniq = energies(bigdata['kic_gmag'],
                                              bigdata['kic_kmag'],
                                              return_all=True)

    ## ingest Amy McQuillans rotation period catalog
    rotfile = 'comparison_datasets/Table_Periodic.txt'
    rotdata = pd.read_table(rotfile, delimiter=',', comment='#', header=None,
                            names=('KID','Teff','logg','Mass','Prot','Prot_err','Rper','LPH','w','DC','Flag'))


    # For every star: compute flare rate at some arbitrary energy
    Epoint = 35
    EpointS = str(Epoint)

    # set the limit on numbers of flares per star required
    Nflare_limit = 100 # 100 total candidates
    Nflare68_cut = 10 # 10 above 68% cut


    ##########      READ DATA FROM THE BIG BAD LOOP      ##########
    ap_loop_file = 'ap_analysis_loop.npz'
    # pull arrays back in via load!
    npz = np.load(ap_loop_file)
    Nflare = npz['Nflare']
    rate_E = npz['rate_E']
    fit_E = npz['fit_E']
    fit_Eerr = npz['fit_Eerr']
    ffd_ab = npz['ffd_ab']
    gr_all = npz['gr_all']
    gi_all = npz['gi_all']
    meanE = npz['meanE']
    maxE = npz['maxE']
    Prot_all = npz['Prot_all']
    dur_all = npz['dur_all']
    ED_all = npz['ED_all']
    ED_all_err = npz['ED_all_err']
    logg_all = npz['logg_all']
    Nflare68 = npz['Nflare68']
    ra = npz['ra']
    dec = npz['dec']
    mass = npz['mass']
    Lkp_all = npz['Lkp_all']
    print('data restored ', datetime.datetime.now())


    # total fractional energy (in seconds) / total duration (in seconds)
    Lfl_Lbol = ED_all / (dur_all * 60. * 60. * 24.)
    Lfl_Lbol_err = ED_all_err / (dur_all * 60. * 60. * 24.)

    tau_all = _tau(mass)
    Rossby = Prot_all / tau_all

    dist_all = np.zeros_like(tau_all) - 1
    for l in range(np.size(tau_all)):
        mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[l]))
        if len(mtch[0]) > 0:
            dist_all[l] = dist_uniq[mtch][0]

    # for Riley outputfile including masses
    if False:
        dfout = pd.DataFrame(data={'kicnum': kicnum_c,
                                   'giclr': gi_all,
                                   'mass': mass,
                                   'Prot': Prot_all,
                                   'tau':tau_all,
                                   'Rossby':Rossby,
                                   'LflLkep': Lfl_Lbol,
                                   'LflLkep_err': Lfl_Lbol_err,
                                   'Nflares': Nflare,
                                   'Nflare68': Nflare68,
                                   'R35': rate_E,
                                   'dist':dist_all
                                   })

        dfout.to_csv('kic_lflare_mass_dist.csv')


    # plots vs R35
    clr = np.log10(fit_E)
    clr_raw = clr

    # stars that have enough flares and have valid rates
    okclr = np.where((Nflare68 >= Nflare68_cut) &
                     np.isfinite(np.log10(fit_E)) &
                     (Nflare >= Nflare_limit))

    # stars that just have valid rates
    # okclr0 = np.where(  # (clr >= clr_rng[0]) & (clr <= clr_rng[1]) &
    #     np.isfinite(clr) & (Nflare >= 0))


    # first, a basic plot of flare rate versus color

    # clr_rng = [np.nanmin(clr), np.nanmax(clr)]
    # rate_range = [[-1, 3], [-20,4]]



    # g-i color range bins
    crng = np.array([[0.5, 0.75],
                     [0.75, 1.0],
                     [1., 1.5],
                     [1.5, 2.],
                     [2., 2.5],
                     [2.5, 3.]])
                     # [0.0, 0.5]]) # a bin I don't expect to understand. Should be F stars

    crng_clrs = np.array(['#253494', '#2c7fb8', '#7fcdbb',
                          '#fd8d3c', '#f03b20', '#bd0026'])

    # the same ED bins used in paper1_plots
    edbins = np.arange(-5, 5, 0.2)
    edbins = np.append(-10, edbins)
    edbins = np.append(edbins, 10)

    if oldplot:
        for k in range(crng.shape[0]):
            ts = np.where((gi_all[okclr] >= crng[k, 0]) &
                          (gi_all[okclr] <= crng[k, 1]) &
                          (Prot_all[okclr] >= 0.1) &
                          (kicnum_c[okclr] != 10924462) &  # manually throw out bad FFDs
                          (kicnum_c[okclr] != 3864443) &
                          (kicnum_c[okclr] != 5559631) &
                          (kicnum_c[okclr] != 7988343) &
                          (kicnum_c[okclr] != 9591503) &
                          (kicnum_c[okclr] != 3240305)
                          )

            # ff.write('# that pass TS color cut: ' + str(len(ts[0])) + '\n')

            plt.figure(figsize=(6,5))
            plt.scatter(Prot_all[okclr][ts], clr_raw[okclr][ts], s=50, alpha=1, lw=0.5, c='k')
            # plt.errorbar(Prot_all[okclr][ts], clr_raw[okclr][ts], yerr=clr_raw_err[okclr][ts], fmt='k,')
            plt.xlabel('P$_{rot}$ (days)')
            plt.ylabel('log R$_{' + EpointS + '}$ (#/day)')
            plt.title(str(crng[k, 0]) + ' < (g-i) < ' + str(crng[k, 1]))
            plt.xscale('log')
            # plt.ylim(-4, 0)
            plt.xlim(0.1, 100)
            plt.savefig(figdir + 'rot_rate' + str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()

            plt.figure(figsize=(6,5))
            plt.scatter(Rossby[okclr][ts], clr_raw[okclr][ts], s=50, alpha=1, lw=0.5, c='k')
            # plt.plot(10. ** np.arange(-3, 1, .01), RoFlare(np.arange(-3, 1, .01), *popt1), c='red', lw=3, alpha=0.75)
            plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
            plt.ylabel('log R$_{' + EpointS + '}$ (#/day)')
            plt.title(str(crng[k, 0]) + ' < (g-i) < ' + str(crng[k, 1]) + ', N=' + str(len(ts[0])))
            plt.xscale('log')
            plt.xlim(0.8e-2, 4e0)
            # plt.ylim(-5, -1.5)
            plt.savefig(figdir + 'Rossby_rate' + str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()



            '''
            ^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*&^&*^&*^&*^&*
            
            For all stars in this (g-i) color bin, pull light curves, make a super-FFD
            
            using methods from the "big bad loop" in paper1_plots
            
            ^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*^&*&^&*^&*^&*^&*
            '''


            plt.figure(figsize=(6,5))
            color = iter(cm.Spectral(np.linspace(0,1,np.size(ts))))

            ts = ts[0][np.argsort(Prot_all[okclr][ts])]

            B_V_ts = getBV(mass[okclr][ts])
            age_ts = MH2008_age(B_V_ts, Prot_all[okclr][ts])

            logE_tsstack = np.array([])
            logt_tsstack = np.array([])
            logR_tsstack = np.array([])
            logRerr_tsstack = np.array([])

            offset68 = 0. # defined way below when I do this again....

            for l in range(np.size(ts)):
                colornext = next(color)

                # find all entires for this star (LLC and SLC data)
                star = np.where((fdata[0].values == kicnum_c[okclr][ts][l]))[0]
                # arrays for FFD
                fnorm = np.zeros_like(edbins[1:], dtype='float')
                fsum = np.zeros_like(edbins[1:], dtype='float')

                # find this star in the KIC data
                mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[okclr][ts][l]))
                Lkp_i = Lkp_uniq[mtch][0]
                # tmp array to hold the total number of flares in each FFD bin
                flare_tot = np.zeros_like(fnorm, dtype='float')

                for i in range(0, len(star)):
                    # Find the portion of the FFD that is above the 68% cutoff
                    ok = np.where((edbins[1:] >= (fdata.loc[star[i], 3]-offset68)))[0]
                    if len(ok) > 0:
                        # add the rates together for a straight mean
                        fsum[ok] = fsum[ok] + np.array(fdata.loc[star[i], 7:].values[ok])

                        # count the number for the straight mean
                        fnorm[ok] = fnorm[ok] + 1.0

                        # add the actual number of flares for this data portion: rate * duration
                        flare_tot[ok] = flare_tot[ok] + (fdata.loc[star[i], 7:].values[ok] * np.float(fdata.loc[star[i], 2]))
                        # flare_tot = flare_tot + (fdata.loc[star[i], 7:].values * np.float(fdata.loc[star[i], 2]))

                        # print('')
                        # print('inner loop ',i,ok)
                        # print(fsum)
                        # print(fnorm)

                # the important arrays for the averaged FFD
                ffd_x = edbins[1:][::-1] + Lkp_i
                ffd_y = np.cumsum(fsum[::-1] / fnorm[::-1])

                # print('ffd params:', ffd_x, ffd_y)

                # the "error" is the Poisson err from the total # flares per bin
                ffd_yerr = _Perror(flare_tot[::-1], down=True) / dur_all[okclr][ts][l]

                # find where in the FFD there are at least 1 valid flares
                ffd_ok = np.where((ffd_y > 0) & np.isfinite(ffd_y) & (fsum[::-1] > 0) &
                                  np.isfinite(ffd_x) & np.isfinite(ffd_yerr) &
                                  (ffd_x < 38.5)) # fix one of the outlier problems

                # if there are any valid bins, find the max energy (bin)
                if len(ffd_ok[0]) > 0:
                    maxE[k] = np.nanmax(ffd_x[ffd_ok])

                # if there are at least 2 energy bins w/ valid flares...
                if len(ffd_ok[0]) > 1:
                    # compute the mean flare energy (bin) for this star
                    meanE = np.append(meanE, np.nanmedian(ffd_x[ffd_ok]))

                    # p0 = [-0.5, np.log10(np.nanmax(ffd_y[ffd_ok]))]
                    # fit, cov = curve_fit(_linfunc, ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), p0=p0,
                    #                      sigma=np.abs(ffd_yerr[ffd_ok] / (ffd_y[ffd_ok] * np.log(10))))

                    plt.plot(ffd_x[ffd_ok], ffd_y[ffd_ok], linewidth=1.5, alpha=0.7, c=colornext)


                    # -- turn off these annotations
                    # plt.annotate(str(Prot_all[okclr][ts][l]), (ffd_x[ffd_ok][0], ffd_y[ffd_ok][0]),
                    #              textcoords='data', size=10, color=colornext)

                    ## stack the FFD propeties, to do a joint FFD evolution fit for this (g-i) bin
                    logE_tsstack = np.append(logE_tsstack, ffd_x[ffd_ok])
                    logt_tsstack = np.append(logt_tsstack, np.ones_like(ffd_x[ffd_ok]) * np.log10(age_ts[l]))
                    logR_tsstack = np.append(logR_tsstack, np.log10(ffd_y[ffd_ok]))
                    yerr_temp = np.abs(ffd_yerr[ffd_ok] / (ffd_y[ffd_ok] * np.log(10.)))
                    yerr_temp = np.sqrt(yerr_temp**2. + np.nanmedian(yerr_temp)**2.)
                    logRerr_tsstack = np.append(logRerr_tsstack, yerr_temp)


            plt.yscale('log')
            plt.xlabel('log Flare Energy (erg)')
            plt.ylabel('Cumulative Flare Freq (#/day)')
            plt.xlim(32,38)
            plt.ylim(1e-4, 1e0)
            plt.title(str(crng[k, 0]) + ' < (g-i) < ' + str(crng[k, 1]) + ', N=' + str(len(ts)))
            plt.savefig(figdir + 'mean_ffd' + str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()


            p0ts = (0., 0., -1., 10.)
            stackOKts = np.where(np.isfinite(logE_tsstack) & np.isfinite(logR_tsstack) &
                               np.isfinite(logt_tsstack) & np.isfinite(logRerr_tsstack))
            Xts = (logE_tsstack[stackOKts], logt_tsstack[stackOKts])

            fitts, covts = curve_fit(FlareEqn0, Xts, logR_tsstack[stackOKts], p0ts,
                                     sigma=logRerr_tsstack[stackOKts])

            print(str(crng[k, 0]) + " <g-i< " + str(crng[k, 1]))
            print('FFD model: ')
            print(fitts)

            plt.figure()
            y1 = FlareEqn0((np.arange(33, 37), np.array([1, 1, 1, 1])), *fitts)
            y2 = FlareEqn0((np.arange(33, 37), np.array([2, 2, 2, 2])), *fitts)
            y3 = FlareEqn0((np.arange(33, 37), np.array([3, 3, 3, 3])), *fitts)
            y4 = FlareEqn0((np.arange(33, 37), np.array([4, 4, 4, 4])), *fitts)

            plt.plot(np.arange(33, 37), 10. ** y1, c='CornflowerBlue')
            plt.plot(np.arange(33, 37), 10. ** y2, c='DarkViolet')
            plt.plot(np.arange(33, 37), 10. ** y3, c='FireBrick')
            plt.plot(np.arange(33, 37), 10. ** y4, c='Red')
            plt.yscale('log')
            plt.xlabel('log Flare Energy (erg)')
            plt.ylabel('Cumulative Flare Freq (#/day)')
            plt.title(str(crng[k, 0]) + ' < (g-i) < ' + str(crng[k, 1]))
            plt.savefig(figdir + 'eqnFFD_' + str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()


        # Rossby figure w/ points colored by mass
        pok = np.where((Prot_all[okclr] > 0.1) &
                       (gi_all[okclr] > 0.5) & # manually throw out the bluest stars
                       (kicnum_c[okclr] != 10924462) &  # manually throw out bad FFDs
                       (kicnum_c[okclr] != 3864443) &
                       (kicnum_c[okclr] != 5559631) &
                       (kicnum_c[okclr] != 7988343) &
                       (kicnum_c[okclr] != 9591503) &
                       (kicnum_c[okclr] != 3240305)
                       )

        plt.figure()
        plt.scatter(Rossby[okclr][pok], clr[okclr][pok],
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, c=mass[okclr][pok], cmap=cm.Spectral)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass (M$_{\odot}$)')
        plt.ylabel('log R$_{' + EpointS + '}$ (#/day)')
        plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
        plt.xscale('log')
        plt.xlim(0.8e-2, 4e0)
        plt.ylim(-4, 0)
        plt.savefig(figdir + 'Rossby_rate_color' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        plt.figure()
        plt.scatter(Rossby[okclr][pok], np.log10(Lfl_Lbol)[okclr][pok],
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, c=mass[okclr][pok], cmap=cm.Spectral)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass (M$_{\odot}$)')
        plt.ylabel('log ($L_{fl}$ $L_{Kp}^{-1}$)')
        plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
        plt.xscale('log')
        plt.xlim(0.8e-2, 4e0)
        plt.ylim(-5, -1.5)
        plt.savefig(figdir + 'Rossby_lfllkp_color' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        plt.figure()
        plt.scatter(Rossby[okclr][pok], np.log10(Lfl_Lbol[okclr][pok] * Chi_fl(gi_all[okclr][pok])),
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, c=mass[okclr][pok], cmap=cm.Spectral)
        cbar = plt.colorbar()
        cbar.set_label(r'Mass (M$_{\odot}$)')
        plt.ylabel('log ($L_{fl}$ $L_{bol}^{-1}$)')
        plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
        plt.xscale('log')
        plt.xlim(0.8e-2, 4e0)
        # plt.ylim(-5, -1.5)
        plt.savefig(figdir + 'Rossby_lfllbol_color' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()




        B_V_pok = getBV(mass[okclr][pok])
        age_pok = MH2008_age(B_V_pok, Prot_all[okclr][pok])



        plt.figure()
        plt.scatter(mass[okclr][pok], age_pok, c=clr[okclr][pok],
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log R$_{' + EpointS + '}$ (#/day)')
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.15,0.35)
        plt.ylabel('Age (Myr)')
        plt.yscale('log')
        plt.savefig(figdir + 'mass_age_R35' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        plt.figure()
        plt.scatter(gi_all[okclr][pok], Prot_all[okclr][pok], c=clr[okclr][pok],
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log R$_{' + EpointS + '}$ (#/day)')
        plt.xlabel('g-i (mag)')
        plt.xlim(0.4,3)
        plt.ylabel(r'P$_{rot}$ (days)')
        plt.yscale('log')
        plt.savefig(figdir + 'color_rot_R35' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        plt.figure()
        plt.scatter(mass[okclr][pok], age_pok, c=np.log10(Lfl_Lbol)[okclr][pok],
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log ($L_{fl}$ $L_{Kp}^{-1}$)')
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.15,0.35)
        plt.ylabel('Age (Myr)')
        plt.yscale('log')
        plt.savefig(figdir + 'mass_age_lfllkp' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        plt.figure()
        plt.scatter(gi_all[okclr][pok], Prot_all[okclr][pok], c=np.log10(Lfl_Lbol)[okclr][pok],
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log ($L_{fl}$ $L_{Kp}^{-1}$)')
        plt.xlabel('g-i (mag)')
        plt.xlim(0.4,3)
        plt.ylabel(r'P$_{rot}$ (days)')
        plt.yscale('log')
        plt.savefig(figdir + 'color_rot_lfllkp' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()



        plt.figure()
        plt.scatter(gi_all[okclr][pok], Prot_all[okclr][pok], c=np.log10(Lfl_Lbol[okclr][pok] * Chi_fl(gi_all[okclr][pok])),
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log ($L_{fl}$ $L_{bol}^{-1}$)')
        plt.xlabel('g-i (mag)')
        plt.xlim(0.4,3)
        plt.ylabel(r'P$_{rot}$ (days)')
        plt.yscale('log')
        plt.savefig(figdir + 'color_rot_lflbol' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


    '''
    >>>>>>>>>>>>>>>>>>>>>>>
    
    Loop over all flare stars, stack data up into big arrays to fit w/ age-dependent evolution
    
    Version 1: ONLY FOR FLARE STARS (okclr)
    
    >>>>>>>>>>>>>>>>>>>>>>>
    '''


    ts = np.where((Prot_all[okclr] >= 0.1) &  # need valid rotation periods
                  (Prot_all[okclr] <= 30.0) & # need valid rotation periods
                  (Rossby[okclr] > 0.01) &  # need valid Rossby number
                  (Rossby[okclr] < 10) &    # need valid Rossby number
                  (gi_all[okclr] >= 0.5) & # manually throw out the bluest stars
                  (kicnum_c[okclr] != 10924462) & # manually throw out a few weird FFDs
                  (kicnum_c[okclr] != 3864443) &
                  (kicnum_c[okclr] != 5559631) &
                  (kicnum_c[okclr] != 7988343) &
                  (kicnum_c[okclr] != 9591503) &
                  (kicnum_c[okclr] != 3240305)
                  )
    ts = ts[0][np.argsort(Rossby[okclr][ts])]

    print('>> TS: number of stars to analyze that pass Prot >=0.1 and g-i>0.5 is ', np.size(ts))

    #######
    # output a simple text file with these KIC ID's
    # (for M. Scoggins' work)

    kicout = pd.DataFrame(data={'kic':kicnum_c[okclr][ts]})
    kicout.to_csv('kics_to_study.txt')



    logE_stack = np.array([])
    logt_stack = np.array([])
    mass_stack = np.array([])
    logR_stack = np.array([])
    logRerr_stack = np.array([])

    R_stack = np.array([]) # non-log versions
    Rerr_stack = np.array([])

    B_V_ts = getBV(mass[okclr][ts])
    age_ts = MH2008_age(B_V_ts, Prot_all[okclr][ts])

    for l in range(np.size(ts)):
        # find all entires for this star (LLC and SLC data)
        star = np.where((fdata[0].values == kicnum_c[okclr][ts][l]))[0]
        # arrays for FFD
        fnorm = np.zeros_like(edbins[1:])
        fsum = np.zeros_like(edbins[1:])

        # find this star in the KIC data
        mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[okclr][ts][l]))
        Lkp_i = Lkp_uniq[mtch][0]
        # tmp array to hold the total number of flares in each FFD bin
        flare_tot = np.zeros_like(fnorm)

        for i in range(0, len(star)):
            # Find the portion of the FFD that is above the 68% cutoff
            ok = np.where((edbins[1:] >= fdata.loc[star[i], 3]))[0]

            # ok = np.where((edbins[1:] >= fdata.loc[star[i], 3]) &
            #               (edbins[1:] < 36.1))[0] # add a cutoff to remove the "break"

            if len(ok) > 0:
                # add the rates together for a straight mean
                fsum[ok] = fsum[ok] + fdata.loc[star[i], 7:].values[ok]

                # count the number for the straight mean
                fnorm[ok] = fnorm[ok] + 1

                # add the actual number of flares for this data portion: rate * duration
                flare_tot[ok] = flare_tot[ok] + (fdata.loc[star[i], 7:].values[ok] * fdata.loc[star[i], 2])

        # the important arrays for the averaged FFD
        ffd_x = edbins[1:][::-1] + Lkp_i
        ffd_y = np.cumsum(fsum[::-1] / fnorm[::-1])

        # the "error" is the Poisson err from the total # flares per bin
        ffd_yerr = _Perror(flare_tot[::-1], down=True) / dur_all[okclr][ts][l]

        # Fit the FFD w/ a line, save the coefficeints
        ffd_ok = np.where((ffd_y > 0) & np.isfinite(ffd_y) & (fsum[::-1] > 0) &
                          np.isfinite(ffd_x) & np.isfinite(ffd_yerr) &
                          (ffd_x < 38.5))  # fix one of the outlier problems


        # if there are at least 2 energy bins w/ valid flares...
        if len(ffd_ok[0]) > 1:
            # combine all dimensions of data into 1 set of arrays
            logE_stack = np.append(logE_stack, ffd_x[ffd_ok])
            logR_stack = np.append(logR_stack, np.log10(ffd_y[ffd_ok]))
            logt_stack = np.append(logt_stack, np.ones_like(ffd_x[ffd_ok]) * np.log10(age_ts[l]))
            mass_stack = np.append(mass_stack, np.ones_like(ffd_x[ffd_ok]) * mass[okclr][ts][l])

            # propgate the log of the error in the FFD
            yerr_temp = np.abs(ffd_yerr[ffd_ok] / (ffd_y[ffd_ok] * np.log(10.)))
            yerr_temp = np.sqrt(yerr_temp**2. + np.nanmedian(yerr_temp)**2.)
            logRerr_stack = np.append(logRerr_stack, yerr_temp)

            R_stack = np.append(R_stack, ffd_y[ffd_ok])
            Rerr_stack = np.append(Rerr_stack, ffd_yerr[ffd_ok])




    # # # # # # # # # # # # # # # # # #
    # fit with model
    #
    # # # # # # # # # # # # # # # # # #

    # stack data for curve_fit
    stackOK = np.where(np.isfinite(logE_stack) & np.isfinite(logR_stack) &
                       np.isfinite(logt_stack) & np.isfinite(mass_stack) &
                       np.isfinite(logRerr_stack))
    X = (logE_stack[stackOK], logt_stack[stackOK], mass_stack[stackOK])


    # version 2 of model: just mass and age terms, no cross terms
    p0 = (0., 0., -0.5,
          -1., 1., 10.)
    fit, cov = curve_fit(FlareEqn, X, logR_stack[stackOK], p0,
                         sigma=logRerr_stack[stackOK])

    modelStack = FlareEqn(X, *fit)
    ChiSq1 = appaloosa.chisq(logR_stack[stackOK], logRerr_stack[stackOK], modelStack)
    BIC1 = ChiSq1 + np.size(p0) * np.log(np.size(X))

    print('>>> FlareEqn coefficients:')
    print(fit)
    # print('Chi, BIC')
    # print(ChiSq1, BIC1)


    #############
    #
    # Use MCMC to refine Least-Sq model fit, get errors
    # %% HERE
    #
    #############

    ndim = 6
    nwalkers = 100
    nsteps0 = 500
    nsteps1 = 1000
    pos = [fit + 1e-3 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler = emcee.EnsembleSampler(nwalkers, ndim, flare_lnprob, args=(X, logR_stack[stackOK], logRerr_stack[stackOK]),
                                    a=3)

    print('> MCMC burn-in: ' + str(nsteps0))
    pos1, prob1, state1 = sampler.run_mcmc(pos, nsteps0)
    sampler.reset()

    print('> MCMC run: ' + str(nsteps1))
    pos2, prob2, state2 = sampler.run_mcmc(pos1, nsteps1, rstate0=state1)

    # from this Gist: https://gist.github.com/banados/2254240
    af = sampler.acceptance_fraction
    af_msg = '''As a rule of thumb, the acceptance fraction (af) should be 
                                    between 0.2 and 0.5
                    If af < 0.2 decrease the a parameter
                    If af > 0.5 increase the a parameter
                    '''
    print(af_msg)
    print(">> Mean acceptance fraction:", np.mean(af))

    samples = sampler.chain.reshape((-1, ndim))
    # print('>> SAMPLER SHAPE', np.shape(samples))

    fig = corner.corner(samples, labels=["$a_1$", "$a_2$", "$a_3$", "$b_1$", "$b_2$", "$b_3$"])
    plt.savefig(figdir + "triangle" + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)


    print(">>> median sample parameters (FIT PARAMS!): ", np.nanmedian(samples, axis=0))
    print(">>> stddev sample parameters (ERR PARAMS!): ", np.nanstd(samples, axis=0))

    try:
        print(">> Autocorrelation time:", sampler.acor)
    except:
        print('chain too short for ACOR')

    fit_orig = fit
    fit = np.nanmedian(samples, axis=0)


    # now some plots exploring this fit
    # 0.5 solar mass star, 4 ages
    ffig = plt.figure(figsize=(6, 5))
    ax = ffig.add_subplot(111)
    y1 = FlareEqn((np.arange(33, 37), np.array([1, 1, 1, 1]), np.array([0.5, 0.5, 0.5, 0.5])), *fit)
    y2 = FlareEqn((np.arange(33, 37), np.array([2, 2, 2, 2]), np.array([0.5, 0.5, 0.5, 0.5])), *fit)
    y3 = FlareEqn((np.arange(33, 37), np.array([3, 3, 3, 3]), np.array([0.5, 0.5, 0.5, 0.5])), *fit)
    y4 = FlareEqn((np.arange(33, 37), np.array([4, 4, 4, 4]), np.array([0.5, 0.5, 0.5, 0.5])), *fit)

    plt.plot(np.arange(33, 37), 10.**y1, c='CornflowerBlue')
    plt.plot(np.arange(33, 37), 10.**y2, c='DarkViolet')
    plt.plot(np.arange(33, 37), 10.**y3, c='FireBrick')
    plt.plot(np.arange(33, 37), 10.**y4, c='Red')

    plt.yscale('log')
    plt.xlabel('log Flare Energy (erg)')
    plt.ylabel('Cumulative Flare Freq (#/day)')
    # plt.xlim(32, 38)
    # plt.ylim(1e-4, 1e0)
    plt.text(0.025, 0.025, r'M = 0.5 M$_{\odot}$, log t = [1,2,3,4] Myr', fontsize=12, transform=ax.transAxes)
    plt.savefig(figdir + 'eqnFFD_mass1' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    # solar-mass star, 4 ages
    ffig = plt.figure(figsize=(6, 5))
    ax = ffig.add_subplot(111)
    y1 = FlareEqn((np.arange(33, 37), np.array([1, 1, 1, 1]), np.array([1.0, 1.0, 1.0, 1.0])), *fit)
    y2 = FlareEqn((np.arange(33, 37), np.array([2, 2, 2, 2]), np.array([1.0, 1.0, 1.0, 1.0])), *fit)
    y3 = FlareEqn((np.arange(33, 37), np.array([3, 3, 3, 3]), np.array([1.0, 1.0, 1.0, 1.0])), *fit)
    y4 = FlareEqn((np.arange(33, 37), np.array([4, 4, 4, 4]), np.array([1.0, 1.0, 1.0, 1.0])), *fit)

    plt.plot(np.arange(33, 37), 10.**y1, c='CornflowerBlue')
    plt.plot(np.arange(33, 37), 10.**y2, c='DarkViolet')
    plt.plot(np.arange(33, 37), 10.**y3, c='FireBrick')
    plt.plot(np.arange(33, 37), 10.**y4, c='Red')

    plt.yscale('log')
    plt.xlabel('log Flare Energy (erg)')
    plt.ylabel('Cumulative Flare Freq (#/day)')
    plt.text(0.025, 0.025, r'M = 1.0 M$_{\odot}$, log t = [1,2,3,4] Myr', fontsize=12, transform=ax.transAxes)
    plt.savefig(figdir + 'eqnFFD_mass2' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    # same as other plots, but for the Sun!!!!!!!!!!!!
    plt.figure(figsize=(6, 5))
    y1 = FlareEqn((np.arange(28, 37), np.log10(4600. * np.ones(9)), np.ones(9)), *fit)

    plt.plot(np.arange(28, 37), (10. ** y1) * 365.25 / (10.**np.arange(28, 37)), c='k')

    plt.yscale('log')
    plt.xlabel('log Flare Energy (erg)')
    plt.ylabel('Cumulative Flare Freq (#/year/erg)')

    plt.title('Sun (4.6Gyr)')
    plt.savefig(figdir + 'eqnFFD_Sun' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    # R35 grid plot, moving towards what was on my original proposal!
    age_y = np.arange(0.1, 3.8, 0.1)
    mass_x = np.arange(0.25, 1.1, 0.05)
    xx,yy = np.meshgrid(mass_x, age_y, indexing='ij')

    rate_grid = np.zeros((mass_x.size, age_y.size))
    for i in range(mass_x.size):
        for j in range(age_y.size):
            rate_grid[i,j] = FlareEqn((35., age_y[j], mass_x[i]), *fit)


    # plt.figure()
    # plt.contourf(xx, yy, rate_grid, cmap=cm.magma_r)
    # plt.xlabel(r'Mass (M$_{\odot}$)')
    # plt.xlim(1.15,0.25)
    # plt.ylabel('log Age (Myr)')
    # cbar = plt.colorbar()
    # cbar.set_label('log R$_{' + EpointS + '}$ (#/day)')
    # plt.savefig(figdir + 'eqnFFD_grid35' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.close()


    # fit the FFD model at the energy the corresponds to 1 sec of the quiescent Luminosity
    # (i.e. 1 sec Equiv Duration)
    R1s_fit = FlareEqn((Lkp_all[okclr][ts], np.log10(age_ts), mass[okclr][ts]), *fit)

    plt.figure()
    plt.scatter(mass[okclr][ts], age_ts, c=(R1s_fit),
                s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
    cbar = plt.colorbar()
    cbar.set_label('log R$_{1s}$ (#/day)')
    plt.xlabel(r'Mass (M$_{\odot}$)')
    plt.xlim(1.15, 0.35)
    plt.ylabel('Age (Myr)')
    plt.yscale('log')
    plt.savefig(figdir + 'mass_age_R1s' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter(gi_all[okclr][ts], Prot_all[okclr][ts], c=(R1s_fit),
                s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
    cbar = plt.colorbar()
    cbar.set_label('log R$_{1s}$ (#/day)')
    plt.xlabel('g-i (mag)')
    plt.xlim(0.4, 3)
    plt.ylabel(r'P$_{rot}$ (days)')
    plt.yscale('log')
    plt.savefig(figdir + 'color_rot_R1s' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    mx, ly = massL()
    age_y = np.arange(1.0, 3.8, 0.05)
    xx, yy = np.meshgrid(mx, age_y, indexing='ij')

    R1s_grid = np.zeros((mx.size, age_y.size))
    for i in range(mx.size):
        for j in range(age_y.size):
            R1s_grid[i, j] = FlareEqn((ly[i], age_y[j], mx[i]), *fit)

    plt.figure()
    plt.contourf(xx,yy, R1s_grid, cmap=cm.magma_r)
    plt.xlabel(r'Mass (M$_{\odot}$)')
    plt.xlim(1.1, 0.25)
    plt.ylabel('log Age (Myr)')
    cbar = plt.colorbar()
    cbar.set_label('log R$_{1s}$ (#/day)')
    plt.savefig(figdir + 'eqnFFD_grid_R1s' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

    # lvls = np.append(0, np.logspace(-2,np.log10(20),10))
    plt.figure()
    plt.contourf(xx, yy + 6, R1s_grid, cmap=cm.Blues)
    plt.xlabel(r'Mass (M$_{\odot}$)')
    plt.xlim(0.2, 1.1)
    plt.ylabel('log Age (years)')
    plt.ylim(7., 9.7)
    cbar = plt.colorbar()
    cbar.set_label('R$_{1s}$ (#/day)')
    CS = plt.contour(xx, yy + 6, R1s_grid, colors='white', linestyle='solid')
    plt.clabel(CS, fontsize=8, inline=1, color='white')
    plt.savefig(figdir + 'proposal_remake' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()




    # make figure of model FFD versus actual data for each target
    # compute model FFD at actual observed bins, calculate ChiSq
    # then make figure of ChiSq vs (color, Prot) to see where we're winning/losing
    
    ### =>> takes a few min to run, so turn off (if False) when not needed
    if True:
        print('making updated FFD for every star in TS')

        chisq_ts = np.zeros(np.size(ts))-1

        for l in range(np.size(ts)):

            # find all entires for this star (LLC and SLC data)
            star = np.where((fdata[0].values == kicnum_c[okclr][ts][l]))[0]
            # arrays for FFD
            fnorm = np.zeros_like(edbins[1:])
            fsum = np.zeros_like(edbins[1:])

            # find this star in the KIC data
            mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[okclr][ts][l]))
            Lkp_i = Lkp_uniq[mtch][0]
            # tmp array to hold the total number of flares in each FFD bin
            flare_tot = np.zeros_like(fnorm)

            ffig = plt.figure()
            ax = ffig.add_subplot(111)
            for i in range(0, len(star)):
                # Find the portion of the FFD that is above the 68% cutoff
                ok = np.where((edbins[1:] >= fdata.loc[star[i], 3]))[0]
                if len(ok) > 0:
                    # add the rates together for a straight mean
                    fsum[ok] = fsum[ok] + fdata.loc[star[i], 7:].values[ok]

                    # count the number for the straight mean
                    fnorm[ok] = fnorm[ok] + 1

                    # add the actual number of flares for this data portion: rate * duration
                    flare_tot = flare_tot + (fdata.loc[star[i], 7:].values * fdata.loc[star[i], 2])

                    plt.plot(edbins[1:][ok][::-1] + Lkp_i,
                             np.log10(np.cumsum(fdata.loc[star[i], 7:].values[ok][::-1]) + 1e-10), # put 1e-10 buffer in for plot
                             alpha=0.35, color='k', linewidth=0.5)
            # the important arrays for the averaged FFD
            ffd_x = edbins[1:][::-1] + Lkp_i
            ffd_y = np.cumsum(fsum[::-1] / fnorm[::-1])

            # the "error" is the Poisson err from the total # flares per bin
            ffd_yerr = _Perror(flare_tot[::-1], down=True) / dur_all[okclr][ts][l]

            # Fit the FFD w/ a line, save the coefficeints
            ffd_ok = np.where((ffd_y > 0) &
                              np.isfinite(ffd_y) & np.isfinite(ffd_x) & np.isfinite(ffd_yerr) & (fsum[::-1] > 0) &
                              (ffd_x < 38.5))  # fix one of the outlier problems


            # if there are at least 2 energy bins w/ valid flares...
            if len(ffd_ok[0]) > 1:
                yerr_temp = np.abs(ffd_yerr[ffd_ok] / (ffd_y[ffd_ok] * np.log(10.)))
                yerr_temp = np.sqrt(yerr_temp ** 2. + np.nanmedian(yerr_temp) ** 2.)

                # make model FFD at same   ffd_x values
                model_y = FlareEqn((ffd_x[ffd_ok], np.log10(age_ts[l]), mass[okclr][ts][l]), *fit)

                # compute chisq!
                chisq_ts[l] = np.sum( ((np.log10(ffd_y[ffd_ok]) - model_y) / yerr_temp)**2.0 ) / np.float(model_y.size)

                plt.plot(ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), linewidth=2, alpha=0.7, c='k')
                plt.errorbar(ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), yerr_temp, fmt='k,')
                plt.plot(ffd_x[ffd_ok], model_y, linewidth=1, alpha=0.7, c='r')

                plt.text(0.025, 0.025,
                         r'$\chi^2$=' + format(chisq_ts[l], '.2F') + ', g-i='+format(gi_all[okclr][ts][l],'.2F') + ', Prot='+format(Prot_all[okclr][ts][l]),
                         fontsize=10, transform=ax.transAxes)
                plt.xlim(np.nanmin(ffd_x[ffd_ok])-0.1, np.nanmax(ffd_x[ffd_ok])+0.1)
                plt.ylim(np.nanmin(np.log10(ffd_y[ffd_ok]))-np.nanmax(yerr_temp),
                         np.nanmax(np.log10(ffd_y[ffd_ok]))+2.*np.nanmin(yerr_temp))
            plt.xlabel('log Flare Energy (erg)')
            plt.ylabel('log Cumulative Flare Freq (#/day)')
            plt.savefig(figdir + 'ffd_models/' + str(kicnum_c[okclr][ts][l]) + '_ffd_fit' + figtype,
                        dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()


        plt.figure()
        plt.scatter(gi_all[okclr][ts], Prot_all[okclr][ts], c=np.log10(chisq_ts),
                    s=50, linewidths=0.5, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label(r'log $\chi^2$')
        plt.xlabel('g-i (mag)')
        plt.xlim(0.4, 3)
        plt.ylabel(r'P$_{rot}$ (days)')
        plt.yscale('log')
        plt.savefig(figdir + 'color_rot_chisq' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()




    '''
    To support Riley's work: 
    Make evolution tracks of flare activity over time for a wide binary. 
    Simple enough, just evaluate eqn for 2 masses at range of ages.
    
    Challenge: 
    His work is in Lfl_Lkep, not FFD space...
    I *think* this works by integrating FFD in Equiv Dur units over a 
    fixed range of equiv dur.
    Also need to undo the cumulative part...
    
    Make primary star = 0.85Msun, secondary take many tracks: 0.8, 0.7, 0.5Msun, etc
    '''
    if False: # set to True to run for Riley
        print('doing Lfl_Lkp tracks for Riley')

        # mass_wb = np.array([0.5,0.7,0.9,1.0])
        mass_wb = np.arange(0.45, 0.9, 0.05)

        # figure out the quiescent lum for these targets to make Energy -> ED
        msort = np.argsort(mass[okclr][ts])
        Lkp_wb = np.interp(mass_wb, mass[okclr][ts][msort], Lkp_all[okclr][ts][msort])

        age_range = np.arange(1.5, 5, 0.1)

        ed_wb = np.linspace(-6, 5, 20)

        # the array to fill w/ integrated model values!
        Lfl_Lkp_wb = np.zeros((mass_wb.size, age_range.size))

        for k in range(mass_wb.size):
            for l in range(age_range.size):
                # make the FFD for this (mass,age)
                # X = (logE, logT, M)
                ffd_kl = (10.**FlareEqn((ed_wb+Lkp_wb[k], age_range[l]*np.ones(20), mass_wb[k]*np.ones(20)), *fit)) / (60.*60.*24.)

                # undo the cumulative nature of the FFD
                for j in range(ffd_kl.size-1):
                    ffd_kl[0: -(1+j)] = ffd_kl[0: -(1+j)] - ffd_kl[-(1+j)]

                if k==0 and l==0:
                    print(k, l, 10.**(ed_wb), ffd_kl)

                # Integrate the FFD to estimate Lfl/Lkep
                # ffd_yi in units of cumulative #/day -> convert to #/sec. Assume duration of 1 sec for data
                Lfl_Lkp_wb[k,l] = np.log10(np.trapz(ffd_kl,
                                                    x=10.**(ed_wb)))


        plt.figure(figsize=(8.1,8))
        for k in range(mass_wb.size):
            plt.plot(Lfl_Lkp_wb[-1, :], Lfl_Lkp_wb[k, :], lw=1, c='Navy', alpha=0.7)
            plt.annotate(format(mass_wb[k] / mass_wb[-1], "4.2"), (Lfl_Lkp_wb[-1, -1], Lfl_Lkp_wb[k, -1]),
                         textcoords='data', size=10)
        # plt.plot(Lfl_Lkp_wb[-1, :], Lfl_Lkp_wb[-1, :], alpha=0.75, c='k', lw=1.5)
        plt.xlabel('log Lfl_Lkp A (0.85 Msun)')
        plt.ylabel('log Lfl_Lkp B (0.85-0.25 Msun)')
        plt.savefig(figdir + 'wb_model' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        np.savez('ABtracks.npz', ABarray=Lfl_Lkp_wb, mass=mass_wb, age=age_range)


    '''
    >>>>>>>>>>>>>>>>>>>>>>>
    
    Loop over all stars AGAIN, stack data up into big arrays to fit w/ age-dependent evolution
    
    Version 2: FOR ALL STARS WITH GOOD ROTATION/COLOR/ETC (i.e. no requirement for Nflares or Nflares68, etc)
    
    This is for the Null Detection (nd) correction, brought up at the Kepler/K2 Science Meeting 2017
    
    >>>>>>>>>>>>>>>>>>>>>>>
    '''

    oknd = np.where((Prot_all >= 0.1) &  # need valid rotation periods
                     (Prot_all <= 30.0) & # need valid rotation periods
                     (Rossby > 0.01) &  # need valid Rossby number
                     (Rossby < 10) &    # need valid Rossby number
                     (gi_all >= 0.5) & # manually throw out the bluest stars
                     (kicnum_c != 10924462) &  # manually throw out a few weird FFDs
                     (kicnum_c != 3864443) &
                     (kicnum_c != 5559631) &
                     (kicnum_c != 7988343) &
                     (kicnum_c != 9591503) &
                     (kicnum_c != 3240305)
                     )


    # sort all stars in order of Rossby number
    oknd = oknd[0][np.argsort(Rossby[oknd])]

    # oknd_y = np.where(np.isfinite(np.log10(fit_E[oknd])) & # have semi-valid FFD fit
    #                    (Nflare68[oknd] >= Nflare68_cut)) # have at least 10 flares above the 68% limit

    B_V_nd = getBV(mass[oknd])
    age_nd = MH2008_age(B_V_nd, Prot_all[oknd])


    #_______________________________
    # figures based on the convo w/ Covey:
    # what is the fraction of null detections as a function of age?
    # i.e. over what domain is the fit going to be valid?

    # plt.figure()
    # plt.scatter(np.log10(age_nd), Nflare68[oknd] + 1, s=2, alpha=0.5)
    # plt.yscale('log')
    # plt.xlabel('log Age (Myr)')
    # plt.ylabel('Nflare68+1')
    # plt.savefig(figdir + 'age_Nflare68.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.close()

    Ncutoff = 3
    Dage = 0.2
    age_bins = np.arange(-0.1, 4.1, Dage)
    frac_flare = np.zeros_like(age_bins)-1

    print('>>> total number of stars that pass EASIER cuts:',
          len(np.where((np.log10(age_nd) >= age_bins[0]) & (np.log10(age_nd) < (age_bins[-1]+Dage)))[0]))

    plt.figure()
    for k in range(len(age_bins)):
        xa = np.where((np.log10(age_nd) >= age_bins[k]) & (np.log10(age_nd) < (age_bins[k]+Dage)))[0]
        if len(xa) > 0:
            # count fraction of stars in this age bin with at least N flares above 68% cutoff
            frac_flare[k] = np.float(sum(Nflare68[oknd][xa] >= Ncutoff)) / np.float(len(xa))

    plt.plot(age_bins[frac_flare > -1], frac_flare[frac_flare > -1], lw=2, color='k')
    # plt.hist(age_bins[frac_flare > -1], weights=frac_flare[frac_flare > -1], histtype='step', lw=2, color='k')
    plt.xlabel('log Age (Myr)')
    plt.ylabel('Fraction Flaring')
    plt.ylim(0,1)
    plt.savefig(figdir + 'age_frac_flare' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # do another quick version of this fraction flaring plot, for stars w/i each color range...
    Ncutoff = 3
    Dage = 0.3
    age_bins = np.arange(-0.1, 4.1, Dage)
    plt.figure()
    for i in range(crng.shape[0]):
        frac_flare_C = np.zeros_like(age_bins)-1
        for k in range(len(age_bins)):
            xa = np.where((np.log10(age_nd) >= age_bins[k]) & (np.log10(age_nd) < (age_bins[k] + Dage)) &
                          (gi_all[oknd] >= crng[i, 0]) & (gi_all[oknd] < crng[i, 1]))[0]
            if len(xa) > 0:
                # count fraction of stars in this age bin with at least N flares above 68% cutoff
                frac_flare_C[k] = np.float(sum(Nflare68[oknd][xa] >= Ncutoff)) / np.float(len(xa))


        Nlbl = len(np.where((np.log10(age_nd) >= age_bins[0]) & (np.log10(age_nd) < (age_bins[-1] + Dage)) &
                            (gi_all[oknd] >= crng[i, 0]) & (gi_all[oknd] < crng[i, 1]))[0])

        plt.plot(age_bins[frac_flare_C > -1], frac_flare_C[frac_flare_C > -1],
                 lw=2, alpha=0.7, color=crng_clrs[i],
                 label=str(crng[i, 0])+'<g-i<'+str(crng[i, 1]) + ', N=' + str(Nlbl))
    plt.xlabel('log Age (Myr)')
    plt.ylabel('Fraction Flaring')
    plt.legend(fontsize=11)
    # plt.ylim(0, 1)
    plt.savefig(figdir + 'age_frac_flare_C' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    # actually do the stacking & analysis for the "nd" run
    if False:
        # SO, what is needed for the fit in the end? That should drive what we store/loop over.
        # For each star (flare or non), we need:
        # age (into an array, same size as E)
        # mass (into an array, same size as E)
        # energy bins to evaluate over (E)
        # flare rates (same size as E)

        logE_ndstack = np.array([])
        logt_ndstack = np.array([])
        mass_ndstack = np.array([])

        prot_ndstack = np.array([])
        clr_ndstack = np.array([])

        logR_ndstack = np.array([])
        logRerr_ndstack = np.array([])

        R_ndstack = np.array([])
        Rerr_ndstack = np.array([])

        # dex below the 68% cutoff to go, FYI: factor of 2 is 0.3.
        # use this param to account for fact that my injection tests aren't perfect...
        offset68 = 0

        if debug:
            print('>> ')
            print('reality check: ', np.size(oknd))

        print('now doing the big loop for ALL stars, including non-flaring...')
        print(datetime.datetime.now())

        for l in range(np.size(oknd)):
            # find all entires for this star (LLC and SLC data)
            star = np.where((fdata[0].values == kicnum_c[oknd][l]))[0]
            # arrays for FFD
            fnorm = np.zeros_like(edbins[1:])
            fsum = np.zeros_like(edbins[1:])

            # tmp array to hold the total number of flares in each FFD bin
            flare_tot = np.zeros_like(fnorm)

            # find this star in the KIC data
            mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[oknd][l]))
            Lkp_i = Lkp_uniq[mtch][0]

            if debug:
                print('l, len(star) = ', l, len(star))

            for i in range(0, len(star)):
                # Find the portion of the FFD that is above the 68% cutoff, but below some global max to catch some outliers
                # (note: in ED units)
                ok = np.where(( edbins[1:] >= (fdata.loc[star[i], 3]-offset68) ))[0]

                if len(ok) > 0: # if there is flare energies that were measurable...
                    # add the rates together from all quarters/months for a straight mean
                    fsum[ok] = fsum[ok] + fdata.loc[star[i], 7:].values[ok]

                    # count the number for the straight mean
                    fnorm[ok] = fnorm[ok] + 1

                    # add the actual number of flares for this data portion: rate * duration
                    flare_tot[ok] = flare_tot[ok] + (fdata.loc[star[i], 7:].values[ok] * fdata.loc[star[i], 2])


            # the important arrays for the averaged FFD
            ffd_x = edbins[1:][::-1] + Lkp_i
            ffd_y = np.cumsum(fsum[::-1] / fnorm[::-1])

            # the "error" is the Poisson err from the total # flares per bin
            ffd_yerr = _Perror(flare_tot[::-1], down=True) / dur_all[oknd][l]

            ffd_ok = np.where(np.isfinite(ffd_y) & #(fsum[::-1] > 0) &
                              np.isfinite(ffd_x) &
                              np.isfinite(ffd_yerr) &
                              (ffd_x < 38.5))  # fix one of the outlier problems (global max)

            if (len(ffd_ok[0]) > 0) & (np.isfinite(np.log10(age_nd[l]))) & (np.isfinite(mass[oknd][l])):
                # combine all dimensions of data into 1 set of arrays
                logE_ndstack = np.append(logE_ndstack, ffd_x[ffd_ok])
                logt_ndstack = np.append(logt_ndstack, np.ones_like(ffd_x[ffd_ok]) * np.log10(age_nd[l]))
                mass_ndstack = np.append(mass_ndstack, np.ones_like(ffd_x[ffd_ok]) * mass[oknd][l])
                logR_ndstack = np.append(logR_ndstack, np.log10(ffd_y[ffd_ok]))

                # propogate the log of the error in the FFD
                yerr_temp = np.abs(ffd_yerr[ffd_ok] / (ffd_y[ffd_ok] * np.log(10.)))
                yerr_temp = np.sqrt(yerr_temp**2. + np.nanmedian(yerr_temp)**2.)
                logRerr_ndstack = np.append(logRerr_ndstack, yerr_temp)

                R_ndstack = np.append(R_ndstack, ffd_y[ffd_ok])
                Rerr_ndstack = np.append(Rerr_ndstack, ffd_yerr[ffd_ok])

                clr_ndstack = np.append(clr_ndstack, np.ones_like(ffd_x[ffd_ok]) * gi_all[oknd][l])
                prot_ndstack = np.append(prot_ndstack, np.ones_like(ffd_x[ffd_ok]) * Prot_all[oknd][l])

                if debug:
                    print('worked once')
                    print('l, ffd_x, ffd_y, ffd_yerr', l, ffd_x[ffd_ok], ffd_y[ffd_ok], ffd_yerr[ffd_ok])
                    print('logE_ndstack', logE_ndstack)
                    print('logt_ndstack', logt_ndstack)
                    print('mass_ndstack', mass_ndstack)
                    print('R_ndstack', R_ndstack)
                    print('Rerr_ndstack', Rerr_ndstack)

                    print(5/0)



        print('loop done')
        print(datetime.datetime.now())


        # # # # # # # # # # # # # # # # # #
        # fit with model
        # # # # # # # # # # # # # # # # # #
        stackOK_nd = np.where(np.isfinite(logE_ndstack) & np.isfinite(R_ndstack) &
                              np.isfinite(logt_ndstack) & np.isfinite(mass_ndstack) &
                              np.isfinite(Rerr_ndstack))
        X_nd = (logE_ndstack[stackOK_nd], logt_ndstack[stackOK_nd], mass_ndstack[stackOK_nd])


        # write output table to read in later, or to fit w/ heirarchical model, etc
        dfffd = pd.DataFrame(data={'logE':logE_ndstack[stackOK_nd],
                                   'logAge':logt_ndstack[stackOK_nd],
                                   'mass':mass_ndstack[stackOK_nd],
                                   'giclr':clr_ndstack[stackOK_nd],
                                   'Prot':prot_ndstack[stackOK_nd],
                                   'FF':R_ndstack[stackOK_nd],
                                   'FFerr':Rerr_ndstack[stackOK_nd],
                                   'logFF':logR_ndstack[stackOK_nd],
                                   'logFFerr':logRerr_ndstack[stackOK_nd]})
        dfffd.to_csv('ensemble_FFD.csv', index=False)

        p0 = fit # use the previous results as the prior for the fit

        # print('reality check: ')
        # print()
        # print(np.shape(X_nd))
        # print(np.shape(stackOK_nd[0]))
        # print(len(Rerr_ndstack), len(logE_ndstack))

        fit_nd, cov_nd = curve_fit(FlareEqn_nolog, X_nd, R_ndstack[stackOK_nd], p0,
                                   sigma=Rerr_ndstack[stackOK_nd])

        # res = minimize(FlareEqn, p0, method='BFGS', args=(X_nd, logR_ndstack[stackOK_nd]))

        modelStack_nd = FlareEqn_nolog(X_nd, *fit_nd)
        ChiSq1_nd = appaloosa.chisq(R_ndstack[stackOK_nd], Rerr_ndstack[stackOK_nd], modelStack_nd)
        BIC1_nd = ChiSq1_nd + np.size(p0) * np.log(np.size(X_nd))

        print('FlareEqn coefficients _nd:')
        print(fit_nd)
        print('Chi, BIC')
        print(ChiSq1_nd, BIC1_nd)


        ##### make some plots ....

        plt.figure(figsize=(6, 5))
        y1 = FlareEqn_nolog((np.arange(28, 37), np.log10(4600. * np.ones(9)), np.ones(9)), *fit_nd)
        plt.plot(np.arange(28, 37), (y1) * 365.25 / (10.**np.arange(28, 37)), c='k')
        plt.yscale('log')
        plt.xlabel('log Flare Energy (erg)')
        plt.ylabel('Cumulative Flare Freq (#/year/erg)')

        plt.title('Sun (4.6Gyr)')
        plt.savefig(figdir + 'eqnFFD_Sun_nd' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        # R35 grid plot, moving towards what was on my original proposal!
        age_y = np.arange(0.1, 3.8, 0.1)
        mass_x = np.arange(0.25, 1.1, 0.05)

        # xx,yy = np.meshgrid(mass_x, age_y, indexing='ij')
        # rate_grid = np.zeros((mass_x.size, age_y.size))
        # for i in range(mass_x.size):
        #     for j in range(age_y.size):
        #         rate_grid[i,j] = np.log10(FlareEqn_nolog((35., age_y[j], mass_x[i]), *fit_nd))

        R1s_fit_nd = np.log10(FlareEqn_nolog((Lkp_all[oknd], np.log10(age_nd), mass[oknd]), *fit_nd))

        plt.figure()
        plt.scatter(mass[oknd], age_nd, c=(R1s_fit_nd),
                    s=np.sqrt(10**R1s_fit_nd)*15., linewidths=0, edgecolors='k', alpha=0.5, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log R$_{1s}$ (#/day)')
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.15, 0.35)
        plt.ylabel('Age (Myr)')
        plt.yscale('log')
        plt.savefig(figdir + 'mass_age_R1s_nd.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        plt.figure()
        plt.scatter(gi_all[oknd], Prot_all[oknd], c=(R1s_fit_nd),
                    s=np.sqrt(10**R1s_fit_nd)*15., linewidths=0, edgecolors='k', alpha=0.5, cmap=cm.magma_r)
        cbar = plt.colorbar()
        cbar.set_label('log R$_{1s}$ (#/day)')
        plt.xlabel('g-i (mag)')
        plt.xlim(0.4, 3)
        plt.ylabel(r'P$_{rot}$ (days)')
        plt.yscale('log')
        plt.savefig(figdir + 'color_rot_R1s_nd.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        mx, ly = massL()
        age_y = np.arange(1.0, 3.8, 0.05)
        xx, yy = np.meshgrid(mx, age_y, indexing='ij')

        R1s_grid_nd = np.zeros((mx.size, age_y.size))
        for i in range(mx.size):
            for j in range(age_y.size):
                R1s_grid_nd[i, j] = np.log10(FlareEqn_nolog((ly[i], age_y[j], mx[i]), *fit_nd))

        plt.figure()
        plt.contourf(xx,yy, R1s_grid_nd, cmap=cm.magma_r)
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.1, 0.25)
        plt.ylabel('log Age (Myr)')
        cbar = plt.colorbar()
        cbar.set_label('log R$_{1s}$ (#/day)')
        plt.savefig(figdir + 'eqnFFD_grid_R1s_nd' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

        # lvls = np.append(0, np.logspace(-2,np.log10(20),10))
        plt.figure()
        plt.contourf(xx, yy + 6, R1s_grid_nd, cmap=cm.Blues)
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(0.2, 1.1)
        plt.ylabel('log Age (years)')
        plt.ylim(7., 9.7)
        cbar = plt.colorbar()
        cbar.set_label('R$_{1s}$ (#/day)')
        CS = plt.contour(xx, yy + 6, R1s_grid_nd, colors='white', linestyle='solid')
        plt.clabel(CS, fontsize=8, inline=1, color='white')
        plt.savefig(figdir + 'proposal_remake_nd' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        ### =>> make per-star FFD results again, but this time with the new model applied,
        if False:
            chisq_nd = np.zeros(np.size(ts))-1

            for l in range(np.size(ts)):

                # find all entires for this star (LLC and SLC data)
                star = np.where((fdata[0].values == kicnum_c[okclr][ts][l]))[0]
                # arrays for FFD
                fnorm = np.zeros_like(edbins[1:])
                fsum = np.zeros_like(edbins[1:])

                # find this star in the KIC data
                mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[okclr][ts][l]))
                Lkp_i = Lkp_uniq[mtch][0]
                # tmp array to hold the total number of flares in each FFD bin
                flare_tot = np.zeros_like(fnorm)

                ffig = plt.figure()
                ax = ffig.add_subplot(111)
                for i in range(0, len(star)):
                    # Find the portion of the FFD that is above the 68% cutoff
                    ok = np.where((edbins[1:] >= (fdata.loc[star[i], 3] - offset68)))[0]
                    if len(ok) > 0:
                        # add the rates together for a straight mean
                        fsum[ok] = fsum[ok] + fdata.loc[star[i], 7:].values[ok]

                        # count the number for the straight mean
                        fnorm[ok] = fnorm[ok] + 1

                        # add the actual number of flares for this data portion: rate * duration
                        flare_tot = flare_tot + (fdata.loc[star[i], 7:].values * fdata.loc[star[i], 2])

                        plt.plot(edbins[1:][ok][::-1] + Lkp_i,
                                 np.log10(np.cumsum(fdata.loc[star[i], 7:].values[ok][::-1]) + 1e-10), # put 1e-10 buffer in for plot
                                 alpha=0.35, color='k', linewidth=0.5)
                # the important arrays for the averaged FFD
                ffd_x = edbins[1:][::-1] + Lkp_i
                ffd_y = np.cumsum(fsum[::-1] / fnorm[::-1])

                # the "error" is the Poisson err from the total # flares per bin
                ffd_yerr = _Perror(flare_tot[::-1], down=True) / dur_all[okclr][ts][l]

                # Fit the FFD w/ a line, save the coefficeints
                ffd_ok = np.where((ffd_y > 0) & (fsum[::-1] > 0) &
                                  np.isfinite(ffd_y) & np.isfinite(ffd_x) & np.isfinite(ffd_yerr) &
                                  (ffd_x < 38.5))  # fix one of the outlier problems


                # if there are at least 2 energy bins w/ valid flares...
                if len(ffd_ok[0]) > 1:
                    yerr_temp = np.abs(ffd_yerr[ffd_ok] / (ffd_y[ffd_ok] * np.log(10.)))
                    yerr_temp = np.sqrt(yerr_temp ** 2. + np.nanmedian(yerr_temp) ** 2.)

                    # make model FFD at same   ffd_x values
                    model_y = FlareEqn_nolog((ffd_x[ffd_ok], np.log10(age_ts[l]), mass[okclr][ts][l]), *fit_nd)

                    # compute chisq!
                    chisq_nd[l] = np.sum( ((np.log10(ffd_y[ffd_ok]) - np.log10(model_y)) / yerr_temp)**2.0 ) / np.float(model_y.size)

                    plt.plot(ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), linewidth=2, alpha=0.7, c='k')
                    plt.errorbar(ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), yerr_temp, fmt='k,')
                    plt.plot(ffd_x[ffd_ok], np.log10(model_y), linewidth=1, alpha=0.7, c='r')

                    plt.text(0.025, 0.025,
                             'g-i='+format(gi_all[okclr][ts][l],'.2F') + ', Prot='+format(Prot_all[okclr][ts][l]),
                             fontsize=10, transform=ax.transAxes)
                    plt.xlim(np.nanmin(ffd_x[ffd_ok])-0.1, np.nanmax(ffd_x[ffd_ok])+0.1)
                    plt.ylim(np.nanmin(np.log10(ffd_y[ffd_ok]))-np.nanmax(yerr_temp),
                             np.nanmax(np.log10(ffd_y[ffd_ok]))+2.*np.nanmin(yerr_temp))
                plt.xlabel('log Flare Energy (erg)')
                plt.ylabel('log Cumulative Flare Freq (#/day)')
                plt.savefig(figdir + 'ffd_models/' + str(kicnum_c[okclr][ts][l]) + '_ffd_fit_nd' + figtype,
                            dpi=300, bbox_inches='tight', pad_inches=0.5)
                plt.close()


            plt.figure()
            plt.scatter(gi_all[okclr][ts], Prot_all[okclr][ts], c=np.log10(chisq_nd),
                        s=50, linewidths=0, edgecolors='k', alpha=0.85, cmap=cm.magma_r)
            cbar = plt.colorbar()
            cbar.set_label(r'log $\chi^2$')
            plt.xlabel('g-i (mag)')
            plt.xlim(0.4, 3)
            plt.ylabel(r'P$_{rot}$ (days)')
            plt.yscale('log')
            plt.savefig(figdir + 'color_rot_chisq_nd.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()




    #### MODEL COMPARISON RABBIT HOLE
    if False:
        ### MODEL COMPARISON
        # we can actually just try out lots of models here, now that we have the X and X_nd datasets

        # p0 = (0., 0., -0.5,
        #       -1., 1., 10.)

        fit3, cov3 = curve_fit(FlareEqn_nolog, X, R_stack[stackOK],p0=fit,
                               sigma=Rerr_stack[stackOK])

        print('fit3: ', fit3)

        R1s_grid_3 = np.zeros((mx.size, age_y.size))
        for i in range(mx.size):
            for j in range(age_y.size):
                R1s_grid_3[i, j] = np.log10(FlareEqn_nolog((ly[i], age_y[j], mx[i]), *fit3))



        p4 = (-1, -1, 0, 0, 0, 0, 1, 10)
        fit4, cov4 = curve_fit(FlareEqn2, X, logR_stack[stackOK], p0=p4,
                               sigma=logRerr_stack[stackOK])
        p5 = (-1, -1, 0, 0, 0, 0, 1, 10)
        fit5, cov5 = curve_fit(FlareEqn2_nolog, X, R_stack[stackOK], p0=p5,
                               sigma=Rerr_stack[stackOK])

        print('fit4: ', fit4)
        print('fit5: ', fit5)


        R1s_grid_4 = np.zeros((mx.size, age_y.size))
        for i in range(mx.size):
            for j in range(age_y.size):
                R1s_grid_4[i, j] = (FlareEqn2((ly[i], age_y[j], mx[i]), *fit4))

        R1s_grid_5 = np.zeros((mx.size, age_y.size))
        for i in range(mx.size):
            for j in range(age_y.size):
                R1s_grid_5[i, j] = np.log10(FlareEqn2_nolog((ly[i], age_y[j], mx[i]), *fit5))

        # fit_nd, cov_nd = curve_fit(FlareEqn_nolog, X_nd, R_ndstack[stackOK_nd], p0=fit,
        #            sigma=Rerr_ndstack[stackOK_nd])

        plt.figure()
        plt.contourf(xx, yy, R1s_grid_3, cmap=cm.magma_r)
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.1, 0.25)
        plt.ylabel('log Age (Myr)')
        cbar = plt.colorbar()
        cbar.set_label('log R$_{1s}$ (#/day)')
        plt.savefig(figdir + 'eqnFFD_grid_R1s_3' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        plt.figure()
        plt.contourf(xx, yy, R1s_grid_4, cmap=cm.magma_r)
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.1, 0.25)
        plt.ylabel('log Age (Myr)')
        cbar = plt.colorbar()
        cbar.set_label('log R$_{1s}$ (#/day)')
        plt.savefig(figdir + 'eqnFFD_grid_R1s_4' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        plt.figure()
        plt.contourf(xx, yy, R1s_grid_5, cmap=cm.magma_r)
        plt.xlabel(r'Mass (M$_{\odot}$)')
        plt.xlim(1.1, 0.25)
        plt.ylabel('log Age (Myr)')
        cbar = plt.colorbar()
        cbar.set_label('log R$_{1s}$ (#/day)')
        plt.savefig(figdir + 'eqnFFD_grid_R1s_5' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        # compare the models directly
        # show the "R1s" flare rate as a function of Age for 3 different Masses (on 1 plot)
        plt.figure()
        plt.plot(age_y, R1s_grid[0, :], c='red')
        plt.plot(age_y, R1s_grid_3[0, :], c='red', linestyle=':')
        plt.plot(age_y, R1s_grid_nd[0, :], c='red', linestyle='--')

        plt.plot(age_y, R1s_grid[50, :], c='orange')
        plt.plot(age_y, R1s_grid_3[50, :], c='orange', linestyle=':')
        plt.plot(age_y, R1s_grid_nd[50, :], c='orange', linestyle='--')

        plt.plot(age_y, R1s_grid[100, :], c='blue')
        plt.plot(age_y, R1s_grid_3[100, :], c='blue', linestyle=':')
        plt.plot(age_y, R1s_grid_nd[100, :], c='blue', linestyle='--')

        plt.ylabel('log R1s')
        plt.xlabel('log Age (Myr)')
        plt.savefig(figdir + 'flare_model_compare.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


if __name__ == "__main__":
    '''
      let this file be called from the terminal directly. e.g.:
      $ python analysis.py
    '''
    paper2_plots()
