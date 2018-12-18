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

