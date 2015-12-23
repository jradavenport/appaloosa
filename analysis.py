'''
Routines to do analysis on the appaloosa flare finding runs. Including
  - plots for the paper
  - check against other sample of flares from Kepler
  - completeness and efficiency tests against FBEYE results
  - completeness and efficiency tests against fake data (?)
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from os.path import expanduser
import os
import appaloosa
import pandas as pd


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
    dist = 10.0**(mu/5.0 + 1)
    return dist


def fbeye_compare(apfile='9726699.flare', fbeyefile='gj1243_master_flares.tbl'):
    '''
    compare flare finding and properties between appaloosa and FBEYE
    '''

    # read the aprun output
    # t_start, t_stop, t_peak, amplitude, FWHM, duration, t_peak_aflare1, t_FWHM_aflare1,
    # amplitude_aflare1, flare_chisq, KS_d_model, KS_p_model,
    # KS_d_cont, KS_p_cont, Equiv_Dur
    apdata = np.loadtxt(apfile, delimiter=',', dtype='float', skiprows=5, comments='#')




    # index of flare start in "gj1243_master_slc.dat"
    # index of flare stop in "gj1243_master_slc.dat"
    # t_start
    # t_stop
    # t_peak
    # t_rise
    # t_decay
    # flux peak (in fractional flux units)
    # Equivalent Duration (ED) in units of per seconds
    # Equiv. Duration of rise (t_start to t_peak)
    # Equiv. Duration of decay (t_peak to t_stop)
    # Complex flag (1=classical, 2=complex) by humans
    # Number of people that identified flare event exists
    # Number of people that analyzed this month
    # Number of flare template components fit to event (1=classical >1=complex)

    # read the FBEYE output
    fbdata = np.loadtxt(fbeyefile, comments='#', dtype='float')


    # step thru each FBEYE flare
    # A) was it found by appaloosa?
    # B) how many events overlap?
    # C) compare the total computed ED's.
    # D) compare the start and stop times.
    for i in range(0, len(fbdata[0,:])):
        # find any appaloosa flares that overlap the FBEYE start/stop times
        # 4 cases to catch: left/right overlaps, totally within, totally without
        x_ap = np.where(((apdata[0,:] <= fbdata[2,i]) & (apdata[1,:] >= fbdata[3,i])) |
                        ((apdata[0,:] >= fbdata[2,i]) & (apdata[0,:] <= fbdata[3,i])) |
                        ((apdata[1,:] >= fbdata[2,i]) & (apdata[1,:] <= fbdata[3,i]))
                        )


    return


def k2_mtg_plots(rerun=False, outfile='plotdata_v2.csv'):
    '''
    Some quick-and-dirty results from the 1st run for the K2 science meeting

    run from dir (at UW currently)
      /astro/store/tmp/jrad/nsf_flares/-HEX-ID-/

    can run as:
      from appaloosa import analysis
      analysis.k2_mtsg_plots()

    '''

    # have list of object ID's to run (from the Condor run)
    home = expanduser("~")
    dir = home + '/Dropbox/research_projects/nsf_flare_code/'
    obj_file = 'get_objects.out'

    kid = np.loadtxt(dir + obj_file, dtype='str',
                     unpack=True, skiprows=1, usecols=(0,))

    if rerun is True:

        # read in KIC file w/ colors
        kic_file = '../kic-phot/kic.txt.gz'
        kic_g, kic_r, kic_i  = np.genfromtxt(kic_file, delimiter='|', unpack=True,dtype=float,
                                            usecols=(5,6,7), filling_values=-99, skip_header=1)
        kicnum = np.genfromtxt(kic_file, delimiter='|', unpack=True,dtype=str,
                               usecols=(15,), skip_header=1)
        print('KIC data ingested')

        # (Galex colors too?)
        #

        # (list of rotation periods?)
        p_file = '../periods/Table_Periodic.txt'
        pnum = np.genfromtxt(p_file, delimiter=',', unpack=True,dtype=str, usecols=(0,),skip_header=1)
        prot = np.genfromtxt(p_file, delimiter=',', unpack=True,dtype=float, usecols=(4,),skip_header=1)
        print('Period data ingested')

        gi_color = np.zeros(len(kid)) - 99.
        ri_color = np.zeros(len(kid)) - 99.
        r_mag = np.zeros(len(kid)) - 99.
        periods = np.zeros(len(kid)) - 99.

        # keep a few different counts
        n_flares1 = np.zeros(len(kid))
        n_flares2 = np.zeros(len(kid))
        n_flares3 = np.zeros(len(kid))
        n_flares4 = np.zeros(len(kid))

        print('Starting loop through aprun files')
        for i in range(0, len(kid)):

            # read in each file in turn
            fldr = kid[i][0:3]
            outdir = 'aprun/' + fldr + '/'
            apfile = outdir + kid[i] + '.flare'

            try:
                data = np.loadtxt(apfile, delimiter=',', dtype='float',
                                  comments='#',skiprows=4)

                # if (i % 10) == 0:
                #     print(i, apfile, data.shape)

                # select "good" flares, count them
                '''
                t_start, t_stop, t_peak, amplitude, FWHM,
                duration, t_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1,
                flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur
                '''

                if data.ndim == 2:
                    # a quality cut
                    good1 = np.where((data[:,9] >= 10) &  # chisq
                                     (data[:,14] >= 0.1)) # ED
                    # a higher cut
                    good2 = np.where((data[:,9] >= 15) &  # chisq
                                     (data[:,14] >= 0.2)) # ED
                    # an amplitude cut
                    good3 = np.where((data[:,3] >= 0.005) & # amplitude
                                     (data[:,13] <= 0.05))   # KS_p
                    # everything cut
                    good4 = np.where((data[:,3] >= 0.005) &
                                     (data[:,13] <= 0.05) &
                                     (data[:,9] >= 15))

                    n_flares1[i] = len(good1[0])
                    n_flares2[i] = len(good2[0])
                    n_flares3[i] = len(good3[0])
                    n_flares4[i] = len(good4[0])

            except IOError:
                print(apfile + ' was not found!')

            # match whole object ID to colors
            km = np.where((kicnum == kid[i]))

            if (len(km[0])>0):
                gi_color[i] = kic_g[km[0]] - kic_i[km[0]]
                ri_color[i] = kic_r[km[0]] - kic_i[km[0]]

            pm = np.where((pnum == kid[i]))

            if (len(pm[0])>0):
                periods[i] = prot[pm[0]]

            if (i % 1000) == 0:
                outdata = np.asarray([n_flares1, n_flares2, n_flares3, n_flares4,
                                      gi_color, ri_color, periods])
                np.savetxt(outfile, outdata.T, delimiter=',')
                print(i, len(kid))

        # save to output lists
        outdata = np.asarray([n_flares1, n_flares2, n_flares3, n_flares4,
                              gi_color, ri_color, periods])
        np.savetxt(outfile, outdata.T, delimiter=',')

    else:
        print('Reading previous results')
        n_flares1, n_flares2, n_flares3, n_flares4, gi_color, ri_color, periods = \
            np.loadtxt(outfile, delimiter=',', dtype='float', unpack=True)



    # goal plots:
    # 1. color vs flare rate
    # 2. galex-g color vs flare rate
    # 3. g-r color vs period, point size/color with flare rate

    #---
    clr = np.log10(n_flares4)
    clr[np.where((clr > 3))] = 3
    clr[np.where((clr < 1))] = 1

    ss = np.argsort(clr)

    cut = np.where((clr[ss] >= 1.2))


    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 14

    plt.figure()
    plt.scatter(gi_color[ss][cut], periods[ss][cut], alpha=.7, lw=0,
                c=clr[ss][cut], cmap=plt.cm.afmhot_r, s=50)
    plt.xlabel('g-i (mag)')
    plt.ylabel('Rotation Period (days)')
    plt.yscale('log')
    plt.xlim((0,3))
    plt.ylim((0.1,100))
    cb = plt.colorbar()
    cb.set_label('log # flares')
    plt.show()


    #---
    clr = np.log10(n_flares4)
    clr[np.where((clr > 2.2))] = 2.2
    clr[np.where((clr < 1))] = 1

    bin2d, xx, yy,_ = binned_statistic_2d(gi_color, np.log10(periods),
                                          clr, statistic='median',
                                          range=[[-1,4],[-1,2]], bins=75)

    plt.figure()
    plt.imshow(bin2d.T, interpolation='nearest',aspect='auto', origin='lower',
               extent=(xx.min(),xx.max(),yy.min(),yy.max()),
               cmap=plt.cm.afmhot_r)

    plt.xlim((0,3))
    plt.ylim((-1,2))
    plt.xlabel('g-i (mag)')
    plt.ylabel('log Period (days)')
    cb = plt.colorbar()
    cb.set_label('log # flares (median)')
    plt.show()


    return


def compare_2_lit(outfile='plotdata_v2.csv'):
    '''
    Things to show:
    1) my flare rates compared to literature values for same stars
    2) literature values for same stars compared to eachother

    '''
    # read in MY flare sample
    n_flares1, n_flares2, n_flares3, n_flares4, gi_color, ri_color, periods = \
        np.loadtxt(outfile, delimiter=',', dtype='float', unpack=True)

    # read in the KIC numbers
    home = expanduser("~")
    dir = home + '/Dropbox/research_projects/nsf_flare_code/'
    obj_file = 'get_objects.out'

    kid = np.loadtxt(dir + obj_file, dtype='str',
                     unpack=True, skiprows=1, usecols=(0,))


    #######################
    # Compare 2 Literature
    #######################
    pitkin = dir + 'comparison_datasets/pitkin2014/table2.dat'
    pid,pnum = np.loadtxt(pitkin, unpack=True, usecols=(0,1),dtype='str')
    pnum = np.array(pnum, dtype='float')
    p_flares = np.zeros(len(kid))

    balona = dir + 'comparison_datasets/balona2015/table1.dat'
    bid_raw = np.loadtxt(balona, dtype='str', unpack=True, usecols=(0,), comments='#')
    bid,bnum = np.unique(bid_raw, return_counts=True)
    b_flares = np.zeros(len(kid))

    shibayama = dir + 'comparison_datasets/shibayama2013/apjs483584t7_mrt.txt'
    sid_raw = np.loadtxt(shibayama, dtype='str', unpack=True, usecols=(0,), comments='#')
    sid,snum = np.unique(sid_raw, return_counts=True)
    s_flares = np.zeros(len(kid))

    for i in range(0,len(kid)):
        x1 = np.where((kid[i] == pid))
        if len(x1[0]) > 0:
            p_flares[i] = pnum[x1]

        x2 = np.where((kid[i] == bid))
        if len(x2[0]) > 0:
            b_flares[i] = bnum[x2]

        x3 = np.where((kid[i] == sid))
        if len(x3[0]) > 0:
            s_flares[i] = snum[x3]




    plt.figure()
    plt.scatter(p_flares, n_flares4)
    plt.scatter(b_flares, n_flares4,c='r')
    plt.scatter(s_flares, n_flares4,c='g')
    plt.show()

    # plt.figure()
    # plt.scatter(p_flares, b_flares)
    # plt.xlabel('Pitkin')
    # plt.ylabel('Balona')
    # plt.show()

    return


def benchmark(objectid='gj1243_master', fbeyefile='gj1243_master_flares.tbl'):

    # run data thru the normal appaloosa flare-finding methodology
    appaloosa.RunLC(objectid, display=False, readfile=True)

    apfile = 'aprun/' + objectid[0:3] + '/' + objectid + '.flare'
    apdata = np.loadtxt(apfile, delimiter=',', dtype='float',
                        comments='#',skiprows=4)

    fbdata = np.loadtxt(fbeyefile, comments='#', dtype='float')
    '''
    t_start, t_stop, t_peak, amplitude, FWHM,
    duration, t_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1,
    flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur
    '''

    time = np.loadtxt(objectid + '.lc.gz', usecols=(1,), unpack=True)
    fbtime = np.zeros_like(time)
    aptime = np.zeros_like(time)

    for k in range(0,len(apdata[:,0])):
        x = np.where((time >= apdata[k,0]) & (time <= apdata[k,1]))
        aptime[x] = 1

    for k in range(0,len(fbdata[:,0])):
        x = np.where((time >= fbdata[k,0]) & (time <= fbdata[k,1]))
        fbtime[x] = 1

    # 4 things to measure:
    # num flares recovered / num in FBEYE
    # num time points recovered / time in FBEYE
    # % of FBEYE flare time recovered (time overlap)
    # % of specific FBEYE flares recovered at all (any overlap)


    n1 = float(len(apdata[:,0])) / float(len(fbdata[:,0]))
    n2 = float(np.sum(aptime)) / float(np.sum(fbtime))

    n3 = float(np.sum((fbtime == 1) & (aptime == 1))) / float(np.sum(fbtime == 1))
    n4_i = np.zeros_like(fbdata[:,0])


    for i in range(0, len(fbdata[:,0])):

        # find ANY overlap
        x_any = np.where(((apdata[:,0] <= fbdata[i,2]) & (apdata[:,1] >= fbdata[i,3])) |
                         ((apdata[:,0] >= fbdata[i,2]) & (apdata[:,0] <= fbdata[i,3])) |
                         ((apdata[:,1] >= fbdata[i,2]) & (apdata[:,1] <= fbdata[i,3]))
                         )
        if (len(x_any[0]) > 0):
            n4_i[i] = 1

    n4 = float(np.sum(n4_i)) / float(len(fbdata[:,0]))


    return (len(apdata[:,0]), len(fbdata[:,0]), n1, n2, n3, n4)


def PostCondor(flares='fakes.lis', outfile='condorout.dat'):
    '''
    This requires the data from the giant Condor run. Either in a tarball,
    or on the WWU cluster

    This code goes thru every .flare file and computes basic stats,
    which are returned in a new big file for plotting, comparing to the KIC, etc.

    '''

    # the fixed ED bins to sum the N flares over
    edbins = np.arange(-5, 5, 0.2)
    edbins = np.append(-10, edbins)
    edbins = np.append(edbins, 10)

    # generated via:
    # $ find aprun/* -name "*.flare" > flares.lis
    # can take a while for filesystem to do this...
    files = np.loadtxt(flares, dtype='str')

    fout = open(outfile, 'w')
    fout.write('# KICnumber, lsflag (0=llc,1=slc), dur (days), log(ed68), tot Nflares, [ Flares/Day (logEDbin) ] \n')

    for k in range(len(files)):
        # read in flare and fake results

        ffake = np.loadtxt(files[k], delimiter=',',
                           dtype='float',comments='#', ndmin=2)
        ''' t_min, t_max, std, nfake, amplmin, amplmax, durmin, durmax, ed68, ed90 '''

        dur = np.nanmax(ffake[:,1]) - np.nanmin(ffake[:,0])

        # pick flares in acceptable energy range (above ed68)
        ed68_all = ffake[:,8]
        x = np.where((ed68_all > - 10))
        if len(x[0]) > 0:
            edcut = np.nanmedian(ed68_all[x])
        else:
            edcut = 9e9

        if os.path.isfile(files[k].replace('.fake', '.flare')):
            fdata = np.loadtxt(files[k].replace('.fake', '.flare'),
                               delimiter=',', dtype='float',comments='#', ndmin=2)
            '''
            t_start, t_stop, t_peak, amplitude, FWHM,
            duration, t_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1,
            flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur,
            ed68_i, ed90_i
            '''

            # flares must be greater than the "average" ED cut, or the localized one
            ok_fl = np.where((fdata[:,14] >= edcut) |
                             (fdata[:,14] >= fdata[:,15])
                             )

            Nflares = len(ok_fl[0])

            ed_hist, _ = np.histogram(np.log10(fdata[ok_fl,14]), bins=edbins)

        else:
            # Produce stats, even if no flares pass cut. The 0's are important
            Nflares = 0
            ed_hist = np.zeros(len(edbins) - 1)


        ed_freq = ed_hist / dur

        # Stats to compute:
        # - # flares
        # - Freq. of Flares
        # - flare rate vs energy
        '''
        here's the problem... how do you combine data for diff months/qtr's
        where you have diff exp times, noise properties, completeness limits?

        need to put data into some kind of relative unit, scale by time or something
        this is prob easiest if binned on to fixed ED bins... but that a bit
        unsatisfying since would like to keep all data.
        '''

        # columns to output:
        # KICnumber, Long/Short flag, Duration (days), ED68cut, Total Nflares,
        #   [in K fixed bins of ED, the total # of flares]

        kicnum = files[k][files[k].find('kplr')+4 : files[k].find('-2')]
        edcut_out = str(np.log10(edcut))
        Nflares_out = str(Nflares)
        dur_out = str(dur)

        if (files[k].find('slc') == -1):
            lsflag = '1'
        else:
            lsflag = '0'

        outstring = kicnum + ', ' + lsflag + ', ' + dur_out + ', ' + edcut_out + ', ' + Nflares_out
        for i in range(len(ed_freq)):
            outstring = outstring + ', ' + str(ed_freq[i])

        fout.write(outstring + '\n')

    fout.close()

    return


def paper1_plots(condorfile='condorout.dat.gz',
                 kicfile='kic.txt.gz'):
    '''
    Make plots for the first paper, which describes the Kepler flare sample.

    The Condor results are aggregated from PostCondor() above

    '''

    # read in KIC file
    # http://archive.stsci.edu/pub/kepler/catalogs/ <- data source
    # http://archive.stsci.edu/kepler/kic10/help/quickcol.html <- info

    kicdata = pd.read_csv(kicfile, delimiter='|')
    # need KICnumber, gmag, imag, logg (for cutting out crap only)
    # kicnum_k = kicdata['kic_kepler_id']


    fdata = pd.read_table(condorfile, delimiter=',', skiprows=1, header=None)
    # need KICnumber, Flare Freq data in units of ED
    kicnum_c = fdata.iloc[:,0].unique()


    # match two dataframes on KIC number
    # bigdata = pd.merge(kicdata, kicnum_c, how='inner',
    #                    left_on='kic_kepler_id', right_on=0)

    bigdata = kicdata[kicdata['kic_kepler_id'].isin(kicnum_c)]

    Lkp_uniq = energies(bigdata['kic_gmag'], bigdata['kic_imag'])

    # goal plots:
    # color vs flare rate
    # combined FFD for a couple months, then for all the months of 1 star
    # combined FFD for all of a couple stars in same mass range



    return


def energies(gmag, imag, isochrone='0.5gyr.dat'):
    '''
    Compute the quiescent energy for every star. Use the KIC (g-i) color,
    with an isochrone, get the absolute Kepler mag for each star, and thus
    the distance & luminosity.

    Isochrone is a 0.5 Gyr track from the Padova CMD v2.7
    http://stev.oapd.inaf.it/cgi-bin/cmd_2.7

    Kepler and Sloan phot system both in AB mags.

    Returns
    -------
    Quiescent Luminosities in the Kepler band
    '''

    # read in Padova isochrone file
    # note, I've cheated and clipped this isochrone to only have the
    # Main Sequence, up to the blue Turn-Off limit.
    dir = dir = os.path.dirname(os.path.realpath(__file__)) + '/misc/'

    Mkp, Mg, Mi = np.loadtxt(dir + isochrone, comments='#',
                             unpack=True, usecols=(8,9,11))

    # To match observed data to the isochrone, cheat:
    # e.g. Find interpolated g, given g-i. Same for Kp
    Mgi = (Mg-Mi)
    ss = np.argsort(Mgi) # needs to be sorted for interpolation

    Mkp_o = np.interp((gmag-imag), Mgi[ss], Mkp[ss])
    Mg_o = np.interp((gmag-imag), Mgi[ss], Mg[ss])

    dist = _DistModulus(gmag, Mg_o)

    dm = (gmag - Mg_o)
    # returns Flux [erg/s/cm^2]
    F_kp = _ABmag2flux(Mkp_o - dm)

    # again, classic bread/butter right here,
    # change Flux to Luminosity [erg/s]
    L_kp = F_kp * (4.0 * np.pi * dist**2.0)

    # !! Should be able to include errors on (g-i), propogate to
    #    errors on Distance, and thus lower error limit on L_kp !!

    return L_kp


'''
  let this file be called from the terminal directly. e.g.:
  $ python analysis.py
'''
if __name__ == "__main__":
    import sys
    print(benchmark(objectid=sys.argv[1], fbeyefile=sys.argv[2]))
