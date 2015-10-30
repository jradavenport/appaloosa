'''
Routines to do analysis on the appaloosa flare finding runs. Including
  - plots for the paper
  - check against other sample of flares from Kepler
  - completeness and efficiency tests against FBEYE results
  - completeness and efficiency tests against fake data (?)
'''

import numpy as np
import matplotlib.pyplot as plt
import appaloosa
from scipy.stats import binned_statistic_2d
from os.path import expanduser


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
                        ((apdata[0,:] > fbdata[2,i]) & (apdata[0,:] < fbdata[3,i])) |
                        ((apdata[1,:] > fbdata[2,i]) & (apdata[1,:] < fbdata[3,i]))
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

    clr = np.log10(n_flares4)
    clr[np.where((clr > 3))] = 3
    clr[np.where((clr < 1))] = 1

    plt.figure()
    plt.scatter(gi_color, periods, alpha=.7, lw=0, c=clr, cmap=plt.cm.afmhot_r)
    plt.xlabel('g-i (mag)')
    plt.ylabel('Period (days)')
    plt.yscale('log')
    plt.xlim((0,3))
    plt.ylim((0.1,100))
    cb = plt.colorbar()
    cb.set_label('log # flares')
    plt.show()


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
    # read in MY flare sample
    n_flares1, n_flares2, n_flares3, n_flares4, gi_color, ri_color, periods = \
        np.loadtxt(outfile, delimiter=',', dtype='float', unpack=True)

    # read in the KIC numbers
    home = expanduser("~")
    dir = home + '/Dropbox/research_projects/nsf_flare_code/'
    obj_file = 'get_objects.out'

    kid = np.loadtxt(dir + obj_file, dtype='str',
                     unpack=True, skiprows=1, usecols=(0,))


    ####################
    # Compare 2 Literature
    ####################
    pitkin = dir + 'comparison_datasets/pitkin2014/table2.dat'
    pid,pnum = np.loadtxt(pitkin, unpack=True, usecols=(0,1),dtype='str')
    pnum = np.array(pnum, dtype='float')

    p_flares = np.zeros(len(kid))

    for i in range(0,len(pid)):
        x = np.where((kid == pid[i]))
        if len(x[0]) > 0:
            p_flares[x] = pnum[i]

    plt.figure()
    plt.scatter(p_flares, n_flares4)
    plt.show()

    return



'''
  let this file be called from the terminal directly. e.g.:
  $ python analysis.py
'''
if __name__ == "__main__":
    k2_mtg_plots()
