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


def k2_mtg_plots():
    '''
    Some quick-and-dirty results from the 1st run for the K2 science meeting

    run from dir (at UW currently)
      /astro/store/tmp/jrad/nsf_flares/-HEX-ID-/

    can run as:
      from appaloosa import analysis
      analysis.k2_mtg_plots()

    '''

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

    # have list of object ID's to run (from the Condor run)
    home = expanduser("~")
    dir = home + '/Dropbox/research_projects/nsf_flare_code/'
    obj_file = 'get_objects.out'

    kid = np.loadtxt(dir + obj_file, dtype='str',
                     unpack=True, skiprows=1, usecols=(0,))

    gi_color = np.zeros(len(kid)) - 99.
    ri_color = np.zeros(len(kid)) - 99.
    n_flares = np.zeros(len(kid))

    periods = np.zeros(len(kid)) - 99.

    print('Starting loop through aprun files')
    for i in range(0, len(kid)):

        if (i % 10) == 0:
            print(i)

        # read in each file in turn
        fldr = kid[i][0:3]
        outdir = 'aprun/' + fldr + '/'
        data = np.loadtxt(outdir + kid[i], delimiter=',', dtype='float',
                          comments='#',skiprows=4)

        # select "good" flares, count them
        '''
        t_start, t_stop, t_peak, amplitude, FWHM,
        duration, t_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1,
        flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur
        '''
        good = np.where((data[:,9] >= 10) & # chisq
                        (data[:,14] >= 0.1)) # ED

        n_flares[i] = len(good[0])

        # match whole object ID to colors
        km = np.where((kicnum == kid[i]))

        if (len(km[0])>0):
            gi_color[i] = kic_g[km[0]] - kic_i[km[0]]
            ri_color[i] = kic_r[km[0]] - kic_i[km[0]]

        pm = np.where((pnum == kid[i]))

        if (len(pm[0])>0):
            periods[i] = prot[pm[0]]

    # save to output lists

    outfile = 'plotdata_v1.csv'
    outdata = np.asarray([n_flares, gi_color, ri_color, periods])
    np.savetxt(outfile, outdata, delimiter=',')



    # goal plots:
    # 1. g-r color vs flare rate
    plt.figure()
    plt.scatter(gi_color, n_flares, alpha=0.5)
    plt.xlim((-5,10))
    plt.ylim((0.1,20))
    plt.yscale('log')
    plt.show()

    # 2. galex-g color vs flare rate
    # 3. g-r color vs period, point size/color with flare rate

    return


'''
  let this file be called from the terminal directly. e.g.:
  $ python analysis.py
'''
if __name__ == "__main__":
    k2_mtg_plots()
