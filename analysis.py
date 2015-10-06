'''
Routines to do analysis on the appaloosa flare finding runs. Including
  - plots for the paper
  - check against other sample of flares from Kepler
  - completeness and efficiency tests against FBEYE results
  - completeness and efficiency tests against fake data (?)
'''

import numpy as np
import matplotlib.pyplot as plt


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