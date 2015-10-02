'''
Routines to do analysis on the appaloosa flare finding runs. Including
  - plots for the paper
  - check against other sample of flares from Kepler
  - completeness and efficiency tests against FBEYE results
  - completeness and efficiency tests against fake data (?)
'''

import numpy as np
import matplotlib.pyplot as plt


def fbeye_compare(apfile='9726699.flare', fbeye_file='9726699.fbeye'):
    '''
    compare flare finding and properties between appaloosa and FBEYE
    '''

    # read the aprun output
    #  t_start, t_stop, t_peak, amplitude, FWHM, duration, t_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1, flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur
    apdata = np.loadtxt(apfile, delimiter=',', dtype='float', skiprows=5, comments='#')

    # read the FBEYE output

    return