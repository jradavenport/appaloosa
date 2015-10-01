'''
Routines to do analysis on the appaloosa flare finding runs. Including
  - plots for the paper
  - check against other sample of flares from Kepler
  - completeness and efficiency tests against FBEYE results
  - completeness and efficiency tests against fake data (?)
'''

import numpy as np


def fbeye_compare(objectid='9726699', fbeye_file=''):
    '''
    compare flare finding and properties between appaloosa and FBEYE
    '''

    return