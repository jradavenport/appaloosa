
import pandas as pd
import glob
import os
from appaloosa import RunLC
import time
import warnings
warnings.filterwarnings('ignore')

path = 'test_suite/'

#include k2varcat later?
test_suite = {'kplr':('{}kplr009726699-2009350155506_llc.fits'.format(path),'davenport'),
              'ktwo':('{}ktwo211121743-c04_llc.fits'.format(path),'davenport'),
              'everest':('{}hlsp_everest_k2_llc_246199087-c12_kepler_v2.0_lc.fits'.format(path),'davenport'),
              'k2sc':('{}hlsp_k2sc_k2_llc_211099743-c04_kepler_v2_lc.fits'.format(path),'median'),
              'vdb':('{}hlsp_k2sff_k2_lightcurve_220132548-c08_kepler_v1_llc-default-aper.txt'.format(path),'davenport'),
              'random':('random','median'),
              'test':('{}testLC'.format(path),'median')
              }

#iteration need to be an even number bc wiener would not work otherwise???
for key, value in test_suite.items():
    print('This is {}. Injections started! {}'.format(key,value))

    RunLC(file=value[0], dbmode=key, display=False,
          debug=False, dofake=True, fakefreq=0.5,mode=value[1],
          iterations=20,)
    print(time.time())

print('Find output files here: /home/USERNAME/research/appaloosa/aprun')
