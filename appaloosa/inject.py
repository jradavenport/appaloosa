
import pandas as pd
import glob
import os
from appaloosa import RunLC
import time
import warnings
warnings.filterwarnings('ignore')

path = 'test_suite/'

#include k2varcat later?
test_suite = {'kplr':('kplr009726699-2009350155506_llc.fits',3),
              #'ktwo':('ktwo211121743-c04_llc.fits',3),
              #'everest':('hlsp_everest_k2_llc_246199087-c12_kepler_v2.0_lc.fits',3),
              #'k2sc':('hlsp_k2sc_k2_llc_211099743-c04_kepler_v2_lc.fits',0),
              #'vdb':('hlsp_k2sff_k2_lightcurve_220132548-c08_kepler_v1_llc-default-aper.txt',3),
              }

#iteration need to be an even number bc wiener would not work otherwise???
for key, value in test_suite.items():
    print('This is {}. Injections started! {}'.format(key,value))
    RunLC('{}{}'.format(path,value[0]), dbmode=key, display=True,
          debug=True, dofake=True, nfake=20,mode=value[1],
          iterations=2,)
    print(time.clock())

print('Find output files here: /home/ekaterina/research/appaloosa/aprun')
