
import pandas as pd
import glob
import os
from appaloosa import RunLC
import time
import warnings
warnings.filterwarnings('ignore')

path = 'test_suite/'
 
#include k2varcat later?
test_suite = {#'kplr':'kplr009726699-2009350155506_llc.fits',
              'ktwo':'ktwo211121743-c04_llc.fits',
             # 'everest':'hlsp_everest_k2_llc_246199087-c12_kepler_v2.0_lc.fits',
             # 'k2sc':'hlsp_k2sc_k2_llc_211121743-c04_kepler_v2_lc.fits',
             # 'vdb':'hlsp_k2sff_k2_lightcurve_220132548-c08_kepler_v1_llc-default-aper.txt',
             }

for key, value in test_suite.items():
    print('This is {}. Injections started! {}'.format(key,value))
    RunLC('{}{}'.format(path,value), dbmode=key, display=True, 
          debug=False, dofake=True, nfake=10,
          iterations=100,) 
    print(time.clock())

