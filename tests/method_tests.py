'''
Clumsy tests I am writing to help evaluate what is going on in each method.
'''

import appaloosa.appaloosa as ap
import numpy as np
import matplotlib.pyplot as plt

def _mockdata(baseline=1e5, sigma=10,
              t0=100, t1=1000, dt=0.02):

    time = np.arange(t0, t1, dt)
    noise = np.random.random(len(time)) * sigma
    flux = np.ones_like(time) * baseline + noise
    error = np.ones_like(time) * sigma
    flags = np.zeros_like(time)

    return time, flux, error, flags


def TestMultiFind():
    time, flux, error, flags = _mockdata()

    # plt.figure()
    # plt.plot(time, flux)
    # plt.show()

    istart, istop, flux_model = ap.MultiFind(time, flux, error, flags, oldway=False)


    if (len(istart) != 0):
        c1 = False
        print('Test 1: FAIL. len(istart) = ' + str(len(istart)) + ', expected 0')
        print(istart)
    else:
        c1 = True

    if (len(istop) != 0):
        c2 = False
        print('Test 2: FAIL. len(istop) = ' + str(len(istop)) + ', expected 0')
        print(istop)
    else:
        c2 = True


    if (len(flux_model) != len(flux)):
        c3 = False
        print('Test 3: FAIL. len(flux_model) = ' + str(len(flux_model)) + ', expected ' + str(len(flux)))

    else:
        c3 = True


    return (c1, c2, c3)