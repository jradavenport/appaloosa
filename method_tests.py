'''
Clumsy tests I am writing to help evaluate what is going on in each method.
'''

import appaloosa as ap
from analysis import benchmark
from aflare import aflare1
import numpy as np
import datetime
import warnings
import matplotlib.pyplot as plt

def _mockdata(baseline=1e5, sigma=10,
              t0=100, t1=200, dt=0.02):

    time = np.arange(t0, t1, dt)
    noise = np.random.random(len(time)) * sigma
    flux = np.ones_like(time) * baseline + noise
    error = np.ones_like(time) * sigma
    flags = np.zeros_like(time)

    return time, flux, error, flags


def TestMultiFind():

    # set this to silence bad fit warnings from polyfit
    warnings.simplefilter('ignore', np.RankWarning)

    print("TESTING MOCK DATA")
    time, flux, error, flags = _mockdata()

    istart, istop, flux_model = ap.MultiFind(time, flux, error, flags)


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


def TestFINDflare():
    ####
    # TEST W/ NO FLARES

    time, flux, error, flags = _mockdata()
    istart, istop = ap.FINDflare(flux, error)
    if (len(istart) != 0):
        c1 = False
        print('TEST FAIL: len(istart) = ' + str(len(istart)) + ', expected 0')
        print(istart)
    else:
        c1 = True
    if (len(istop) != 0):
        c2 = False
        print('TEST FAIL: len(istop) = ' + str(len(istop)) + ', expected 0')
        print(istop)
    else:
        c2 = True

    binout = ap.FINDflare(flux, error, returnbinary=True)
    if (sum(binout) > 0):
        c3 = False
        print('Test FAIL: sum(binout) > 0')
    else:
        c3 = True


    ####
    # TEST W/ STEP FUNCTION FLARE

    time = np.arange(40)
    flux = np.ones_like(time)
    fl1 = np.array([0,0,0,0,0,0,0,0,0,0,
                    1,1,1,1,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,0,0,0])
    error = np.ones_like(time) * 1e-5

    print(np.std(flux+fl1))
    istart, istop = ap.FINDflare(flux + fl1, error, debug=False)

    if (len(istart) != 1):
        c4 = False
        print('TEST FAIL: len(istart) = ' + str(len(istart)) + ', expected 1')
        print(istart)
    else:
        c4 = True
    if (len(istop) != 1):
        c5 = False
        print('TEST FAIL: len(istop) = ' + str(len(istop)) + ', expected 1')
        print(istop)
    else:
        c5 = True

    binout = ap.FINDflare(flux + fl1, error, returnbinary=True)
    if (sum(binout) == 0):
        c6 = False
        print('Test FAIL: sum(binout) == 0')
        print(binout)
    else:
        c6 = True


    ####
    # TEST W/ STEP REALISTIC BIG FLARE
    # fl1 = aflare1(time, 110.0, 0.06, 150.0)

    return (c1, c2, c3, c4, c5, c6)


if __name__ == "__main__":
    print(str(datetime.datetime.now()))
    rec = TestMultiFind()
    print(rec)

    print(str(datetime.datetime.now()))
    print("RUNNING GJ1243 BENCHMARK")
    rec2 = benchmark()
    print(rec2)

    print(str(datetime.datetime.now()))
    rec3 = TestFINDflare()
    print(rec3)

    print("FINISHED TESTING")
    print(str(datetime.datetime.now()))