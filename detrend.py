'''
Hold the detrending method(s) to use.

1) Write a sliding local polynomial smoother
2) translate softserve (?)

'''
import numpy as np

def polysmooth(time, flux, error):
    smo = flux
    return smo