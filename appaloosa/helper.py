import numpy as np

def EquivDur(time, flux):
    '''
    Compute the Equivalent Duration of an event. This is simply the area
    under the flare, in relative flux units.

    Flux must be array in units of zero-centered RELATIVE FLUX

    Time must be array in units of DAYS

    Output has units of SECONDS
    '''

    p = np.trapz(flux, x=(time * 60.0 * 60.0 * 24.0))
    return p
