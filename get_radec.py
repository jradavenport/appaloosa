import pandas as pd
import numpy as np

condorfile='condorout.dat.gz'
kicfile='kic.txt.gz'

kicdata = pd.read_csv(kicfile, delimiter='|')
fdata = pd.read_table(condorfile, delimiter=',', skiprows=1, header=None)
''' KICnumber, lsflag (0=llc,1=slc), dur [days], log(ed68), tot Nflares, sum ED, [ Flares/Day (logEDbin) ] '''

# need KICnumber, Flare Freq data in units of ED
kicnum_c = fdata.iloc[:,0].unique()

# just the KIC data that is in the Condor output
bigdata = kicdata[kicdata['kic_kepler_id'].isin(kicnum_c)]



##### I'm sure there's a better way to do this with Pandas...

ra = np.zeros_like(kicnum_c)-99.
dec = np.zeros_like(kicnum_c)-99.

# loooooop thru all the KIC objects and dump ra,dec
for k in range(len(kicnum_c)):
    # find this star in the KIC data
    mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[k]))
    if len(mtch[0])>0:
        ra[k] = bigdata['kic_degree_ra'].values[mtch][0]
        dec[k] = bigdata['kic_dec'].values[mtch][0]

dfout = pd.DataFrame(data={'kicnum':kicnum_c,
                           'ra':ra, 'dec':dec})
dfout.to_csv('kic_radec.csv')