from astropy.io import fits
import pandas as pd
import numpy as np
from os.path import expanduser
from random import choice as choose_random_item
import os
from lightkurve import KeplerTargetPixelFile
from lightkurve.mast import ArchiveError
#load KeplerTargetPixelFile
# flatten with k2SFF i.e. create LightCurveFile
# transform to FlareLightCurveFile
# define methods



def Get(mode, file='', objectid='', win_size=3):

    '''

    Call a get function depending on the mode,
    processes loading steps common to all modes.
    Generates error from short time median scatter
    if not given elsewhere.

    Parameters:
    ------------
    mode: str
        type of light curve, e.g. EVEREST LC, Vanderburg LC,
        raw MAST .fits file, random K2
    file: '' or str
        lightcurve file location
    win_size: 3 or int
        window size for scatter generator

    Returns:
    ------------
    lc: pandas DataFrame
        light curve

    '''

    def GetObjectID(mode):
        if mode == 'kplr':
            return str(int( file[file.find('kplr')+4:file.find('-')]))
        elif mode == 'ktwo':
            return str(int( file[file.find('ktwo')+4:file.find('-')] ))
        elif mode == 'vdb':
            str(int(file[file.find('lightcurve_')+11:file.find('-')]))
        elif mode == 'everest':
            return str(int(file[file.find('everest')+15:file.find('-')]))
        elif mode == 'k2sc':
            return str(int(file[file.find('k2sc')+12:file.find('-')]))
        elif mode == 'txt':
            return file[0:3]
        elif mode == 'csv':
            return '0000'
        elif mode == 'random':
            return 'random'

    def GetOutDir(mode):

        home = expanduser("~")
        outdir = '{}/research/appaloosa/aprun/{}/'.format(home,mode)
        if not os.path.isdir(outdir):
            try:
                os.makedirs(outdir)
            except OSError:
                pass
        return outdir

    def GetOutfile(mode,file='random'):

        if mode == 'everest':
            return GetOutDir(mode) + file[file.find('everest')+15:]

        elif mode == 'k2sc':
            return GetOutDir(mode) + file[file.find('k2sc')+12:]

        elif mode == 'kplr':
            return GetOutDir(mode) + file[file.find('kplr')+4:]

        elif mode in ('vdb','csv'):
            return GetOutDir(mode) + file[file.find('lightcurve_')+11:]

        elif mode in ('ktwo'):
            return GetOutDir(mode) + file[file.find('ktwo')+4:]

        elif mode in ('txt','random','test'):
            return GetOutDir(mode) + file[-6:]

    modes = {'kplr': GetLCfits,
             'ktwo': GetLCfits,
             'vdb': GetLCvdb,
             'everest': GetLCeverest,
             'k2sc': GetLCk2sc,
             'txt': GetLCtxt,
             'csv': GetLCvdb,
             'random':GetLClightkurve}

    if mode == 'test':
        lc = pd.read_csv('test_suite/test/testlc.csv').dropna(how='any')
    else:
        lc = modes[mode](file=file).dropna(how='any')

    t = lc.time.values
    dt = np.nanmedian(t[1:] - t[0:-1])
    if (dt < 0.01):
        dtime = 54.2 / 60. / 60. / 24.
    else:
        dtime = 30 * 54.2 / 60. / 60. / 24.

    lc['exptime'] = dtime
    lc['qtr'] = 0

    if 'flags' not in lc.columns:
        lc['flags'] = 0

    if 'error' not in lc.columns:
        lc['error'] = np.nanmedian(lc.flux_raw.rolling(win_size, center=True).std())

    print(GetOutfile(mode, file=file))
    return GetOutfile(mode, file=file), GetObjectID(mode), lc

def GetLCfits(file=''):

    '''
    Parameters
    ----------
    file : light curve file location for a MAST archive .fits file

    Returns
    -------
    lc: light curve DataFrame with columns [time, quality, flux_raw, error]
    '''
    hdu = fits.open(file)
    data_rec = hdu[1].data
    lc = pd.DataFrame({'time':data_rec['TIME'].byteswap().newbyteorder(),
                      'flux_raw':data_rec['SAP_FLUX'].byteswap().newbyteorder(),
                      'error':data_rec['SAP_FLUX_ERR'].byteswap().newbyteorder(),
                      'flags':data_rec['SAP_QUALITY'].byteswap().newbyteorder()})


    return lc


def GetLCvdb(file=''):

    '''
    Parameters
    ----------
    file : light curve file location for a Vanderburg de-trended .txt file

    Returns
    -------
    lc: light curve DataFrame with columns [time, flux_raw]
    '''

    lc = pd.read_csv(file,index_col=False)
    lc.rename(index=str,
              columns={'BJD - 2454833': 'time',' Corrected Flux':'flux_raw'},
              inplace=True,
              )
    return lc


def GetLCeverest(file=''):

    '''
    Parameters
    ----------
    file : light curve file location for a Vanderburg de-trended .txt file

    Returns
    -------
    lc: light curve DataFrame with columns [time, flux_raw]
    '''

    hdu = fits.open(file)
    data_rec = hdu[1].data
    lc = pd.DataFrame({'time':np.array(data_rec['TIME']).byteswap().newbyteorder(),
                      'flux_raw':np.array(data_rec['FLUX']).byteswap().newbyteorder(),})

    #keep the outliers... for now
    #lc['quality'] = data_rec['OUTLIER'].byteswap().newbyteorder()

    return lc


def GetLCk2sc(file=''):


    '''
    Parameters
    ----------
    file : light curve file location for a Vanderburg de-trended .txt file
    Returns
    -------
    lc: light curve DataFrame with columns [time, flux_raw]
    '''

    hdu = fits.open(file)
    data_rec = hdu[1].data

    lc = pd.DataFrame({'time':np.array(data_rec['time']).byteswap().newbyteorder(),
                      'flux_raw':np.array(data_rec['flux']).byteswap().newbyteorder(),})
                      #'error':np.array(data_rec['error']).byteswap().newbyteorder(),})
    hdu.close()
    del data_rec
    #keep the outliers... for now
    #lc['quality'] = data_rec['OUTLIER'].byteswap().newbyteorder()

    return lc

def GetLCtxt(file=''):

    '''
    Parameters
    ----------
    file : '' or
    light curve file location for a basic .txt file

    Returns
    -------
    lc: light curve DataFrame with columns [time, flux_raw, error]
    '''
    lc = pd.read_csv(file,
                     index_col=False,
                     usecols=(0,1,2),
                     skiprows=1,
                     header = None,
                     names = ['time','flux_raw','error'])

    return lc

def GetLClightkurve(file='',**kwargs):

    '''
    Construct a light curve from either
    - a local KeplerTargetPixelFile, or
    - a random K2 KeplerTargetPixelFile from the archive
    using the lightkurve built-in correct function.

    Parameters
    ----------
    file : '' or str
        light curve file path. Default will download random file from archive.
    **kwargs : dict
        Keyword arguments to that will be passed to the KeplerTargetPixelFile
        constructor.

    Returns
    -------
    lc: pandas DataFrame
        light curve with columns ['time', 'flux_raw', 'error']
    '''
    if file == '':
        print('Choose a random LC from the archives...')
        idlist = pd.read_csv('stars_shortlist/share/helpers/GO_all_campaigns_to_date.csv',
                              usecols=['EPIC ID'])

        ID = choose_random_item(idlist['EPIC ID'].values)
        tpf = None
        try:
            tpf = KeplerTargetPixelFile.from_archive(ID, cadence='long')
        except ArchiveError:
            print('EPIC {} was observed during several campaigns.'
                  '\nChoose the earliest available.'.format(ID))
            C = 0
            while C < 20:
                print(C)
                try:
                    tpf = KeplerTargetPixelFile.from_archive(ID, cadence='long',
                                                             campaign=C)
                except ArchiveError:
                    C += 1
                    pass
                if tpf != None:
                    break
    else:
        tpf = KeplerTargetPixelFile(file, quality_bitmask='default')

    lc = tpf.to_lightcurve(method='aperture')
    lc = lc.correct(windows=20)
    LC = pd.DataFrame({'flux_raw': lc.flux,
                        'time':np.copy(lc.time).byteswap().newbyteorder(),
                        'error':lc.flux_err,
                        'flags':np.copy(lc.quality).byteswap().newbyteorder(),
    })

    return LC

# UNUSED, UNTESTED, DELETE?
# def OneCadence(data):
#     '''
#     Within each quarter of data from the database, pick the data with the
#     fastest cadence. We want to study 1-min if available. Don't want
#     multiple cadence observations in the same quarter, bad for detrending.
#
#     Parameters
#     ----------
#     data : numpy array
#         the result from MySQL database query, using the getLC() function
#
#     Returns
#     -------
#     Data array
#
#     '''
#     # get the unique quarters
#     qtr = data[:,0]
#     cadence = data[:,5]
#     uQtr = np.unique(qtr)
#
#     indx = []
#
#     # for each quarter, is there more than one cadence?
#     for q in uQtr:
#         x = np.where( (np.abs(qtr-q) < 0.1) )
#
#         etimes = np.unique(cadence[x])
#         y = np.where( (cadence[x] == min(etimes)) )
#
#         indx = np.append(indx, x[0][y])
#
#     indx = np.array(indx, dtype='int')
#
#     data_out = data[indx,:]
#     return data_out
#
# def GetLCdb(objectid, type='', readfile=False,
#           savefile=False, exten = '.lc.gz',
#           onecadence=False):
#     '''
#     Retrieve the lightcurve/data from the UW database.
#
#     Parameters
#     ----------
#     objectid
#     type : str, optional
#         If either 'slc' or 'llc' then just get 1 type of cadence. Default
#         is empty, so gets both
#     readfile : bool, optional
#         Default is False
#     savefie : bool, optional
#         Default is False
#     exten : str, optional
#         Extension for file saving. Default is '.lc.gz'
#     onecadence : bool, optional
#         For quarters with Long and Short cadence, remove the Long data.
#         Default is False. Can be done later to the data output
#
#     Returns
#     -------
#     numpy array with many columns:
#         QUARTER, TIME, PDCSAP_FLUX, PDCSAP_FLUX_ERR,
#         SAP_QUALITY, LCFLAG, SAP_FLUX, SAP_FLUX_ERR
#     '''
#

    # try:
    #   import MySQLdb
    #   haz_mysql = True
    # except ImportError:
    #   haz_mysql = False

#     isok = 0 # a flag to check if database returned sensible answer
#     ntry = 0
#
#     if readfile is True:
#         # attempt to find file in working dir
#         if os.path.isfile(str(objectid) + exten):
#             data = np.loadtxt(str(objectid) + exten)
#             isok = 101
#
#
#     while isok<1:
#         # this holds the keys to the db... don't put on github!
#         auth = np.loadtxt('auth.txt', dtype='string')
#
#         # connect to the db
#         db = MySQLdb.connect(passwd=auth[2], db="Kepler",
#                              user=auth[1], host=auth[0])
#
#         query = 'SELECT QUARTER, TIME, PDCSAP_FLUX, PDCSAP_FLUX_ERR, ' +\
#                 'SAP_QUALITY, LCFLAG, SAP_FLUX, SAP_FLUX_ERR ' +\
#                 'FROM Kepler.source WHERE KEPLERID=' + str(objectid)
#
#         # only get SLC or LLC data if requested
#         if type=='slc':
#             query = query + ' AND LCFLAG=0 '
#         if type=='llc':
#             query = query + ' AND LCFLAG=1 '
#
#         query = query + ' ORDER BY TIME;'
#
#         # make a cursor to the db
#         cur = db.cursor()
#         cur.execute(query)
#
#         # get all the data
#         rows = cur.fetchall()
#
#         # convert to numpy data array
#         data = np.asarray(rows)
#
#         # close the cursor to the db
#         cur.close()
#
#         # make sure the database returned the actual light curve
#         if len(data[:,0] > 99):
#             isok = 10
#         # only try 10 times... shouldn't ever need this limit
#         if ntry > 9:
#             isok = 2
#         ntry = ntry + 1
#         time.sleep(10) # give the database a breather
#
#     if onecadence is True:
#         data_raw = data.copy()
#         data = OneCadence(data_raw)
#
#     if savefile is True:
#         # output a file in working directory
#         np.savetxt(str(objectid) + exten, data)
#
#     #---------------------------------------------------
#         # data columns are:
#         # QUARTER, TIME, PDCFLUX, PDCFLUX_ERR, SAP_QUALITY, LCFLAG, SAPFLUX, SAPFLUX_ERR
#
#         qtr = data[:,0]
#         time = data[:,1]
#         lcflag = data[:,4] # actual SAP_QUALITY
#
#         exptime = data[:,5] # actually the LCFLAG
#         exptime[np.where((exptime < 1))] = 54.2 / 60. / 60. / 24.
#         exptime[np.where((exptime > 0))] = 30 * 54.2 / 60. / 60. / 24.
#
#         if ftype == 'sap':
#             flux_raw = data[:,6]
#             error = data[:,7]
#         else: # for PDC data
#             flux_raw = data[:,2]
#             error = data[:,3]
#
#         # put flare output in to a set of subdirectories.
#         # use first 3 digits to help keep directories to ~1k files
#         fldr = objectid[0:3]
#         outdir = 'aprun/' + fldr + '/'
#         if not os.path.isdir(outdir):
#             try:
#                 os.makedirs(outdir)
#             except OSError:
#                 pass
#         # open the output file to store data on every flare recovered
#         outfile = outdir + objectid
#
#     return data
