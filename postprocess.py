import numpy as np
import os


def PostCondor(flares='fakes.lis', outfile='condorout.dat'):
    '''
    This requires the data from the giant Condor run.

    Run on WWU workstation in dir: ~/research/kepler-flares/
    or on cluster in dir: ~/data/HEXRUNID/

    This code goes thru every .flare file and computes basic stats,
    which are returned in a new big file for plotting, comparing to the KIC, etc.

    the list of "fakes" is generated on the WWU iMac like so:
    find 0x56ff1094_aprun/* -name "*.fake" > 0x56ff1094_fakes.lis

    '''

    # the fixed ED bins to sum the N flares over
    edbins = np.arange(-5, 5, 0.2)
    edbins = np.append(-10, edbins)
    edbins = np.append(edbins, 10)

    # generated via:
    # $ find aprun/* -name "*.flare" > flares.lis
    # can take a while for filesystem to do this...
    files = np.loadtxt(flares, dtype='str')

    fout = open(outfile, 'w')
    fout.write('# KICnumber, lsflag (0=llc,1=slc), dur [days], log(ed68), tot Nflares, sum ED, sum ED err, [ Flares/Day (logEDbin) ] \n')

    for k in range(len(files)):
        # read in flare and fake results

        ffake = np.loadtxt(files[k], delimiter=',',
                           dtype='float',comments='#', ndmin=2)
        ''' t_min, t_max, std, nfake, amplmin, amplmax, durmin, durmax, ed68, ed90 '''

        dur = np.nanmax(ffake[:,1]) - np.nanmin(ffake[:,0])

        # pick flares in acceptable energy range (above ed68)
        ed68_all = ffake[:,8]
        x = np.where((ed68_all > - 10))
        if len(x[0]) > 0:
            edcut = np.nanmedian(ed68_all[x])
        else:
            edcut = 9e9

        if os.path.isfile(files[k].replace('.fake', '.flare')):
            fdata = np.loadtxt(files[k].replace('.fake', '.flare'),
                               delimiter=',', dtype='float',comments='#', ndmin=2)
            '''
            t_start, t_stop, t_peak, amplitude, FWHM,
            duration (days), t_peak_aflare1, t_FWHM_aflare1, amplitude_aflare1,
            flare_chisq, KS_d_model, KS_p_model, KS_d_cont, KS_p_cont, Equiv_Dur,
            ed68_i, ed90_i
            '''

            # flares must be greater than the "average" ED cut, or the localized one
            ok_fl = np.where((fdata[:,14] >= edcut) |
                             (fdata[:,14] >= fdata[:,15])
                             )

            Nflares = len(ok_fl[0])

            ed_hist, _ = np.histogram(np.log10(fdata[ok_fl,14]), bins=edbins)

            # the errors (from chi sq) are approximately:
            # sigma_ED ~ sqrt( ED^2 / N / chisq )
            if (files[k].find('slc') == -1):
                expt = 1./60./24.
            else:
                expt = 30./60./24.
            npts = fdata[ok_fl,5] / expt # this is approximate... but faster than a total re-run

            ed_errors_n = np.sqrt(fdata[ok_fl,14]**2. / (fdata[ok_fl,9] * npts))

            sum_ed_err = str(np.sqrt(np.sum((ed_errors_n**2.))))
            sum_ed = str(np.sum(fdata[ok_fl, 14]))

        else:
            # Produce stats, even if no flares pass cut. The 0's are important
            Nflares = 0
            ed_hist = np.zeros(len(edbins) - 1)
            sum_ed = '0'
            sum_ed_err = '0'


        ed_freq = ed_hist / dur

        # Stats to compute:
        # - # flares
        # - Freq. of Flares
        # - flare rate vs energy
        '''
        here's the problem... how do you combine data for diff months/qtr's
        where you have diff exp times, noise properties, completeness limits?

        need to put data into some kind of relative unit, scale by time or something
        this is prob easiest if binned on to fixed ED bins... but that a bit
        unsatisfying since would like to keep all data.
        '''

        # columns to output:
        # KICnumber, Long/Short flag, Duration (days), ED68cut, Total Nflares,
        #   [in K fixed bins of ED, the total # of flares]

        kicnum = files[k][files[k].find('kplr')+4 : files[k].find('-2')]
        edcut_out = str(np.log10(edcut))
        Nflares_out = str(Nflares)
        dur_out = str(dur)

        if (files[k].find('slc') == -1):
            lsflag = '1'
        else:
            lsflag = '0'

        outstring = kicnum + ', ' + lsflag + ', ' + dur_out + ', ' + edcut_out + ', ' + \
                    Nflares_out + ', ' + sum_ed + ', ' + sum_ed_err
        for i in range(len(ed_freq)):
            outstring = outstring + ', ' + str(ed_freq[i])

        fout.write(outstring + '\n')

    fout.close()

    print('Be sure to compress the output file:')
    print('    gzip ' + outfile)
    print('for use in analysis.py next.')

    return

if __name__ == "__main__":
    # import sys
    PostCondor()
