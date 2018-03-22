import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
import matplotlib
import os
import appaloosa
import pandas as pd
import datetime
import warnings
from scipy.optimize import curve_fit, minimize
from astropy.stats import funcs
import appaloosa.analysis as analysis

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})


def paper1_plots(condorfile='condorout.dat.gz',
                 kicfile='kic.txt.gz', statsfile='stats.txt',
                 rerun=False,
                 figdir='figures/', figtype='.pdf'):
    '''
    Make plots for the first paper, which describes the Kepler flare sample.

    The Condor results are aggregated from PostCondor()

    Run on WWU workstation in dir: ~/research/kepler-flares/

    set rerun=True to go through all the flare files again, takes about 40min to run
    use rerun=False for quicker results using save file

    '''

    # read in KIC file
    # http://archive.stsci.edu/pub/kepler/catalogs/ <- data source
    # http://archive.stsci.edu/kepler/kic10/help/quickcol.html <- info

    print(datetime.datetime.now())
    kicdata = pd.read_csv(kicfile, delimiter='|')
    # need KICnumber, gmag, imag, logg (for cutting out crap only)
    # kicnum_k = kicdata['kic_kepler_id']


    fdata = pd.read_table(condorfile, delimiter=',', skiprows=1, header=None)
    ''' KICnumber, lsflag (0=llc,1=slc), dur [days], log(ed68), tot Nflares, sum ED, sum ED err, [ Flares/Day (logEDbin) ] '''

    # need KICnumber, Flare Freq data in units of ED
    kicnum_c = fdata.iloc[:,0].unique()

    print(datetime.datetime.now())
    # total # flares in dataset!
    print('total # flare candidates found: ', fdata.loc[:,4].sum())


    num_fl_tot = fdata.groupby([0])[4].sum()

    plt.figure()
    plt.hist(num_fl_tot.values, bins=100, range=(0,1000), histtype='step', color='k')
    plt.yscale('log')
    plt.xlabel('# Flares per Star')
    plt.ylabel('# Stars')
    plt.savefig(figdir + 'Nflares_hist' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    plt.close()


    ff = open(statsfile, 'w')
    ff.write('This is the stats file for appaloosa.analysis.paper1_plots() \n')
    ff.write('Total # flares found in entire sample: ' + str(fdata.loc[:,4].sum()) + '\n')
    ff.write('N stars with 25 or more flares: ' + str(len(np.where((num_fl_tot.values >= 25))[0])) + '\n')
    ff.write('Total num flares on stars with 25 or more flares: ' +
             str(np.sum((num_fl_tot.values)[np.where((num_fl_tot.values >= 25))])) + '\n')

    # match two dataframes on KIC number
    # bigdata = pd.merge(kicdata, kicnum_c, how='outer',
    #                    left_on='kic_kepler_id', right_on=0)

    bigdata = kicdata[kicdata['kic_kepler_id'].isin(kicnum_c)]


    # compute the distances and luminosities of all stars
    Lkp_uniq, dist_uniq, mass_uniq = analysis.energies(bigdata['kic_gmag'],
                                              bigdata['kic_kmag'],
                                              return_all=True)
    # plt.figure()
    # plt.scatter(bigdata['kic_gmag'] - bigdata['kic_kmag'], mass_uniq, alpha=0.2)
    # plt.show()


    # ingest Amy McQuillans rotation period catalog
    rotfile = 'comparison_datasets/Table_Periodic.txt'

    rotdata = pd.read_table(rotfile, delimiter=',', comment='#', header=None)
    # KID,Teff,logg,Mass,Prot,Prot_err,Rper,LPH,w,DC,Flag

    Prot_all = np.zeros(len(kicnum_c)) - 99.


    #########################################
    #      Explanatory Figures
    # including detrending examples, sample LC portions, etc
    # put early in script so can remamke quickly
    print(datetime.datetime.now())


    edbins = np.arange(-5, 5, 0.2)
    edbins = np.append(-10, edbins)
    edbins = np.append(edbins, 10)


    # make FFD plot for specific stars:
    #        GJ 1243, [ R. Clarke     ], J. Cornet,[  A. Boeck     ],  "Pearl"
    s_num = [9726699, 10387822, 10452709, 6224062, 4171937, 12314646, 11551430,
             9349698, 6928206, 5516671, 3222610] # random stars from Walkowicz (2011)

    s_num_all = np.loadtxt('kic4041.txt', skiprows=1)

    # For every star: compute flare rate at some arbitrary energy
    Epoint = 35
    EpointS = str(Epoint)

    # set the limit on numbers of flares per star required
    Nflare_limit = 100
    Nflare68_cut = 10

    ##########      THIS IS THE BIG BAD LOOP      ##########

    ap_loop_file = 'ap_analysis_loop.npz'
    print(datetime.datetime.now())

    if rerun is True:
        # silence some fit errors
        warnings.simplefilter('ignore', np.RankWarning)

        Nflare = np.zeros(len(kicnum_c)) # total num flares per star, same as before but slower...
        Nflare68 = np.zeros(len(kicnum_c))

        rate_E = np.zeros(len(kicnum_c)) - 99.
        fit_E = np.zeros(len(kicnum_c)) - 99.
        fit_Eerr = np.zeros(len(kicnum_c)) - 99.

        # vars for L_fl/L_kp (e.g. Lurie 2015)
        dur_all = np.zeros(len(kicnum_c)) - 99. # total duration of the star's LC
        ED_all = np.zeros(len(kicnum_c)) - 99. # sum of all Equiv Dur's for the star
        ED_all_err = np.zeros(len(kicnum_c)) - 99.

        # Also, compute FFD for every star
        ffd_ab = np.zeros((2,len(kicnum_c)))

        gr_all = np.zeros(len(kicnum_c)) - 99. # color used in prev work
        gi_all = np.zeros(len(kicnum_c)) - 99. # my preferred color

        logg_all = np.zeros(len(kicnum_c)) - 99. # use log g from KIC, with some level of trust

        maxE = np.zeros(len(kicnum_c)) - 99

        ra = np.zeros_like(kicnum_c) - 99.
        dec = np.zeros_like(kicnum_c) - 99.
        mass = np.zeros_like(kicnum_c) - 99.
        Lkp_all = np.zeros_like(kicnum_c) - 99.

        meanE = []

        for k in range(len(kicnum_c)):
        # for k in range(199836,199938):

            # find the k'th star in the KIC list in the Flare outputs
            star = np.where((fdata[0].values == kicnum_c[k]))[0]

            Nflare[k] = np.sum(fdata.loc[star,4].values) # total Num of flares
            dur_all[k] = np.sum(fdata.loc[star,2].values) # total duration (units: days)
            ED_all[k] = np.sum(fdata.loc[star,5].values) # total flare energy in Equiv Dur (units: seconds)
            ED_all_err[k] = np.sqrt(np.sum((fdata.loc[star,6].values)**2.))

            # arrays for FFD
            fnorm = np.zeros_like(edbins[1:])
            fsum = np.zeros_like(edbins[1:])

            # find this star in the KIC data
            mtch = np.where((bigdata['kic_kepler_id'].values == kicnum_c[k]))
            if len(mtch[0])>0:
                gr_all[k] = bigdata['kic_gmag'].values[mtch][0] - \
                            bigdata['kic_rmag'].values[mtch][0]

                gi_all[k] = bigdata['kic_gmag'].values[mtch][0] - \
                            bigdata['kic_imag'].values[mtch][0]

                logg_all[k] = bigdata['kic_logg'].values[mtch][0]

                ra[k] = bigdata['kic_degree_ra'].values[mtch][0]
                dec[k] = bigdata['kic_dec'].values[mtch][0]

                mass[k] = mass_uniq[mtch][0]
                Lkp_all[k] = Lkp_uniq[mtch][0]
                Lkp_i = Lkp_uniq[mtch][0]

                # for stars listed in the "to plot list", make a FFD figure
                if kicnum_c[k] in s_num:
                    doplot = True
                    plt.figure()
                else:
                    doplot = False

                if kicnum_c[k] in s_num_all:
                    doplot = True
                    ffig = plt.figure()
                    ax = ffig.add_subplot(111)

                # tmp array to hold the total number of flares in each FFD bin
                flare_tot = np.zeros_like(fnorm)

                Nflare68tmp = 0 # count number of flares above threshold

                for i in range(0,len(star)):
                    # Find the portion of the FFD that is above the 68% cutoff
                    ok = np.where((edbins[1:] >= fdata.loc[star[i],3]))[0]

                    if len(ok) > 0:
                        # add the rates together for a straight mean
                        fsum[ok] = fsum[ok] + fdata.loc[star[i],7:].values[ok]

                        # count the number for the straight mean
                        fnorm[ok] = fnorm[ok] + 1

                        # add the actual number of flares for this data portion: rate * duration
                        flare_tot = flare_tot + (fdata.loc[star[i],7:].values * fdata.loc[star[i],2])

                        # save the number of flares above E68 for this portion
                        Nflare68tmp = Nflare68tmp + sum((fdata.loc[star[i], 7:].values[ok] * fdata.loc[star[i], 2]))

                        if fdata.loc[star[i],1] == 1:
                            pclr = 'red' # long cadence data
                        else:
                            pclr = 'blue' # short cadence data

                        if doplot is True:
                            plt.plot(edbins[1:][ok][::-1] + Lkp_i,
                                     np.cumsum(fdata.loc[star[i],7:].values[ok][::-1]),
                                     alpha=0.35, color=pclr)

                Nflare68[k] = Nflare68tmp

                # the important arrays for the averaged FFD
                ffd_x = edbins[1:][::-1] + Lkp_i
                ffd_y = np.cumsum(fsum[::-1]/fnorm[::-1])

                # the "error" is the Poisson err from the total # flares per bin
                ffd_yerr = analysis._Perror(flare_tot[::-1], down=True) / dur_all[k]

                # Fit the FFD w/ a line, save the coefficeints
                ffd_ok = np.where((ffd_y > 0) & np.isfinite(ffd_y) &
                                  np.isfinite(ffd_x) & np.isfinite(ffd_yerr))

                # if there are any valid bins, find the max energy (bin)
                if len(ffd_ok[0])>0:
                    maxE[k] = np.nanmax(ffd_x[ffd_ok])

                # if there are at least 2 energy bins w/ valid flares...
                if len(ffd_ok[0])>1:
                    # compute the mean flare energy (bin) for this star
                    meanE = np.append(meanE, np.nanmedian(ffd_x[ffd_ok]))

                    '''
                    # the weights, in log rate units
                    ffd_weights = 1. / np.abs(ffd_yerr[ffd_ok]/(ffd_y[ffd_ok] * np.log(10)))

                    # fit the FFD w/ a line
                    fit, cov = np.polyfit(ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), 1, cov=True, w=ffd_weights) # fit using weights

                    # evaluate the FFD fit at the Energy point
                    fit_E[k] = 10.**(np.polyval(fit, Epoint))
                    '''

                    p0 = [-0.5, np.log10(np.nanmax(ffd_y[ffd_ok]))]
                    # fit, cov = curve_fit(_plaw, ffd_x[ffd_ok], ffd_y[ffd_ok], sigma=ffd_yerr[ffd_ok],
                    #                      absolute_sigma=False, p0=p0)
                    # fit_E[k] = _plaw(Epoint, *fit)


                    fit, cov = curve_fit(analysis._linfunc, ffd_x[ffd_ok], np.log10(ffd_y[ffd_ok]), p0=p0,
                                         sigma=np.abs(ffd_yerr[ffd_ok]/(ffd_y[ffd_ok] * np.log(10))) )

                    fit_E[k] = 10.0**analysis._linfunc(Epoint, *fit)

                    ffd_ab[:,k] = fit


                    # determine uncertainty on the fit evaluation point, with help from:
                    # http://stackoverflow.com/questions/28505008/numpy-polyfit-how-to-get-1-sigma-uncertainty-around-the-estimated-curve
                    TT = np.vstack([Epoint**(1-i) for i in range(2)]).T
                    # yi = np.dot(TT, fit)  # matrix multiplication calculates the polynomial values
                    C_yi = np.dot(TT, np.dot(cov, TT.T)) # C_y = TT*C_z*TT.T
                    fit_Eerr[k] = np.sqrt(np.diag(C_yi))  # Standard deviations are sqrt of diagonal


                # determine the actual value of the FFD at the Energy point using the fit
                # if (sum(ffd_x >= Epoint) > 0):
                #     rate_E[k] = max(ffd_y[ffd_x >= Epoint])

                if doplot is True:
                    # print(kicnum_c[k])
                    # print('ffd_ok:', ffd_ok)
                    # print('ffd_x:', ffd_x)
                    # print('ffd_y:', ffd_y)
                    # print('ffd_yerr:', ffd_yerr)
                    # print('meanE:', meanE)

                    plt.plot(ffd_x, ffd_y, linewidth=2, color='black', alpha=0.7)
                    plt.errorbar(ffd_x, ffd_y, ffd_yerr, fmt='k,')
                    if len(ffd_ok[0])>1:
                        # print('FIT: ', fit)

                        # plt.plot(ffd_x[ffd_ok], 10.0**(np.polyval(fit, ffd_x[ffd_ok])),
                        #          color='orange', linewidth=4, alpha=0.5)

                        # plt.plot(ffd_x[ffd_ok], _plaw(ffd_x[ffd_ok], *fit),
                        #          color='orange', linewidth=4, alpha=0.5)

                        plt.plot(ffd_x[ffd_ok], 10.0**analysis._linfunc(ffd_x[ffd_ok], *fit),
                                 color='navy', linewidth=4, alpha=0.75)


                        plt.yscale('log')
                        plt.xlim(np.nanmin(ffd_x[ffd_ok])-0.5, np.nanmax(ffd_x[ffd_ok])+0.5)
                        # plt.ylim(1e-3, 3e0)

                    # plt.title('KIC' + str(kicnum_c[k]) + ': ' +
                    #           'log R$_{'+EpointS+'}$ = ' + str(np.log10(fit_E[k])))
                    plt.xlabel('log Flare Energy (erg)')
                    plt.ylabel('Cumulative Flare Freq (#/day)')

                    if kicnum_c[k] in s_num_all:
                        plt.title('KIC ' + str(kicnum_c[k]))
                        plt.text(0.025, 0.025,
                                 r'$\alpha$=' + format(fit[0], '.2F') + r', $\beta$=' + format(fit[1], '.2F'),
                                 fontsize=10, transform=ax.transAxes)
                        plt.savefig('allFFDs/' + str(kicnum_c[k]) + '_ffd' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    else:
                        plt.savefig(figdir + str(kicnum_c[k]) + '_ffd' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close()



            # now match this star to the rotation period data
            rotmtch = np.where(rotdata.iloc[:,0].values == kicnum_c[k])
            if len(rotmtch[0])>0:
                Prot_all[k] = rotdata.iloc[:,4].values[rotmtch]

        # save results for faster reuse
        np.savez(ap_loop_file,
                 Nflare=Nflare, Nflare68=Nflare68, rate_E=rate_E, fit_E=fit_E, fit_Eerr=fit_Eerr,
                 ffd_ab=ffd_ab, gr_all=gr_all, gi_all=gi_all, meanE=meanE, maxE=maxE,
                 Prot_all=Prot_all, ED_all=ED_all, ED_all_err=ED_all_err, dur_all=dur_all, logg_all=logg_all,
                 ra=ra, dec=dec, mass=mass, Lkp_all=Lkp_all)

        ##### END OF THE BIG BAD LOOP #####

    else:
        # pull arrays back in via load!
        npz = np.load(ap_loop_file)
        Nflare = npz['Nflare']
        rate_E = npz['rate_E']
        fit_E = npz['fit_E']
        fit_Eerr = npz['fit_Eerr']
        ffd_ab = npz['ffd_ab']
        gr_all = npz['gr_all']
        gi_all = npz['gi_all']
        meanE = npz['meanE']
        maxE = npz['maxE']
        Prot_all = npz['Prot_all']
        dur_all = npz['dur_all']
        ED_all = npz['ED_all']
        ED_all_err = npz['ED_all_err']
        logg_all = npz['logg_all']
        Nflare68 = npz['Nflare68']
        ra = npz['ra']
        dec = npz['dec']
        mass = npz['mass']
        Lkp_all = npz['Lkp_all']
    print(datetime.datetime.now())


    #### a histogram of the average flare energy per star
    # plt.figure()
    # _htmp = plt.hist(meanE[np.where(np.isfinite(meanE))], bins=50)
    # plt.xlabel('mean energy (log E)')
    # plt.ylabel('# stars')
    # plt.savefig(figdir + 'mean_energy' + figtype,dpi=100)
    # plt.close()

    ff.write('total # of stars ' + str(len(kicnum_c)) + '\n')

    ff.write('mean flare energy: ' + str(np.nanmean(meanE[np.where(np.isfinite(meanE))])) + '\n')

    ### plot of maxE vs color
    Eok = np.where((maxE > 0))

    plt.figure()
    plt.scatter(gi_all[Eok], maxE[Eok], alpha=0.5, linewidths=0)
    plt.xlabel('g-i (mag)')
    plt.ylabel('Max log Flare Energy (erg)')
    plt.xlim(-1,3)
    plt.ylim(28,40)
    plt.savefig(figdir + 'maxE_vs_gi' + figtype, dpi=100)
    plt.close()

    ### histogram of maxE
    plt.figure()
    _ = plt.hist(maxE[Eok], bins=50)
    plt.xlabel('Max log Flare Energy (erg)')
    plt.ylabel('# stars')
    plt.savefig(figdir + 'logE_hist' + figtype, dpi=100)
    plt.close()


    ############################
    # TURN THESE R35 PLOTS OFF FOR NOW
    if False:
        # the big master plot, style taken from the K2 meeting plot...
        # plots vs R35
        clr = np.log10(fit_E)
        clr_raw = clr

        clr_raw_err = np.abs(fit_Eerr / (fit_E * np.log(10)))

        ff.write(str(len((np.where(np.isfinite(clr)))[0])) + ' stars have valid R_' + EpointS + ' values \n')

        isF = np.where(np.isfinite(clr))

        clr_rng = np.array([-2., 2.] )* np.nanstd(clr) + np.nanmedian(clr)


        ff.write('Nflare_limit = ' + str(Nflare_limit) + '\n')
        ff.write('# flares on stars that pass this limit: ' +
                 str(np.sum(Nflare[np.where((Nflare >= Nflare_limit))])) + '\n')
        ff.write('# stars that pass this limit: ' +
                 str(len(np.where((Nflare >= Nflare_limit))[0])) + '\n')

        ff.write('Nflare68_limit = ' + str(Nflare68_cut) + '\n')
        ff.write('# flares on stars that pass this limit: ' +
                 str(np.sum(Nflare[np.where((Nflare68 >= Nflare68_cut))])) + '\n')
        ff.write('# stars that pass this limit: ' +
                 str(len(np.where((Nflare68 >= Nflare68_cut))[0])) + '\n')


        # stars that have enough flares and have valid rates
        okclr = np.where((Nflare68 >= Nflare68_cut) & #(logg_all >= 3.8) &
                         np.isfinite(clr) & (Nflare >= Nflare_limit))

        # stars that just have valid rates
        okclr0 = np.where(  # (clr >= clr_rng[0]) & (clr <= clr_rng[1]) &
            np.isfinite(clr) & (Nflare >= 0))



        plt.figure()
        hh = plt.hist(clr[isF], bins=100, histtype='step', color='k')
        plt.xlabel('log R$_{'+EpointS+'}$ (#/day)')
        plt.ylabel('# Stars')
        plt.yscale('log')
        plt.savefig(figdir + 'R_' + EpointS + '_hist' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()



        # lets breifly revisit Nflares, look at where to pick a limit
        plt.figure()
        _ = plt.hist(np.log10(Nflare + 0.01), bins=100, cumulative=True, normed=True,
                     histtype='step', color='k')
        plt.xlim(-1,3)

        plt.axvline(x=np.log10(Nflare_limit), linewidth=3, color='red', alpha=0.5)
        plt.xlabel('log # Flares per Star')
        plt.ylabel('Cumulative Fraction of Stars')
        plt.savefig(figdir + 'cumulative_hist' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()



        ff.write('okclr len is ' + str(len(okclr[0])) + '\n')
        ff.write('okclr0 len is ' + str(len(okclr0[0])) + '\n')

        # ff.write('# stars that pass Nflare_limit and have valid rotation periods: '+
        #          str( len(np.where((Nflare >= Nflare_limit) & (Prot_all > 0.1))[0]) ) + '\n')
        print(datetime.datetime.now())



        # first, a basic plot of flare rate versus color

        rate_range = [[0,3], [10.**clr_rng[0], 10.**clr_rng[1]]]

        plt.figure()
        plt.hist2d(gi_all[okclr], fit_E[okclr], bins=100, range=rate_range,
                   alpha=1.0, norm=LogNorm(), cmap=cm.Greys)
        plt.xlabel('g-i (mag)')
        plt.yscale('log')
        plt.ylabel('R$_{'+EpointS+'}$ (#/day)')
        plt.savefig(figdir + 'flarerate_okclr' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        plt.figure()
        plt.hist2d(gi_all[okclr0], fit_E[okclr0], bins=100, range=rate_range,
                   alpha=1.0, norm=LogNorm(), cmap=cm.Greys)
        plt.xlabel('g-i (mag)')
        plt.yscale('log')
        plt.ylabel('R$_{'+EpointS+'}$ (#/day)')
        plt.savefig(figdir + 'flarerate_okclr0' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        ##### The science plot! #####
        ##### try it as a scatter plot


        ####    do it again, but now with (g-i) color   ###
        plt.figure()
        plt.scatter(gi_all[okclr], Prot_all[okclr], c=clr[okclr],
                    alpha=0.7, lw=0.5, cmap=cm.afmhot_r, s=50)
        plt.xlabel('g-i (mag)')
        plt.ylabel('P$_{rot}$ (days)')
        plt.yscale('log')
        plt.xlim((0,3))
        plt.ylim((0.1,100))
        cb = plt.colorbar()
        cb.set_label('log R$_{'+EpointS+'}$ (#/day)')
        plt.savefig(figdir + 'masterplot_okclr_gi' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        ####
        plt.figure()
        plt.scatter(gi_all[okclr0], Prot_all[okclr0], c=clr[okclr0],
                    alpha=0.7, lw=0.5, cmap=cm.afmhot_r, s=50)
        plt.xlabel('g-i (mag)')
        plt.ylabel('P$_{rot}$ (days)')
        plt.yscale('log')
        plt.xlim((0,3))
        plt.ylim((0.1,100))
        cb = plt.colorbar()
        cb.set_label('log R$_{'+EpointS+'}$ (#/day)')
        plt.savefig(figdir + 'masterplot_okclr0_gi' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


        ################
        # pick target star color range, look at evolution of Rate vs Rotation

        crng = np.array([[0.5, 0.75],[0.75, 1.],[1.50, 2.],[2.25, 2.75]])

        for k in range(crng.shape[0]):
            ts = np.where((gi_all[okclr]  >= crng[k,0]) &
                          (gi_all[okclr] <= crng[k,1]) &
                          (Prot_all[okclr] > 0.1))

            ff.write('# that pass TS color cut: '+str(len(ts[0])) + '\n')

            plt.figure()
            # plt.scatter(Prot_all[okclr0][ts0], clr_raw[okclr0][ts0], s=20, alpha=0.7,lw=0.5,c='red')
            plt.scatter(Prot_all[okclr][ts], clr_raw[okclr][ts], s=50, alpha=1,lw=0.5, c='k')
            # plt.errorbar(Prot_all[okclr][ts], clr_raw[okclr][ts], yerr=clr_raw_err[okclr][ts], fmt='k,')
            plt.xlabel('P$_{rot}$ (days)')
            plt.ylabel('log R$_{'+EpointS+'}$ (#/day)')
            plt.title(str(crng[k,0])+' < (g-i) < '+str(crng[k,1]))
            plt.xscale('log')
            plt.ylim(-4,0)
            plt.xlim(0.1,100)
            plt.savefig(figdir + 'rot_rate'+str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
            plt.close()


    ##### make master (color,rot,rate) figure as a pixelated plot
    '''
    bin2d, xx, yy, _ = binned_statistic_2d(gr_all[okclr], np.log10(Prot_all[okclr]), clr[okclr],
                                           statistic='median', range=[[-1,4],[-1,2]], bins=75)

    plt.figure()

    plt.imshow(bin2d.T, interpolation='nearest', aspect='auto', origin='lower',
               extent=(xx.min(),xx.max(),yy.min(),yy.max()),
               cmap=plt.cm.afmhot_r)

    plt.xlabel('g-r (mag)')
    plt.ylabel('log P$_{rot}$ (days)')
    plt.xlim((0,1.7))
    plt.ylim(-1,2)
    cb = plt.colorbar()
    cb.set_label('log R$_{'+EpointS+'}$ (#/day)')
    plt.savefig('masterplot_pixel.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    '''


    #########################################
    #########################################
    #    plots as a function of Lfl_Lbol

    # total fractional energy (in seconds) / total duration (in seconds)
    Lfl_Lbol = ED_all / (dur_all * 60. * 60. * 24.)
    Lfl_Lbol_err = ED_all_err / (dur_all * 60. * 60. * 24.)

    Lfl_Lbol_label = 'log ($L_{fl}$ $L_{Kp}^{-1}$)'



    clr = np.log10(Lfl_Lbol)
    clr_err = np.abs(Lfl_Lbol_err/ (Lfl_Lbol * np.log(10.)))
    clr_raw = clr
    isF = np.where(np.isfinite(clr))

    #### THIS IS WHERE I MAKE MY PRIMARY GOOD SAMPLE CUT
    #      down to ~4k stars
    okclr = np.where((Nflare68 >= Nflare68_cut) & #(logg_all >= 3.5) &
                     np.isfinite(clr) & (Nflare >= Nflare_limit))

    # print('eee:', np.shape(mass[okclr]), np.shape(ffd_ab[1,okclr]), np.shape(rate_E[okclr]))


    # investigatory figure of N flares vs Lfl_Lkp. Are we missing "good" stars w/ too few flares?
    plt.figure()
    plt.scatter(Nflare68, clr, alpha=0.3, s=5, lw=0, color='k')
    plt.scatter(Nflare68[okclr], clr[okclr], lw=0, alpha=0.2, s=20)
    plt.xlabel('Nflare68')
    plt.xscale('log')
    plt.ylim(-10,0)
    plt.xlim(0.8,1e4)
    plt.ylabel(Lfl_Lbol_label)
    plt.savefig(figdir + 'Nfl68_vs_LflLkp.png', dpi=300)
    plt.close()



    # spit out table of KID, color (g-i), Lfl/Lbol
    dfout = pd.DataFrame(data={'kicnum': kicnum_c[okclr],
                               'giclr': gi_all[okclr],
                               'mass': mass[okclr],
                               'Prot':Prot_all[okclr],
                               'LflLkep': Lfl_Lbol[okclr],
                               'LflLkep_err': Lfl_Lbol_err[okclr],
                               'Nflares':Nflare[okclr],
                               'Nflare68':Nflare68[okclr],
                               'R35':rate_E[okclr],
                               'alpha':np.squeeze(ffd_ab[1, okclr]),
                               'beta':np.squeeze(ffd_ab[0, okclr])
                               })

    # dfout.to_csv('kic_lflare.csv')
    dfout.to_csv('kepler_flare_output.csv')

    tau_all = analysis._tau(mass)
    Rossby = Prot_all / tau_all

    # lets check to make sure the Rossby number calculations are put together right
    plt.figure()
    plt.scatter(gi_all[okclr], tau_all[okclr])
    plt.xlabel('g-i')
    plt.ylabel(r'$\tau$')
    plt.savefig(figdir + 'gi_vs_tau' + figtype)
    plt.close()

    plt.figure()
    plt.scatter(Rossby[okclr], tau_all[okclr])
    plt.xlabel('Ro')
    plt.ylabel(r'$\tau$')
    plt.savefig(figdir + 'Ro_vs_tau' + figtype)
    plt.close()



    ff.write('OKCLR rules: Lfl/Lkp>0, Nflare>'+str(Nflare_limit)+', Nflare68>'+str(Nflare68_cut)+'\n')
    ff.write('# stars that pass final "OKCLR" cuts: ' + str(len(okclr[0])) + '\n')
    ff.write('# flares on stars that pass final OKCLR cut: ' + str(np.sum(Nflare[okclr])) + '\n')
    ff.write('# flares over E68 on stars that pass final OKCLR cut: ' + str(np.sum(Nflare68[okclr])) + '\n')
    ff.write('# stars that pass OKCLR cut, and have Prot>0.1: ' +
             str(len(np.where((Prot_all[okclr] > 0.1))[0]))+'\n')

    plt.figure()
    plt.scatter(gi_all, Prot_all, c=clr_raw,
                alpha=0.7, lw=0.5, cmap=cm.afmhot_r, s=25)
    plt.xlabel('g-i (mag)')
    plt.ylabel('P$_{rot}$ (days)')
    plt.yscale('log')
    plt.xlim((0,3))
    plt.ylim((0.1,100))
    cb = plt.colorbar()
    cb.set_label(Lfl_Lbol_label)
    plt.savefig(figdir + 'masterplot_lfl_lkep_raw' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # get isochrone data to convert g-i to mass for second axis label
    try:
        __file__
    except NameError:
        __file__ = os.getenv("HOME") +  '/python/appaloosa/analysis.py'

    isodir = os.path.dirname(os.path.realpath(__file__)) + '/../misc/'
    isochrone = '1.0gyr.dat'
    massi, Mkp, Mg, Mi, Mk = np.loadtxt(isodir + isochrone, comments='#', unpack=True, usecols=(2,8,9,11,18))
    Mgi  = (Mg - Mi)
    ss = np.argsort(Mgi)  # needs to be sorted for interpolation
    mass_o = np.interp(np.arange(0.5,3.5,0.5), Mgi[ss], massi[ss])
    mass_s = map(lambda x: format(x, '.2F'), mass_o)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twiny()
    plt.scatter(gi_all[okclr], Prot_all[okclr], c=clr[okclr],
                alpha=0.7, lw=0.5, cmap=cm.afmhot_r, s=50)
    ax1.set_xlabel('g-i (mag)')
    ax1.set_ylabel('P$_{rot}$ (days)')
    plt.yscale('log')
    ax1.set_xlim((0,3))
    plt.ylim((0.1,100))

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(np.arange(0.5, 3.5, 0.5))
    ax2.set_xticklabels(mass_s)
    ax2.set_xlabel(r'Mass ($M_\odot$)')

    cb = plt.colorbar()
    cb.set_label(Lfl_Lbol_label)

    plt.savefig(figdir + 'masterplot_lfl_lkep' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    # a diagnostic plot
    plt.figure()
    plt.scatter(Nflare+1, Nflare68+1, alpha=0.25, linewidths=0, c='k')
    plt.scatter(Nflare[okclr]+1, Nflare68[okclr]+1, alpha=0.5, linewidths=0)
    plt.xlabel('Nflare')
    plt.ylabel('Nflare68')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(figdir + 'Nflare_vs_Nflare68' + figtype, dpi=100)
    plt.close()

    # Eok = np.where((maxE > 0))
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.scatter(gi_all[okclr], maxE[okclr], alpha=0.5, linewidths=0, c='k')
    ax1.set_xlabel('g-i (mag)')
    ax1.set_ylabel('Max log Flare Energy (erg)')
    ax1.set_xlim(-1, 3)
    ax1.set_ylim(32, 40)

    ax2.set_xlim(ax1.get_xlim())
    mass_o = np.interp(np.arange(0., 3.5, 0.5), Mgi[ss], massi[ss])
    mass_s = map(lambda x: format(x, '.2F'), mass_o)
    ax2.set_xticks(np.arange(0., 3.5, 0.5))
    ax2.set_xticklabels(mass_s)
    ax2.set_xlabel(r'Mass ($M_\odot$)')

    plt.savefig(figdir + 'maxE_vs_gi_okclr' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    ### plot of Nflares vs color
    plt.figure()
    plt.scatter(gi_all[okclr], Nflare[okclr], alpha=0.5, linewidths=0, c='k')
    plt.xlabel('g-i (mag)')
    plt.ylabel('Number of Flares')
    plt.yscale('log')
    plt.ylim(0.8e2,1e4)
    plt.xlim(-1,3)
    plt.savefig(figdir + 'Nflare_vs_gi' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter(gi_all[okclr], Nflare68[okclr], alpha=0.5, linewidths=0, c='k')
    plt.xlabel('g-i (mag)')
    plt.ylabel('Number of Flares (E > E$_{68}$)')
    plt.yscale('log')
    plt.ylim(0.8e2,1e4)
    plt.xlim(-1,3)
    plt.savefig(figdir + 'Nflare68_vs_gi' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    plt.figure()
    plt.scatter(gi_all[isF], Nflare[isF], alpha=0.5, linewidths=0, c='k')
    plt.xlabel('g-i (mag)')
    plt.ylabel('Number of Flares')
    plt.yscale('log')
    plt.ylim(0.9e2,1e5)
    plt.xlim(-1,3)
    plt.savefig(figdir + 'Nflare_vs_gi_raw' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    _ = plt.hist(Nflare68, bins=np.arange(0, 1000, 25), histtype='step', color='g', alpha=0.6)
    _ = plt.hist(Nflare68[okclr], bins=np.arange(0,1000,25), histtype='step', color='k')
    plt.xlabel('Number of Flares per Star (E > E$_{68}$)')
    plt.ylabel('Number of Stars')
    plt.yscale('log')
    plt.xlim(0,1000)
    plt.savefig(figdir + 'Nflare68' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    crng = np.array([[0.5, 0.75],
                     [0.75, 1.0],
                     [1., 1.5],
                     [1.5, 2.],
                     [2., 2.5],
                     [2.5, 3.]])
                     # [0.0, 0.5]]) # a bin I don't expect to understand. Should be F stars

    frac_flaring = np.zeros(crng.shape[0])
    frac_flaring_err = np.zeros_like(crng)
    # frac_flaring_perr = np.zeros _like(crng)

    Ro_peakL = np.zeros(crng.shape[0])
    Ro_slope = np.zeros(crng.shape[0])
    Ro_break = np.zeros(crng.shape[0])

    fparam = 0.

    for k in range(crng.shape[0]):
        ts = np.where((gi_all[okclr]  > crng[k,0]) &
                      (gi_all[okclr] <= crng[k,1]) &
                      (Prot_all[okclr] > 0.1))

        ff.write('# that pass color cut: '+str(len(ts[0])) + '\n')

        plt.figure()
        # plt.scatter(Prot_all[okclr0][ts0], clr_raw[okclr0][ts0], s=20, alpha=0.7,lw=0.5,c='red')
        plt.scatter(Prot_all[okclr][ts], clr_raw[okclr][ts], s=50, alpha=1,lw=0.5, c='k')
        plt.errorbar(Prot_all[okclr][ts], clr_raw[okclr][ts], yerr=clr_err[okclr][ts], fmt='none', ecolor='k', capsize=0)
        plt.xlabel('P$_{rot}$ (days)')
        plt.ylabel(Lfl_Lbol_label)
        plt.title(str(crng[k,0])+' < (g-i) < '+str(crng[k,1]) + ', N='+str(len(ts[0])))
        plt.xscale('log')
        plt.ylim(-6,-1)
        plt.xlim(0.1,100)
        plt.savefig(figdir + 'rot_lfllkp'+str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()

        print(str(crng[k,0])+' < (g-i) < '+str(crng[k,1]))

        p0 = (-3., -0.2, -1.)  # saturation level, Prot break, slope
        popt1, pcov = curve_fit(analysis.analysis.RoFlare, np.log10(Prot_all[okclr][ts]),
                                clr_raw[okclr][ts], p0=p0)
        lfit_k = np.polyfit(np.log10(Prot_all[okclr][ts]), clr[okclr][ts], 1)
        bic_l = appaloosa.chisq(clr[okclr][ts], clr_err[okclr][ts] + fparam, np.polyval(lfit_k, np.log10(Prot_all[okclr][ts]))) + 2. * np.log(len(ts[0]))
        bic_k = appaloosa.chisq(clr[okclr][ts], clr_err[okclr][ts] + fparam, analysis.RoFlare(np.log10(Prot_all[okclr][ts]), *popt1)) + 3. * np.log(len(ts[0]))
        print('> BIC_k rotation ', bic_k, bic_l)

        # compute the fraction of stars within each color bin that pass our flare cuts
        doflare = np.where((gi_all[okclr]  > crng[k,0]) & (gi_all[okclr] <= crng[k,1]))
        allstars = np.where((gi_all > crng[k,0]) & (gi_all <= crng[k,1]))

        frac_flaring[k] = np.float(len(doflare[0])) / np.float(len(allstars[0]))
        frac_flaring_err[k,:] = funcs.binom_conf_interval(np.float(len(doflare[0])), np.float(len(allstars[0])), interval='wald')
        # frac_flaring_perr[k, :] = _Perror(len(doflare[0]), full=True)

        # print(len(doflare[0]), len(allstars[0]), frac_flaring[k], frac_flaring_err[k,:])


        # the rossby number figure (incl fit)
        p0 = (-3., -1.2, -1.) # saturation level, Ro break, slope
        popt1, pcov = curve_fit(analysis.RoFlare, np.log10(Rossby[okclr][ts]),
                                clr_raw[okclr][ts], p0=p0)
        perr1 = np.sqrt(np.diag(pcov))
        ff.write('Rossby Parameters ' + str(k) + ': ' + str(popt1) + str(perr1) + '\n')

        lfit_k = np.polyfit(np.log10(Rossby[okclr][ts]), clr[okclr][ts], 1)
        bic_l = appaloosa.chisq(clr[okclr][ts], clr_err[okclr][ts] + fparam, np.polyval(lfit_k, np.log10(Rossby[okclr][ts]))) + 2. * np.log(len(ts[0]))
        bic_k = appaloosa.chisq(clr[okclr][ts], clr_err[okclr][ts] + fparam, analysis.RoFlare(np.log10(Rossby[okclr][ts]), *popt1)) + 3. * np.log(len(ts[0]))
        print('> BIC_k rossby ', bic_k, bic_l)

        Ro_peakL[k], Ro_break[k], Ro_slope[k] = popt1

        plt.figure()
        plt.scatter(Rossby[okclr][ts], clr_raw[okclr][ts], s=50, alpha=1, lw=0.5, c='k')
        plt.plot(10.**np.arange(-3, 1, .01), analysis.RoFlare(np.arange(-3, 1, .01), *popt1), c='red', lw=3, alpha=0.75)

        # lfit, lcov = np.polyfit(np.log10(Rossby[okclr][ts]), clr_raw[okclr][ts], 1, cov=True)
        # plt.plot(10.**np.arange(-3, 1, .01), np.polyval(lfit, np.arange(-3, 1, .01)), c='blue')

        plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
        plt.ylabel(Lfl_Lbol_label)
        plt.title(str(crng[k, 0]) + ' < (g-i) < ' + str(crng[k, 1]) + ', N=' + str(len(ts[0])))
        plt.xscale('log')
        plt.xlim(0.8e-2, 4e0)
        plt.ylim(-5, -1.5)
        plt.savefig(figdir + 'Rossby_lfllkp' + str(k) + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()


    pok = np.where((Prot_all[okclr] > 0.1) &
                   (gi_all[okclr] > 0.75)) # manually throw out the bluest stars

    pokF = np.where((Prot_all[isF] > 0.1) &
                    (gi_all[isF] > 0.75))

    pok_e = np.where((Prot_all[okclr] > 0.1) &
                     (gi_all[okclr] > 0.75) &
                     (gi_all[okclr] <= 1.5))

    pok_l = np.where((Prot_all[okclr] > 0.1) &
                     (gi_all[okclr] >= 1.5) &
                     (gi_all[okclr] <= 3.))

    print('typical error in Lfl_Lkp: ', np.median(clr_err[okclr][pok]))

    p0 = (-3., -0.8, -1.)
    popt1, pcov = curve_fit(analysis.RoFlare, np.log10(Rossby[okclr][pok]),
                            clr[okclr][pok], p0=p0)
    perr1 = np.sqrt(np.diag(pcov))
    print('Rossby numbers:', popt1, perr1)
    ff.write('Rossby Parameters: ' + str(popt1) + str(perr1) + '\n')

    # the referee wants us to compute just a straight line too
    lfit, lcov = np.polyfit(np.log10(Rossby[okclr][pok]), clr[okclr][pok], 1, cov=True)
    lerr = np.sqrt(np.diag(lcov))
    print('Powerlaw numbers:', lfit, lerr)
    ff.write('Powerlaw Parameters: ' + str(lfit) +  str(lerr) + '\n')

    bic_fit = appaloosa.chisq(clr[okclr][pok], clr_err[okclr][pok] + fparam,
                              analysis.RoFlare(np.log10(Rossby[okclr][pok]), *popt1)) + \
              3. * np.log(len(pok[0]))
    bic_flat = appaloosa.chisq(clr[okclr][pok], clr_err[okclr][pok] + fparam,
                               np.polyval(lfit, np.log10(Rossby[okclr][pok]))) + \
               2. * np.log(len(pok[0]))

    print('> BIC_fit: ', bic_fit)
    print('> BIC_flat: ', bic_flat)

    print('Chisqs: ',
          appaloosa.chisq(clr[okclr][pok], clr_err[okclr][pok], analysis.RoFlare(np.log10(Rossby[okclr][pok]), *popt1)),
          appaloosa.chisq(clr[okclr][pok], clr_err[okclr][pok], np.polyval(lfit, np.log10(Rossby[okclr][pok]))) )

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    plt.scatter(Rossby[okclr][pok], clr[okclr][pok], s=50, lw=0, c='k', alpha=0.5)
    # plt.errorbar(Rossby[okclr], clr[okclr], yerr=clr_err[okclr], fmt='none', ecolor='k', capsize=0)

    plt.plot(10.**np.arange(-3, 1,.01), analysis.RoFlare(np.arange(-3, 1,.01), *popt1),
             c='red', lw=3, alpha=0.75)

    plt.plot(10.**np.arange(-3, 1,.01), np.polyval(lfit, np.arange(-3,1,.01)),
             c='blue', lw=2, alpha=0.6, linestyle='--')

    plt.ylabel(Lfl_Lbol_label)
    ax1.set_xlabel(r'Ro = P$_{rot}$ / $\tau$')
    ax1.set_xscale('log')
    ax1.set_xlim(0.8e-2, 4e0)
    ax1.set_xticks((0.01, 0.1, 1))
    ax1.set_xticklabels(('0.01', '0.1', '1.0'))
    plt.ylim(-5, -1.5)
    plt.savefig(figdir + 'Rossby_lfllkp' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # try the rossby figure colored by mass
    plt.figure()
    plt.scatter(Rossby[okclr][pok], clr[okclr][pok],
                s=50, lw=0, alpha=0.75, c=mass[okclr][pok])
    cbar = plt.colorbar()
    cbar.set_label('Mass')
    plt.ylabel(Lfl_Lbol_label)
    plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
    plt.xscale('log')
    plt.xlim(0.8e-2, 4e0)
    plt.ylim(-5, -1.5)
    plt.savefig(figdir + 'Rossby_lfllkp_color' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter(Rossby[isF][pokF], clr[isF][pokF],
                lw=0, alpha=0.35, c=mass[isF][pokF], s=5)
    cbar = plt.colorbar()
    cbar.set_label('Mass')
    plt.ylabel(Lfl_Lbol_label)
    plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
    plt.xscale('log')
    plt.xlim(1e-3, 1e1)
    plt.ylim(-8, -1.5)
    plt.savefig(figdir + 'Rossby_lfllkp_color_all' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    # now 2 more versions of the Rossby figure using 2 bigger mass bins
    plt.figure()
    plt.scatter(Rossby[okclr][pok_e], clr[okclr][pok_e],
                s=50, lw=0, alpha=0.5, c='k')
    plt.ylabel(Lfl_Lbol_label)
    plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
    plt.xscale('log')
    plt.xlim(0.8e-2, 4e0)
    plt.ylim(-5, -1.5)
    plt.savefig(figdir + 'Rossby_lfllkp_e' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter(Rossby[okclr][pok_l], clr[okclr][pok_l],
                s=50, lw=0, alpha=0.5, c='k')
    plt.ylabel(Lfl_Lbol_label)
    plt.xlabel(r'Ro = P$_{rot}$ / $\tau$')
    plt.xscale('log')
    plt.xlim(0.8e-2, 4e0)
    plt.ylim(-5, -1.5)
    plt.savefig(figdir + 'Rossby_lfllkp_l' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    ### fraction of flaring stars
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.errorbar((crng[:,0] + crng[:,1])/2., frac_flaring, xerr=(crng[:,1] - crng[:,0])/2.,
                 ecolor='k', capsize=0, fmt='none', yerr=frac_flaring_err.T)
    ax1.set_xlabel('g-i (mag)')
    ax1.set_xlim([0,3])
    ax1.set_ylabel('Fraction of Flaring Stars')
    ax1.set_ylim([0, 0.1])

    ax2.set_xlim(ax1.get_xlim())
    mass_o = np.interp(np.arange(0.5, 3.5, 0.5), Mgi[ss], massi[ss])
    mass_s = map(lambda x: format(x, '.2F'), mass_o)
    ax2.set_xticks(np.arange(0.5,3.5,0.5))
    ax2.set_xticklabels(mass_s)
    ax2.set_xlabel(r'Mass ($M_\odot$)')
    # add second axis label of Mass for the referee
    # based on example here: http://stackoverflow.com/a/10517481

    plt.savefig(figdir + 'frac_flaring' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    ### plot of Rossby parameters vs color
    plt.figure()
    plt.scatter((crng[:,0] + crng[:,1])/2., Ro_break)
    plt.xlim(0,3)
    plt.savefig(figdir + 'Ro_break' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter((crng[:, 0] + crng[:, 1]) / 2., Ro_peakL)
    plt.xlim(0, 3)
    plt.savefig(figdir + 'Ro_peakL' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter((crng[:, 0] + crng[:, 1]) / 2., Ro_slope)
    plt.xlim(0, 3)
    plt.savefig(figdir + 'Ro_slope' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    #    / plots as a function of Lfl_Lbol
    #########################################

    plt.figure()
    plt.scatter(gi_all[okclr], ffd_ab[0,okclr], alpha=0.5, lw=0)
    plt.xlabel('g-i (mag)')
    plt.xlim(0, 3)
    plt.ylabel(r'$\beta$ (log rate per energy)')
    plt.ylim(-2, 0.1)
    plt.savefig(figdir + 'ffd_a' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()

    plt.figure()
    plt.scatter(gi_all[okclr], ffd_ab[1, okclr], alpha=0.5, lw=0)
    plt.xlabel('g-i (mag)')
    plt.xlim(0, 3)
    plt.ylabel(r'$\alpha$ (log # flares per day)')
    plt.savefig(figdir + 'ffd_b' + figtype, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    #########################################
    #      NGC 6811 plots



    ocfile='comparison_datasets/meibom2011_tbl1.txt'

    # Remake the gyrochronology plot from Meibom et al (2011) for NGC 6811 (color vs Prot),
    #  and add another panel of (color vs flare rate) or something similar

    ocdata = pd.read_table(ocfile, header=None, comment='#', delim_whitespace=True)
    # col's I care about:
    # KIC=0, g=7, r=8, Per=9


    ##### simple rotation period plot remake from paper
    plt.figure()
    plt.scatter((ocdata.iloc[:,7]-ocdata.iloc[:,8]), ocdata.iloc[:,9])
    plt.xlabel('g-r (mag)')
    plt.ylabel(r'P$_{rot}$ (days)')
    plt.savefig(figdir + 'ngc6811_gyro.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    plt.close()

    rate_oc = np.zeros(ocdata.shape[0]) - 99.
    fit_oc = np.zeros(ocdata.shape[0]) - 99.
    Lfl_Lbol_oc = np.zeros(ocdata.shape[0]) - 99.

    for k in range(ocdata.shape[0]):
        mtch = np.where((kicnum_c == ocdata.iloc[:,0].values[k]))
        if len(mtch[0])>0:
            rate_oc[k] = rate_E[mtch]
            fit_oc[k] = np.polyval(ffd_ab[:,mtch], Epoint)
            Lfl_Lbol_oc[k] = clr_raw[mtch]


    '''
    #####
    plt.figure()
     # add contours for the entire field
    # plt.hist2d(gr_all[okclr], fit_E[okclr], bins=100, range=rate_range,
    #            alpha=1.0, norm=LogNorm(), cmap=cm.Greys)
    plt.scatter((ocdata.iloc[:,7]-ocdata.iloc[:,8]), fit_oc)
    plt.xlabel('g-r (mag)')
    plt.xlim(0.2, 0.9)
    plt.ylim(-3.5, -2)
    plt.ylabel('R$_{'+EpointS+'}$ (#/day)')
    plt.savefig('ngc6811_flare_all.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()



    #####

    plt.figure()
    plt.scatter(gr_all[okclr], Prot_all[okclr], c=clr[okclr], alpha=0.7, lw=0, cmap=cm.afmhot_r, s=50)
    cb = plt.colorbar()
    cb.set_label('log R$_{'+EpointS+'}$ (#/day)')
    plt.scatter((ocdata.iloc[:,7]-ocdata.iloc[:,8]), ocdata.iloc[:,9], c=np.log10(fit_oc), cmap=cm.YlGnBu_r, s=50)
    plt.xlabel('g-r (mag)')
    plt.ylabel('P$_{rot}$ (days)')
    plt.yscale('log')
    plt.xlim((0,1.7))
    plt.ylim((0.1,100))
    cb2 = plt.colorbar()
    plt.savefig('masterplot_cluster.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    # plt.show()
    '''

    ####
    plt.figure()
    plt.scatter((ocdata.iloc[:, 7] - ocdata.iloc[:, 8]), Lfl_Lbol_oc, s=50)
    plt.xlabel('g-r (mag)')
    plt.ylabel(Lfl_Lbol_label)
    plt.ylim(-7, 1)
    # plt.xlim(0, 1.7)
    plt.savefig(figdir + 'ngc6811_Lfl.png', dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()


    #      /NGC 6811 plots
    #########################################




    ### stars with 50 largest E flares
    Esort = np.argsort(maxE)[::-1]

    Efair = np.where((maxE[Esort] < 40))

    ff.write('__ top 50 energy flare stars __' + '\n')
    for k in range(0, 50):
        ff.write(str(kicnum_c[Esort][Efair][k]) +
                 ', ' + str(maxE[Esort][Efair][k]) + '\n')


    ff.close() # close the output stats file
    return


if __name__ == "__main__":
    '''
      let this file be called from the terminal directly
    '''

    paper1_plots()
