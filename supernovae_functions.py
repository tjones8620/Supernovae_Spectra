#!/usr/bin/env python
# coding: utf-8

# # Contains functions for both parts of supernovae assignment

import numpy as np
import os
import uncertainties as unc
from uncertainties import unumpy
from scipy.optimize import curve_fit

def get_filepaths(directory):
    """
    This function will generate the file names in a directory 
    tree by walking the tree either top-down or bottom-up. For each 
    directory in the tree rooted at directory top (including top itself), 
    it yields a 3-tuple (dirpath, dirnames, filenames).
    """
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths  # Self-explanatory.


def gaussian(x, mean, sigma, b, a):
    """
    Gaussian function 
    """
    return a / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mean)**2 / sigma**2) + b


def redshift(lam_obs, lam_rest, dlam_obs):
    """
    - Calculates the redshift of a line 
      from a spectrum compared to the 
      wavelength of that line at rest
    - Returns the redshift and associated
      error
    """
    value = (lam_obs - lam_rest)/lam_rest
    err = dlam_obs / lam_rest
    
    return value, err


def spectra_plot(ax, i, title, wl, flux):
    """
    Plots all spectra in dataset
    with indices < i
    """
    ax[i].set_title(title[i], fontsize = 15, fontweight = 'bold')
    ax[i].plot(wl[i], flux[i], color = "blue")
    ax[i].set_xlim(wl[i].min(), wl[i].max())
    ax[i].set_ylim(flux[i].min()-0.1, flux[i].max()+0.1)
    ax[i].set_xlabel(r"Wavelength $(\AA)$", fontsize = 10)
    ax[i].set_ylabel("Relative Flux", fontsize = 10)


def halpha_fit(ax, i, plot = False, **kwargs):
    """
    - This function fits the halpha lines in the spectrum of each supernova
    given the guess for the wavelength of the halpha line and the range of 
    points over which to fit the halpha line. 
    
    - The fitting function used is a gaussian which gives the mean of the
    peak and the standard deviation which is taken to be the error.
    
    -The function returns the mean and standard deviation of the lines that
    are fitted, along with plots of the lines with their respective fits.
    """

    halpha_guess = kwargs['halpha_guess']
    fit_range = kwargs['fit_range']
    wl = kwargs['wl']
    flux = kwargs['flux']
    halpha_means = kwargs['halpha_means']
    halpha_error = kwargs['halpha_error']
    sn_names = kwargs['sn_names']


    #changing range to just fit over the vicinity of the halpha peak
    arguments = np.where((wl[i] > halpha_guess[i] - fit_range[i])&(wl[i] < halpha_guess[i] + fit_range[i])) 
    
    wl_restrict = wl[i][arguments]
    flux_restrict = flux[i][arguments]
    
    #fitting each halpha line with a gaussian 
    fit, var = curve_fit(gaussian, wl_restrict, flux_restrict, p0 = [halpha_guess[i], 10, 3, 0])
    mean, sigma, b, a = fit
    
    mean_error = np.sqrt(np.diag(var)[0]) #error in the mean of the halpha line
    
    #adding halpha mean wl and error to arrays    
    halpha_means[i] = mean 
    halpha_error[i] = np.sqrt(np.diag(var)[0]) 
    
    #linear array to plot the fitted gaussian over
    x = np.linspace(mean - fit_range[i], mean + fit_range[i] , 100)
    
    if plot == True:
        ax[i].set_title(sn_names[i], fontsize = 15, fontweight = 'bold')
        ax[i].plot(wl_restrict, flux_restrict, 'k.', markersize = 8)
        ax[i].plot(x, gaussian(x, mean, sigma, b, a), color = 'magenta') 
        ax[i].set_xlabel(r"Wavelength $(\AA)$")
        ax[i].set_ylabel("Relative Flux")
    
    return mean, sigma


def lc_fit(ax, i, plot = False, **kwargs):
    """
    - This function plots and fits the lightcurve data of each inputed supernova with
    a polynomial of given index.
    
    - It then returns the peak magnitude in the B band and the lightcurve parameter
    delta(m_15(B)), corresponding to the change in apparent B magnitude after 15
    days from the time of the peak magnitude, along with the errors in these values. 
    
    - The function works as follows:
            * Extracts the times, B magnitudes and B magnitude errors from dataset
            * Excludes data for where measurements were not taken on that day
            * Makes correction for time dilation, by dividing the times by 1+z
              and brings all times back to zero for simplicity
            * Fits the data over a defined range and outputs the fitting params
              of the polynomials along with their errors. 
            * Finds the peak magnitude by differentiating the fitting polynomials
              and finding the smallest positive root, corresponding to the time
              of the peak magnitude
            * Uses uncertainties package to input this root and output the peak
              magnitude and its associated error.
            * Finds the difference between this peak magnitude and the magnitude
              at 15 days after this peak, which is the lightcurve parameter
              delta(m_15(B)).
            * Plots the lightcurves as a function of time in days after the peak
              magnitude, and returns B_max and delta(m_15(B)).
    """

    lightcurves = kwargs['lightcurves']
    redshifts = kwargs['redshifts']
    error_redshifts = kwargs['error_redshifts']
    max_times = kwargs['max_times']
    fit_index = kwargs['fit_index']
    plot_range = kwargs['plot_range']
    updated_names = kwargs['updated_names']

    times = lightcurves[i].T[0]    #extracting values from lightcurves data
    Bmag = lightcurves[i].T[3]
    Bmag_error = lightcurves[i].T[4]


    index = np.where(Bmag < 99)    #finding index where measurements were taken
    times = times[index]    #removing values from arrays for days when no measurements were taken
    Bmag = Bmag[index]
    Bmag_error = Bmag_error[index]
    
    
    times = times/(1+redshifts[i])    #making correction for time dilation
    times -= times.min()    #bringing times back to zero for simplicity
    times_err = times * error_redshifts[i] #finding error in time data

    
    restricting_times = np.where(times < max_times[i])    #arguments associated with chosen range for each datset
    times = times[restricting_times]    #restricting arrays to chosen range with these arguments
    times_err = times_err[restricting_times]
    Bmag = Bmag[restricting_times]
    Bmag_error = Bmag_error[restricting_times]
    
    
    fit, covar = np.polyfit(times, Bmag, fit_index[i], cov = True)    #fitting poly over restricted data
    parameter_errors = np.sqrt(np.diag(covar))    #finding errors in poly parameters from covariance matrix
    
    
    #given fit parameters and their errors the output of function will be the value and its associated error
    polyw_unc = unc.wrap(np.poly1d)
    array_fit = unumpy.uarray(fit, parameter_errors)    #defining unumpy array with fit parameters and their errors


    y = np.poly1d(fit)     #defining poly without parameter errors
    y_prime = y.deriv()    #differentiating this polynomial
    roots = np.real(np.roots(y_prime))    #finding roots of polynomial to find peak
    roots = roots[np.where(roots > 0)]    #taking only roots with times greater than zero
    lc_max = np.min(roots)    #minimum root is associated with Bmagnitude maximum (minimum value)


    unc_fit = polyw_unc(array_fit)    #defining curve fit including parameter errors
    peak_bright = unc_fit(lc_max)    #peak brightness at Bmagnitude max (output is value and error)
    day15_bright = unc_fit(lc_max + 15) #finding Bmag 15 days after this peak magnitidue
    deltam_15 = day15_bright  - peak_bright #finding difference in these two Bmag values
    
    
    t = np.linspace(times.min(), plot_range[i], 100) #defining array for which to plot fit over
    
    times -= lc_max #setting times so that t=0 corresponds with B max
    
    if plot == True:
        ax[i].set_title(updated_names[i], fontsize = 15, fontweight = "bold")
        ax[i].set_xlabel(r"Days after $B_{max}$", fontsize = 12)
        ax[i].set_ylabel("B", fontsize = 12)
        ax[i].plot(t-lc_max, y(t), 'lightgreen', lw = 3)
        ax[i].errorbar(times, Bmag, xerr = times_err, yerr = Bmag_error, fmt = 'k.', markersize = 7)
        
        ax[i].axhline(peak_bright.nominal_value, color = "black", linestyle = "--", linewidth = 1)
        ax[i].axhline(day15_bright.nominal_value, color = "black", linestyle = "--", linewidth = 1)
        ax[i].axvline(0, color = "black", linestyle = "--", linewidth = 1)
        ax[i].axvline(15, color = "black", linestyle = "--", linewidth = 1)

        ax[i].set_ylim(Bmag.max()+0.5, Bmag.min()-1)
        ax[i].set_xlim(t.min()-lc_max -1, t.max()-lc_max+1)
    
    return peak_bright, deltam_15    


def abs_mag(app_mag, d):
    """
    - Calculates absolute magnitude and error in this value
      from the distance modulus 
    """
    y = unumpy.nominal_values(app_mag) - 5*np.log10(unumpy.nominal_values(d)) - 25 
    
    log_d_err = 5 * (1/np.log(10))*(unumpy.std_devs(d)/unumpy.nominal_values(d)) #calculating error in 5log(d)
    dy = np.sqrt(unumpy.std_devs(app_mag) ** 2 + log_d_err **2) #calculating cumulative error of absolute mag
    return y, dy


def lin_func(x, a, b):
    """
    - Simple linear function
    
    - Parameters: 
         x : 1d array of values
         a : slope 
         b : y-intercept
    """
    return a*x + b


def mb_corr(m, slope, m15B):
    """
    - Function used in correcting peak 
    magnitudes of supernovae light curves
    in order to be standardise the peaks
    """
    return m - slope * (m15B - 1.1)


def dmod_lc(mb, x1, c, **kwargs):
    """
    - Calculates the distance modulus from the lightcurve parameters
    using the modern method from ***.
    - Makes correction to magnitude M_B if stellar mass of the host
    galaxy is > 10e+10 solar masses
    - Computes the error in the distance modulus from Gauss' error law
    """

    LogMst = kwargs['LogMst']
    Mb = kwargs['Mb']
    del_m = kwargs['del_m']
    alpha = kwargs['alpha']
    beta = kwargs['beta']
    d_alpha = kwargs['d_alpha']
    d_x1 = kwargs['d_x1']
    d_beta = kwargs['d_beta']
    d_c = kwargs['d_c']
    dMb = kwargs['dMb']
    d_delm = kwargs['d_delm']
    d_mb = kwargs['d_mb']

    args = np.where(LogMst >= 10)  #arguments where log(mass_star) < 10
    M_b = np.zeros(len(mb))  #array of zeros 
    M_b += Mb  #updating all elements to equal Mb
    M_b[args] += del_m  #updating values of Mb with del m for args
     
    mu = mb - M_b + alpha * x1 - beta * c
    
    def err_mult(x, dx, y, dy):
        """
        Gauss' cumulitive error for 
        multiplying errors
        """
        return (x*y * np.sqrt((dx/x)**2 + (dy/y)**2))
    
    d_alx1 = err_mult(alpha, d_alpha, x1, d_x1)  
    d_btc = err_mult(beta, d_beta, c, d_c)
    dM_b = np.zeros(len(mb)) 
    dM_b += dMb
    dM_b[args] = np.sqrt(d_delm**2 + dM_b[args]**2)
    
    err_mu = np.sqrt(d_mb**2 + dM_b**2 + d_alx1**2 + d_btc**2) #error in the distance modulae
    
    return mu, err_mu


def dmod_Mpc(d):
    """
    Calculates distance modulus
    for part D in Mpc
    """
    return 5*np.log10(d) + 25 


def residuals(fit, data,s=1):
    """
    Define residuals for a model fit
    """
    return (data - fit) / s

