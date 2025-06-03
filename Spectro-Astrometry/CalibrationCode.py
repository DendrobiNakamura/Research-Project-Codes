#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 16:19:30 2021

@author: jim chung
"""
#The output files are: photometric light curves for each band, photometric data files in .dat format, 
#and calibrated spectra files.

#first import the relevant modules 

from astropy.io import fits #to read in fits file
import matplotlib.pyplot as plt #to plot
import numpy as np #data table handling
import pandas as pd #to read in csv file

from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import glob

#Define the location of photometric and spectroscopic output data from LCOGT
photo_loc = '/Data/All/*.fz'
spectra_loc = '/Data/All/*.fits'

#Define where you want the output files to go
outloc = '/Data/calibration_save/'

#Input the ra and dec of the objects. If new objects are used, add another row to the dictionary
ra_dec = {'NGC3783':[174.7571232337532, -37.7386134993465],
          'IC4329A':[207.3302530340835, -30.3095067046753],
          'IRAS09149-6206':[139.039025905958,-62.3248777195977],
          'Fairall51':[281.224955700999,-62.3646865952261]}
#Input the redshift of the objects. If new objects are used, add another row to the dictionary
redshift = {'NGC3783':0.010,
            'IC4329A':0.016,
            'IRAS09149-6206':0.057,
            'Fairall51':0.014}

#Zero points information. Add new row if new telescope is used
zero_points = {'LCOGT node at SAAO':{'gp':23.00,'ip':22.10,'rp':22.75},
               'LCOGT node at Cerro Tololo Inter-American Observatory':{'gp':23.05,'ip':22.15,'rp':22.80},
               'LCOGT node at Siding Spring Observatory':{'gp':23.05,'ip':22.10,'rp':22.80}}

#For spectrum calibration, we can either match the spectrum magnitude to the nearest photometric magnitude
# or interpolate the photometric light curve. Put 'nearest' for matching to the nearest photometric measurement
# put'linear' for interpolation.
photo_matching = 'linear'

#The bands used and their centers. If new bands are used, need to modify the entire code.
bands = ['gp', 'rp', 'ip']
centers = [4770, 6215, 7545]

#Location of transmission functions
filters = {'gp': '/Users/jchung1/Desktop/Uni Works/ASC/Brad Tucker/Data/All/sdss.gp.txt',
           'rp': '/Users/jchung1/Desktop/Uni Works/ASC/Brad Tucker/Data/All/sdss.rp.txt',
           'ip': '/Users/jchung1/Desktop/Uni Works/ASC/Brad Tucker/Data/All/sdss.ip.txt'}

#A dictionary used to sort the files. If a new object is added, simply add another element to the dictionary.

super_array = {'NGC3783':{'gp':np.array([[],[],[]]),
                          'rp':np.array([[],[],[]]),
                          'ip':np.array([[],[],[]])},
               'IC4329A':{'gp':np.array([[],[],[]]),
                          'rp':np.array([[],[],[]]),
                          'ip':np.array([[],[],[]])},
               'IRAS09149-6206':{'gp':np.array([[],[],[]]),
                                 'rp':np.array([[],[],[]]),
                                 'ip':np.array([[],[],[]])},
               'Fairall51':{'gp':np.array([[],[],[]]),
                            'rp':np.array([[],[],[]]),
                            'ip':np.array([[],[],[]])}}

#Reads in all the files and sorting them



hdu = []
for each_file in glob.glob(photo_loc):
    hdu.append(fits.open(each_file))
    
All_hdu_header = []
All_cats_data = []


#store the header and the data in a separate list
for i in hdu:
    All_hdu_header.append(i[1].header)
    All_cats_data.append(i[2].data)
    
#combine the list to have easy access to to the data and header
All_photo = [All_cats_data,All_hdu_header]



#Checking if need to add zero point information if new telescope used.
for i in range(len(All_photo[0])):
    if All_photo[1][i]['TELID']!='1m0a':
        print('New zero point information needs to be added at zero_points dictionary defined at the beginning')
        print(All_photo[1][i]['SITE'],All_photo[1][i]['TELID'])


# Define where the spectrum is stored
spectra = []
for each_file in glob.glob(spectra_loc):
    spectra.append(fits.open(each_file))
    
#creating lists to store useful information
spectra_header = []
spectra_data = []
maxwl = []
minwl = []
spectra_wavelength = []
spectra_variance = []
spectra_date = []
spectra_galaxy = []

for i in range(len(spectra)):
    spectra_header.append(spectra[i][0].header)
    spectra_data.append(spectra[i][0].data[0][0])
    spectra_variance.append(spectra[i][0].data[3][0])
    spectra_date.append(spectra_header[i]['MJD-OBS'])
    spectra_galaxy.append(spectra_header[i]['OBJECT'])
    maxwl.append(float(spectra_header[i]['XMAX']))
    minwl.append(float(spectra_header[i]['XMIN']))
    spectra_wavelength.append(np.linspace(minwl[i],maxwl[i],len(spectra_data[i])))

#combining the lists for easy access 
spectra = np.array([spectra_data,spectra_variance,spectra_wavelength,spectra_date,spectra_galaxy,spectra_header],dtype = object)

#0: flux
#1: var
#2: wavelength
#3: date
#4: galaxy
#5: header information

#sort by date
spectra_ind = np.argsort(spectra[3])
spectra[0] = spectra[0][spectra_ind]
spectra[1] = spectra[1][spectra_ind]
spectra[2] = spectra[2][spectra_ind]
spectra[3] = spectra[3][spectra_ind]
spectra[4] = spectra[4][spectra_ind]
spectra[5] = spectra[5][spectra_ind]

#Defining all the functions that will be used

#Photometry
#Look for the object in the photometry file
def fluxes_and_fluxerr(input_object):
    fluxes = []
    fluxerr = []
    for j in range(len(input_object[0])): 
        name = input_object[1][j]['OBJECT']
        closest_object = 0.01
        for i in range(len(input_object[0][j]['ra'])):
            distance = ((input_object[0][j]['ra'][i]-ra_dec[name][0])**2 + 
                        (input_object[0][j]['dec'][i]-ra_dec[name][1])**2)
            if distance < closest_object:
                closest_object, closest_flux, closest_flux_err = (distance, 
                                                                    input_object[0][j]['flux'][i],
                                                                    input_object[0][j]['fluxerr'][i])
        fluxes.append(closest_flux)
        fluxerr.append(closest_flux_err)
    return (fluxes,fluxerr)


#Finding the Zero Point Magnitude for each telescope
def zeropoint (input_object):
    zp = []
    for i in range(len(input_object[1])):
        observatory = input_object[1][i]['SITE']
        lens = input_object[1][i]['FILTER']
        
        zp.append(zero_points[observatory][lens])
    return (zp)


#so now we have to calculate the mag and mag errs of each object in cat in the three filters
# and then we can find the difference in mag from the OzDES catalogue. 
def mag_err_t(galaxy): 
    '''Returns the mag and mag error as a numpy array given a mag zeropoint, array of fluxes and flux errors.'''
    mag_zp = zeropoint(galaxy)
    fluxes,flux_errs = fluxes_and_fluxerr(galaxy)
    
    mags = []
    mag_errs = []
    time = []
    for i in range(0,len(fluxes)):
        if fluxes[i] > 0: 
            mag = mag_zp[i] - (2.5*np.log10(fluxes[i]))
            mag_err = (2.5/np.log(10)) * (flux_errs[i]/fluxes[i])
            mags.append(mag)
            mag_errs.append(mag_err)
            time.append(galaxy[1][i]['MJD-OBS'])
        else: 
            mag = 99.0 
            mag_err = 99.0 
            mags.append(mag)
            mag_errs.append(mag_err)
    return mags,mag_errs,time
    
    

    
    

#spectrum
#Read and store the transmission function
class filterCurve:
    """A filter"""

    def __init__(self):
        self.wave = np.array([], 'float')
        self.trans = np.array([], 'float')
        return

    def read(self, file):
        factor = 10 #Transmission file given in nm
        
        file = open(file, 'r')
        for line in file.readlines():
            if line[1] != "l":
                entries = line.split()
                self.wave = np.append(self.wave, float(entries[0]))
                self.trans = np.append(self.trans, float(entries[1]))
        file.close()
        # We use Angstroms for the wavelength in the filter transmission file
        self.wave = self.wave * factor
        return
    
def readFilterCurves(bands, filters):
    filterCurves = {}
    for f in bands:
        filterCurves[f] = filterCurve()
        filterCurves[f].read(filters[f])

    return filterCurves

#Find the absolute magnitudes of the spectrum in each band
def computeABmag(trans_flux, trans_wave, spectral_wavelength, spectral_flux, spectral_var):
    # Takes and returns variance
    # trans_ : transmission function data
    # tmp_ : spectral data

    # trans/tmp not necessarily defined over the same wavelength range
    # first determine the wavelength range over which both are defined
    minV = min(trans_wave)
    if minV < min(spectral_wavelength):
        minV = min(spectral_wavelength)
    maxV = max(trans_wave)
    if maxV > max(spectral_wavelength): 
        maxV = max(spectral_wavelength)

    interp_wave = []
    spectral_flux2 = []
    spectral_var2 = []

    # Make new vectors for the flux just using that range (assuming spectral binning)

    for i in range(len(spectral_wavelength)):
        if minV < spectral_wavelength[i] < maxV:
            interp_wave.append(spectral_wavelength[i])
            spectral_flux2.append(spectral_flux[i])
            spectral_var2.append(spectral_var[i])

    # interpolate the transmission function onto this range
    # the transmission function is interpolated as it is generally much smoother than the spectral data
    trans_flux2 = interp1d(trans_wave, trans_flux)(interp_wave)

    # And now calculate the magnitude and uncertainty

    c = 2.992792e18  # Angstrom/s
    Num = np.nansum(spectral_flux2 * trans_flux2 * interp_wave)
    Num_var = np.nansum(spectral_var2 * (trans_flux2 * interp_wave) ** 2)
    Den = np.nansum(trans_flux2 / interp_wave)

    with np.errstate(divide='raise'):
        try:
            magAB = -2.5 * np.log10(Num / Den / c) - 48.60
            magABvar = 1.17882 * Num_var / (Num ** 2)
        except FloatingPointError:
            magAB = 99.
            magABvar = 99.

    return magAB, magABvar


#Finds flux of nearest photometry and use that value
def nearest_photo(photo, spectral_mjd, bands):
    mags = np.zeros(3)
    errs = np.zeros(3)

    #sort photos by filters:
    g_photo = [[],[],[]] #gives magnitude, variance, and date
    r_photo = [[],[],[]]
    i_photo = [[],[],[]]
    
    #photo:
    #0: mag
    #1: err
    #2: time
    #3: filter
    #4: galaxy
    
    for i in range(len(photo[3])):
        if photo[3][i] == bands[0]:
            g_photo[0].append(photo[0][i])
            g_photo[1].append(photo[1][i])
            g_photo[2].append(photo[2][i])
        if photo[3][i] == bands[1]:
            r_photo[0].append(photo[0][i])
            r_photo[1].append(photo[1][i])
            r_photo[2].append(photo[2][i])
        if photo[3][i] == bands[2]:
            i_photo[0].append(photo[0][i])
            i_photo[1].append(photo[1][i])
            i_photo[2].append(photo[2][i])
    
    for l in range(len(g_photo[2]) - 1):
        if g_photo[2][l] < spectral_mjd < g_photo[2][l + 1]:
            nearest = [g_photo[2][l + 1] - spectral_mjd,spectral_mjd - g_photo[2][l]]
            if np.min(nearest) == nearest[0]:
                mags[0] = g_photo[0][l+1]
            if np.min(nearest) == nearest[1]:
                mags[0] = g_photo[0][l]
            errs[0] = np.abs(r_photo[0][l+1]-g_photo[0][l])
            
        if r_photo[2][l] < spectral_mjd < r_photo[2][l + 1]:
            nearest = [r_photo[2][l + 1] - spectral_mjd,spectral_mjd - r_photo[2][l]]
            if np.min(nearest) == nearest[0]:
                mags[1] = r_photo[0][l+1]
            if np.min(nearest) == nearest[1]:
                mags[1] = r_photo[0][l]
            errs[1] = np.abs(r_photo[0][l+1]-r_photo[0][l])
            
        if i_photo[2][l] < spectral_mjd < i_photo[2][l + 1]:
            nearest = [i_photo[2][l + 1] - spectral_mjd,spectral_mjd - i_photo[2][l]]
            if np.min(nearest) == nearest[0]:
                mags[2] = i_photo[0][l+1]
            if np.min(nearest) == nearest[1]:
                mags[2] = i_photo[0][l]
            errs[2] = np.abs(i_photo[0][l+1]-i_photo[0][l])
   
    #accounting for the situation if the date of the spectrum is later than the latest photo, then choose latest photo
    #error will be the mag difference between last photo and second last photo
    if mags[0]==0:
        mags[0] = g_photo[0][-1]
        errs[0] = np.abs(g_photo[0][-1]-g_photo[0][-2])
        mags[1] = r_photo[0][-1]
        errs[1] = np.abs(r_photo[0][-1]-r_photo[0][-2])
        mags[2] = i_photo[0][-1]
        errs[2] = np.abs(i_photo[0][-1]-i_photo[0][-2])
        
    
    return mags, errs #list[g,r,i]


#Next two functions are the interpolation method
def interpolatePhot(x, y, s, val):
    # takes sigma returns variance
    # x - x data points (list)
    # y - y data points (list)
    # s - sigma on y data points (list)
    # val - x value to interpolate to (number)

    mag = y[0] + (val - x[0]) * (y[1] - y[0]) / (x[1] - x[0])
    err = s[0] ** 2 + (s[0] ** 2 + s[1] ** 2) * ((val - x[0]) / (x[1] - x[0])) ** 2
    return mag, err

def des_photo(photo, spectral_mjd, bands):

    """Takes in an mjd from the spectra, looks through a light curve file to find the nearest photometric epochs and
    performs linear interpolation to get estimate at date, return the photo mags.   """

    mags = np.zeros(3)
    errs = np.zeros(3)
    
            
    #sort photos by filters:
    g_photo = [[],[],[]] #gives magnitude, variance, and date
    r_photo = [[],[],[]]
    i_photo = [[],[],[]]
    
    
    for i in range(len(photo[3])):
        if photo[3][i] == bands[0]:
            g_photo[0].append(photo[0][i])
            g_photo[1].append(photo[1][i])
            g_photo[2].append(photo[2][i])
        if photo[3][i] == bands[1]:
            r_photo[0].append(photo[0][i])
            r_photo[1].append(photo[1][i])
            r_photo[2].append(photo[2][i])
        if photo[3][i] == bands[2]:
            i_photo[0].append(photo[0][i])
            i_photo[1].append(photo[1][i])
            i_photo[2].append(photo[2][i])          
    #photo:
    #0: mag
    #1: err
    #2: time
    #3: filter
    #4: galaxy
    
    g_date_v = np.zeros(2)
    r_date_v = np.zeros(2)
    i_date_v = np.zeros(2)
    g_mag_v = np.zeros(2)
    r_mag_v = np.zeros(2)
    i_mag_v = np.zeros(2)
    g_err_v = np.zeros(2)
    r_err_v = np.zeros(2)
    i_err_v = np.zeros(2)
    
    
    for l in range(len(g_photo[2]) - 1):
        if g_photo[2][l] < spectral_mjd < g_photo[2][l + 1]:
            g_date_v = np.array([g_photo[2][l], g_photo[2][l + 1]])
            g_mag_v = np.array([g_photo[0][l], g_photo[0][l + 1]])
            g_err_v = np.array([g_photo[1][l], g_photo[1][l + 1]])
        if r_photo[2][l] < spectral_mjd < r_photo[2][l + 1]:
            r_date_v = np.array([r_photo[2][l], r_photo[2][l + 1]])
            r_mag_v = np.array([r_photo[0][l], r_photo[0][l + 1]])
            r_err_v = np.array([r_photo[1][l], r_photo[1][l + 1]])
        if i_photo[2][l] < spectral_mjd < i_photo[2][l + 1]:
            i_date_v = np.array([i_photo[2][l], i_photo[2][l + 1]])
            i_mag_v = np.array([i_photo[0][l], i_photo[0][l + 1]])
            i_err_v = np.array([i_photo[1][l], i_photo[1][l + 1]])
    if g_mag_v[0] == 0: #accounting for the situation where the date of spectrum is later than the latest photo
        g_date_v = np.array([g_photo[2][-1],g_photo[2][-1]+1e-10]) #add 1e-10 to avoid dividing by 0 during interpolation
        g_mag_v = np.array([g_photo[0][-1],g_photo[0][-1]+1e-10])
        g_err_v = np.array([g_photo[1][-1],g_photo[1][-1]+1e-10])
        r_date_v = np.array([r_photo[2][-1],r_photo[2][-1]+1e-10])
        r_mag_v = np.array([r_photo[0][-1],r_photo[0][-1]+1e-10])
        r_err_v = np.array([r_photo[1][-1],r_photo[1][-1]+1e-10])
        i_date_v = np.array([i_photo[2][-1],i_photo[2][-1]+1e-10])
        i_mag_v = np.array([i_photo[0][-1],i_photo[0][-1]+1e-10])
        i_err_v = np.array([i_photo[1][-1],i_photo[1][-1]+1e-10])
        

    mags[0], errs[0] = interpolatePhot(g_date_v, g_mag_v, g_err_v, spectral_mjd)
    mags[1], errs[1] = interpolatePhot(r_date_v, r_mag_v, r_err_v, spectral_mjd)
    mags[2], errs[2] = interpolatePhot(i_date_v, i_mag_v, i_err_v, spectral_mjd)


    return mags, errs



# -------------------------------------------------- #
# ---------------- scale_factors  ------------------ #
# -------------------------------------------------- #
# Calculates the scale factor and variance needed to #
# change spectroscopically derived magnitude to the  #
# observed photometry.                               #
# -------------------------------------------------- #

def scale_factors(mag_diff, mag_diff_var):
    # takes and returns variance

    flux_ratio = np.power(10., 0.4 * mag_diff)  # f_synthetic/f_photometry
    scale_factor = (1. / flux_ratio)
    scale_factor_sigma = mag_diff_var * (scale_factor * 0.4 * 2.3) ** 2   # ln(10) ~ 2.3

    return scale_factor, scale_factor_sigma



# -------------------------------------------------- #
# ---------------- scaling_Matrix ------------------ #
# -------------------------------------------------- #
# finds the nearest photometry and interpolates mags #
# to find values at the time of the spectroscopic    #
# observations.  Calculates the mag that would be    #
# observed from the spectra and calculates the scale #
# factor to bring them into agreement. Saves the     #
# data in the scaling matrix.                        #
# -------------------------------------------------- #

def scaling_Matrix(spectra, photo, bands, filters, interpFlag):
    # scale factors for each extension saved in the following form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagError = scaling[9,:] (interpolated from neighbouring observations)
    # rMag = scaling[10,:]   rMagError = scaling[11,:]
    # iMag = scaling[12,:]   iMagError = scaling[13,:]
    
    scaling = np.zeros((14, len(spectra[0])))

            
    from_spectrum = np.zeros((3, len(spectra[0])))
    from_photo = np.zeros((3, len(spectra[0])))

    from_spectrum_err = np.zeros((3, len(spectra[0])))
    from_photo_err = np.zeros((3, len(spectra[0])))

    filterCurves = readFilterCurves(bands, filters)

    #Find spectral magnitude 
    for i in range(len(spectra[0])):
        
        from_spectrum[0, i], from_spectrum_err[0, i] = computeABmag(filterCurves[bands[0]].trans, 
                                                                    filterCurves[bands[0]].wave,
                                                                    spectra[2][i], spectra[0][i], #0:flux,1:err,2:wl
                                                                    spectra[1][i])
        from_spectrum[1, i], from_spectrum_err[1, i] = computeABmag(filterCurves[bands[1]].trans, 
                                                                    filterCurves[bands[1]].wave,
                                                                    spectra[2][i], spectra[0][i],
                                                                    spectra[1][i])
        from_spectrum[2, i], from_spectrum_err[2, i] = computeABmag(filterCurves[bands[2]].trans, 
                                                                    filterCurves[bands[2]].wave,
                                                                    spectra[2][i], spectra[0][i],
                                                                    spectra[1][i])

    
        #Find photometry magnitude

        if interpFlag == 'nearest':
            from_photo[:,i], from_photo_err[:,i] = nearest_photo(photo, spectra[3][i], bands)

            scaling[8, :] = from_photo[0, :]
            scaling[10, :] = from_photo[1, :]
            scaling[12, :] = from_photo[2, :]

            scaling[9, :] = from_photo_err[0, :]
            scaling[11, :] = from_photo_err[1, :]
            scaling[13, :] = from_photo_err[2, :]


            # Find DES photometry
        elif interpFlag == 'linear':
            from_photo[:,i], from_photo_err[:,i] = des_photo(photo, spectra[3][i], bands)

            scaling[8,:] = from_photo[0,:]
            scaling[10,:] = from_photo[1,:]
            scaling[12,:] = from_photo[2,:]

            scaling[9,:] = from_photo_err[0,:]
            scaling[11,:] = from_photo_err[1,:]
            scaling[13,:] = from_photo_err[2,:]

            # Find Scale Factor

        scaling[0,:], scaling[3,:] = scale_factors(np.array(from_photo[0,:]) - np.array(from_spectrum[0,:]),
                                                     np.array(from_photo_err[0,:]) + np.array(from_spectrum_err[0,:]))
        scaling[1,:], scaling[4,:] = scale_factors(np.array(from_photo[1,:]) - np.array(from_spectrum[1,:]),
                                                     np.array(from_photo_err[1,:]) + np.array(from_spectrum_err[1,:]))
        scaling[2,:], scaling[5,:] = scale_factors(np.array(from_photo[2,:]) - np.array(from_spectrum[2,:]),
                                                     np.array(from_photo_err[2,:]) + np.array(from_spectrum_err[2,:]))

    return scaling



# -------------------------------------------------- #
# ----------------- warp_spectra  ------------------ #
# -------------------------------------------------- #
# Fits polynomial to scale factors and estimates     #
# associated uncertainties with gaussian processes.  #
# If the plotFlag variable is not False it will save #
# some diagnostic plots.                             #
# -------------------------------------------------- #

def warp_spectra(scaling, scaleErr, flux, variance, wavelength, centers):

    # associate scale factors with centers of bands and fit 2D polynomial to form scale function.
    scale = InterpolatedUnivariateSpline(centers, scaling, k=2)
    fluxScale = flux * scale(wavelength)

    # add in Gaussian process to estimate uncertainties, /10**-17 because it gets a bit panicky if you use small numbers
    stddev = (scaleErr ** 0.5) / 10 ** -17
    scale_v = scaling / 10 ** -17

    kernel = kernels.RBF(length_scale=300, length_scale_bounds=(.01, 2000.0))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=stddev**2)

    xprime = np.atleast_2d(centers).T
    yprime = np.atleast_2d(scale_v).T

    gp.fit(xprime, yprime)
    xplot_prime = np.atleast_2d(wavelength).T
    y_pred, sigma = gp.predict(xplot_prime, return_std=True)

    y_pred = y_pred[:,0]

    sigModel = (sigma/y_pred)*scale(wavelength)

    # now scale the original variance and combine with scale factor uncertainty
    varScale = variance * pow(scale(wavelength), 2) + sigModel ** 2


    return fluxScale, varScale, scale(wavelength)



# -------------------------------------------------- #
# ------------------- calibSpec -------------------- #
# -------------------------------------------------- #
# This function does the bulk of the work.  It will  #
# 1) determine extensions which can be calibrated    #
# 2) calculate the scale factors                     #
# 3) calculate the warping function                  #
# 4) output new fits file with scaled spectra        #
# -------------------------------------------------- #

def calibSpec(spectra, photo, bands, filters, centers, interpFlag):
    # Assumes scaling given is of the form
    # gScale = scaling[0,:]   gError = scaling[3,:]
    # rScale = scaling[1,:]   rError = scaling[4,:]
    # iScale = scaling[2,:]   iError = scaling[5,:]
    # inCoaddWeather = scaling[6,:]
    # inCoaddPhoto = scaling[7,:]
    # gMag = scaling[8,:]   gMagErr = scaling[9,:]
    # rMag = scaling[10,:]  rMagErr = scaling[11,:]
    # iMag = scaling[12,:]  iMagErr = scaling[13,:]
    # interpFlag = 'nearest' or 'linear'

    # We calculate the scale factors
    
    scaling = scaling_Matrix(spectra,photo, bands, filters, interpFlag)
    
    warped_spectra = []
    warped_spectra_var = []
    scaling_func = []

    # Warp the data
    for s in range(len(spectra[3])):
        flux, var, sc_func = warp_spectra(scaling[0:3, s], scaling[3:6, s], spectra[0][s],
                                                                  spectra[1][s], spectra[2][s], centers)
        warped_spectra.append(flux)
        warped_spectra_var.append(var)
        scaling_func.append(sc_func)
        
    return warped_spectra,warped_spectra_var,scaling



# -------------------------------------------------- #
# ------------ create_output_single  --------------- #
# -------------------------------------------------- #
# Outputs the warped spectra to a new fits file.     #
# -------------------------------------------------- #
def create_output_single(obj_name, scaling, spectra, spectraName, photoName, outBase, redshift):
    
    outName = outBase + obj_name + "_scaled.fits"

    hdu = fits.PrimaryHDU()
    hdul = fits.HDUList([hdu])


    for i in range(len(warped_spectra_var)):
        header = fits.Header()
        header['OBJECT'] = spectra[5][0]['OBJECT']
        header['RA'] = spectra[5][i]['RA']
        header['DEC'] = spectra[5][i]['DEC']
        header['FIELD'] = ''
        header['CRPIX1'] = spectra[5][i]['CRPIX1']
        header['CRVAL1'] = spectra[5][i]['CRVAL1']
        header['CDELT1'] = float("{:.3f}".format(spectra[2][i][1]-spectra[2][i][0])) #increment of wavelength
        header['CTYPE1'] = 'wavelength'
        header['CUNIT1'] = 'angstrom'
        header['EPOCHS'] = len(spectra[3])
        header['z'] = redshift
        header['AVGDATE'] = spectra[5][i]['MJD-OBS']

        # save the names of the input data and the extensions ignored
        header['SFILE'] = spectraName
        header['PFILE'] = photoName
        header['NOPHOTO'] = ''
        header['BADQC'] = ''

        # save the original spectrum's extension number and some other details
        header["EXT"] = i #index number
        header["UTMJD"] = spectra[3][i] #mjd
        header["EXPOSE"] = ''
        header["QC"] = ''

        # save scale factors/uncertainties
        header["SCALEG"] = scaling[0, i]
        header["ERRORG"] = scaling[3, i]
        header["SCALER"] = scaling[1, i]
        header["ERRORR"] = scaling[4, i]
        header["SCALEI"] = scaling[2, i]
        header["ERRORI"] = scaling[5, i]

        # save photometry/uncertainties used to calculate scale factors
        header["MAGG"] = scaling[8, i]
        header["MAGUG"] = scaling[9, i]
        header["MAGR"] = scaling[10, i]
        header["MAGUR"] = scaling[11, i]
        header["MAGI"] = scaling[12, i]
        header["MAGUI"] = scaling[13, i]

        hdul[0].header['SOURCE'] = spectra[5][0]['OBJECT']
        hdul[0].header['RA'] = spectra[5][i]['RA']
        hdul[0].header['DEC'] = spectra[5][i]['DEC']
        hdul[0].header['FIELD'] = ''
        hdul[0].header['CRPIX1'] = spectra[5][i]['CRPIX1']
        hdul[0].header['CRVAL1'] = spectra[5][i]['CRVAL1']
        hdul[0].header['CDELT1'] = float("{:.3f}".format(spectra[2][i][1]-spectra[2][i][0]))
        hdul[0].header['CTYPE1'] = 'wavelength'
        hdul[0].header['CUNIT1'] = 'angstrom'
        hdul[0].header['EPOCHS'] = len(spectra[3])
        hdul[0].header['z'] = redshift
        hdul[0].header['AVGDATE'] = spectra[5][i]['MJD-OBS']

        # save the names of the input data and the extensions ignored
        hdul[0].header['SFILE'] = spectraName
        hdul[0].header['PFILE'] = photoName
        hdul[0].header['NOPHOTO'] = ''
        hdul[0].header['BADQC'] = ''

        # save the original spectrum's extension number and some other details
        hdul[0].header["EXT"] = i
        hdul[0].header["UTMJD"] = spectra[3][i]
        hdul[0].header["EXPOSE"] = ''
        hdul[0].header["QC"] = ''

        # save scale factors/uncertainties
        hdul[0].header["SCALEG"] = scaling[0, i]
        hdul[0].header["ERRORG"] = scaling[3, i]
        hdul[0].header["SCALER"] = scaling[1, i]
        hdul[0].header["ERRORR"] = scaling[4, i]
        hdul[0].header["SCALEI"] = scaling[2, i]
        hdul[0].header["ERRORI"] = scaling[5, i]

        # save photometry/uncertainties used to calculate scale factors
        hdul[0].header["MAGG"] = scaling[8, i]
        hdul[0].header["MAGUG"] = scaling[9, i]
        hdul[0].header["MAGR"] = scaling[10, i]
        hdul[0].header["MAGUR"] = scaling[11, i]
        hdul[0].header["MAGI"] = scaling[12, i]
        hdul[0].header["MAGUI"] = scaling[13, i]
        hdul[0].data = warped_spectra[0] #warped_spectra is a global variable which will be defined later
        hdul.append(fits.ImageHDU(data=warped_spectra[i], header=header))
        hdul.append(fits.ImageHDU(data=warped_spectra_var[i], header=header))
        hdul.append(fits.ImageHDU(data=[], header=header))

    hdul.writeto(outName,overwrite=True)
    hdul.close()
    return


#runs the code
photo_mag, photo_err, photo_time = mag_err_t(All_photo) 

# and saves the magnitudes, err, time into a new list
photo = np.array([photo_mag, photo_err, photo_time])



#order files by date
photo_ind = np.argsort(photo[2])
photo[0] = photo[0][photo_ind]
photo[1] = photo[1][photo_ind]
photo[2] = photo[2][photo_ind]

#The original All_photo file which contain all the headers
Al_photo = np.array(All_photo[1],dtype=object)
Al_photo = Al_photo[photo_ind]


#sorting by galaxy and filter.
for i in range(len(photo[0])):
    super_array[Al_photo[i]['OBJECT']][Al_photo[i]['FILTER']] = np.concatenate(
        (super_array[Al_photo[i]['OBJECT']][Al_photo[i]['FILTER']] , np.array([photo[:,i]]).T), axis=1)
    
    
    
    
#Plotting to see and save plots
for gal in super_array:
    fig = plt.figure()
    gs = fig.add_gridspec(3, hspace=0)
    axs = gs.subplots(sharex=True)
    axs[0].errorbar(super_array[gal]['gp'][2],super_array[gal]['gp'][0],
                    super_array[gal]['gp'][1],fmt='og',ecolor='black',capsize=3)
    axs[0].set(ylabel='', xlabel='', title= gal+' Photometric Light Curves')
    axs[0].legend(['g'])
    axs[1].errorbar(super_array[gal]['rp'][2],super_array[gal]['rp'][0],
                    super_array[gal]['rp'][1],fmt='or',ecolor='black',capsize=3)
    axs[1].set(ylabel='Magnitude', xlabel='', title= '')
    axs[1].legend(['r'])
    axs[2].errorbar(super_array[gal]['ip'][2],super_array[gal]['ip'][0],
                    super_array[gal]['ip'][1],fmt='ob',ecolor='black',capsize=3)
    axs[2].set(ylabel='', xlabel='Days (MJD)', title= '')
    axs[2].legend(['r'])
    fig.set_figheight(7)
    for ax in axs:
        ax.label_outer()
    plt.savefig(outloc+gal+'_Photometric_Combined_LC.pdf')
    plt.close()

    plt.errorbar(super_array[gal]['gp'][2],super_array[gal]['gp'][0],
                    super_array[gal]['gp'][1],fmt='og',ecolor='black',capsize=3)
    plt.ylabel('Magnitude')
    plt.xlabel('Days (MJD)')
    plt.legend(['g'])
    plt.title(gal + ' PhotometricLC g')
    plt.savefig(outloc+gal+'_Photometric_LC_g.pdf')
    plt.close()
    
    plt.errorbar(super_array[gal]['rp'][2],super_array[gal]['rp'][0],
                    super_array[gal]['rp'][1],fmt='or',ecolor='black',capsize=3)
    plt.ylabel('Magnitude')
    plt.xlabel('Days (MJD)')
    plt.legend(['r'])
    plt.title(gal + ' PhotometricLC r')
    plt.savefig(outloc+gal+'_Photometric_LC_r.pdf')
    plt.close()
    
    plt.errorbar(super_array[gal]['ip'][2],super_array[gal]['ip'][0],
                    super_array[gal]['ip'][1],fmt='ob',ecolor='black',capsize=3)
    plt.ylabel('Magnitude')
    plt.xlabel('Days (MJD)')
    plt.legend(['i'])
    plt.title(gal + ' PhotometricLC i')
    plt.savefig(outloc+gal+'_Photometric_LC_i.pdf')
    plt.close()
    
for n in super_array:
    ob_mag = np.array([super_array[n]['gp'][0],super_array[n]['rp'][0],super_array[n]['ip'][0]],dtype=object).ravel().tolist()
    ob_err = np.array([super_array[n]['gp'][0],super_array[n]['rp'][0],super_array[n]['ip'][0]],dtype=object).ravel().tolist()
    ob_mjd = np.array([super_array[n]['gp'][0],super_array[n]['rp'][0],super_array[n]['ip'][0]],dtype=object).ravel().tolist()
    ob_filters = []
    for j in bands:
        for i in range(len(super_array[n]['gp'][0])):
            ob_filters.append(j)
            
    ob_file = pd.DataFrame(data={'MJD': ob_mjd, 'MAG': ob_mag, 'MAGERR':ob_err, 'BAND':ob_filters})
    ob_file.to_csv(outloc+n+'_lc.dat', index=False)
    
    #read file
    OBJECT = pd.read_csv(outloc+n+'_lc.dat')
    obj_photo = np.array([OBJECT['MAG'],OBJECT['MAGERR'],OBJECT['MJD'],OBJECT['BAND']])
    
    #Separate by galaxy
    obj_spectra = np.array([[],[],[],[],[],[]])

    for i in range(len(spectra[0])):
        if spectra[4][i] == n:
            obj_spectra = np.concatenate((obj_spectra , np.array([spectra[:,i]]).T), axis=1)
    
    warped_spectra, warped_spectra_var, scaling = calibSpec(obj_spectra, obj_photo, bands, filters, centers, photo_matching)
    
    
    #Make sure all the spectra have the same length by setting the length to 3000 for all spectra

    for j in range(len(warped_spectra)):
        warped_spectra[j] = warped_spectra[j][:3000]
        warped_spectra_var[j] = warped_spectra_var[j][:3000]


    spectraName = spectra[4]
    photoName = spectra[4]
    
    
    
    create_output_single(n, scaling, obj_spectra, n, n, outloc, redshift[n])
print('Done!')
