#!/usr/bin/env python3
#
# Function for generating noisefiles and SNR files given an instrument config object
# Authors: By Ann Raiho, Adapted from David Thompson
# Import dependencies requires installation of the isofit repository
import sys
import numpy as np
import os
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
from sbg_utils import load_chn, rule_07, blackbody, local_fit
from isofit.core.common import load_spectrum, resample_spectrum, load_wavelen
import logging

def calculate_instrument_model(instType, instrument_configs, mainfile, output_noisefile, output_snrfile):

    if instType == 'sbg_cbe_chromaA':
        import instrument_chroma_a as instrument

    if instType == 'sbg_cbe_chromaD':
        import instrument_chroma_d as instrument

    # Load MODTRAN 6.0 precalculated coefficients
    coszen = np.cos(45.0 / 360.0 * 2.0 * np.pi)
    m6file = os.path.join(mainfile, 'modtran/sbg_reference.chn')

    # Wavelength, irradiance, path reflectance, atmospehric
    # transmittance, spherical albedo, and upward transmittance
    wl_m6, irr, rhoatm, transm, sphalb, tup = load_chn(m6file, coszen)
    print(wl_m6.shape)

    # Load instrument wavelength grid
    wvfile = os.path.join(mainfile, 'wavelengths.txt')
    q, wl, fwhm = np.loadtxt(wvfile).T * 1000.0

    print(wl.shape)

    if wl.shape > wl_m6.shape:
        print("wavelength file longer than modtran wavelengths.")

    if wl.shape < wl_m6.shape:
        print("wavelength file shorter than modtran wavelengths.")

    # Load CBE Case
    sbg_CBE = instrument.Instrument(instrument_configs)

    # Create SNR/radiance pairs
    # NEdL = Noise-equivalent change in radiance
    radiances, nedls, snrs, snrx = [], [], [], []

    # Iterate over four reflectance magnitudes
    for si in range(4):

        rfl = si * 0.1

        # Calculate radiance at sensor using the Vermote
        # et al. relation.  According to Jeff Dozier this
        # dates to Chandrasekhar
        rhotoa = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
        rdn = rhotoa / np.pi * (irr * coszen)

        # Calculate the associated SNR from our most accurate
        # component-wise instrument noise model
        snr = sbg_CBE.snr(rdn)

        # build our lists
        radiances.append(rdn)
        nedls.append(rdn / snr)
        snrs.append(snr)

    # Form Python arrays
    radiances = np.array(radiances)
    snrs = np.array(snrs)
    nedls = np.array(nedls)
    snrx = np.array(snrx)

    # Now use nonlinear optimization to fit our three-coefficient model.
    # Initial guess
    x0 = np.array([0.2, 0.1, 0.1])

    # Error between predicted and actual noise-equivalent change in radiance
    # Given a particular radiance value and a three-coefficient state vector

    # Now fit three-coefficient models to all wavelengths independently
    params, errs = [], []

    # Call least-squares optimization for each wavelength
    for i, w in enumerate(wl):
        args = (radiances[:, i], nedls[:, i])
        res = least_squares(err, x0, args=args, xtol=1e-5)
        x = res.x
        params.append(x)
        errs.append([np.sqrt(sum(pow(err(res.x, args[0], args[1]), 2)))])

    # Write the noise model to an ASCII file
    params = np.array(params)
    errs = np.array(errs)
    np.savetxt(str(output_noisefile),
               np.concatenate((wl[:, np.newaxis], params, errs), axis=1))
    # to recreate gsd v snr plots need to run at .25 rfl only
    rfl = 0.25
    rhotoa = rhoatm + transm * rfl / (1.0 - sphalb * rfl)
    rdn = rhotoa / np.pi * (irr * coszen)
    np.savetxt(str(output_snrfile), sbg_CBE.snr(rdn))


def err(x, L, nedl):
    if any(x < 0):
        return 9e99
    noi = x[0] * np.sqrt(x[1] + L) + x[2]
    return (nedl - noi)
