#!/usr/bin/env python3
#
# Authors: Ann Raiho with inspiration from Adam Chlus and David Thompson

import copy
import pathlib
import json
import shutil
import logging
import os.path
import pandas
import math

import numpy as np
import spectral as sp
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

from isofit.core.isofit import Isofit
from isofit.core.fileio import IO
from isofit.core.common import resample_spectrum
from isofit.utils import empirical_line, segment, extractions
from isofit.utils.apply_oe import write_modtran_template

from scipy.linalg import inv, svd
from scipy.optimize import leastsq
from scipy.optimize import minimize
import spectral
import spectral.io.envi as envi
from scipy.io import loadmat
import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int, c_double
from multiprocessing import Pool
from isofit.core.common import VectorInterpolator

logger = logging.getLogger(__name__)

##################################################
##################################################


def do_vegetation_algorithm(outdir2: pathlib.Path,
                            algorithm_file: pathlib.Path,
                            est_refl_file: pathlib.Path):
    logger.info("INSIDE vegetation algorithm function")
    with open(algorithm_file) as json_file:
        trait_model = json.load(json_file)

    print(outdir2)
    print(algorithm_file)
    print(est_refl_file)

    coeffs = np.array(trait_model['model']['coefficients'])
    intercept = np.array(trait_model['model']['intercepts'])
    model_waves = np.array(trait_model['wavelengths'])
    transform = trait_model['model']['transform'][0]
    trait_name = trait_model['name']
    trait_file_name = os.path.join(outdir2, "trait.csv")

    # Load reflectance
    image = sp.open_image(str(est_refl_file) + ".hdr")
    wavelengths = image.metadata['wavelength']
    sw = np.asarray(wavelengths)
    swy = sw.astype(np.float)

    fwhm = np.ones(len(model_waves)) * 10
    # print(model_waves)

    # reflectance
    imagem = image.open_memmap(writable=True)
    imagem_resampled = np.zeros(shape=(imagem.shape[0], imagem.shape[1], len(model_waves)))  # hard coded! was 10,10,36

    print(len(swy))
    print('NEXT')
    print(len(model_waves))
    print('NEXT')

    print(imagem.shape)

    # resample to match
    for pxlsx in range(0, imagem.shape[0]):
        for pxlsy in range(0, imagem.shape[1]):
            resp_spec = resample_spectrum(imagem[pxlsx, pxlsy, :], swy, model_waves, fwhm)  # image, image, model, model
            imagem_resampled[pxlsx, pxlsy, :] = resp_spec / math.sqrt(sum(resp_spec ** 2))

    # idx = range(0,36) #need to match wavelengths to model_waves here
    # print(imagem.shape)
    # print(imagem_resampled.shape)

    # find_nearest()
    # rolling window over 10nm or nearest

    # Assign wavelength resampling settings
    # image.resampler['out_waves'] = model_waves
    # image.resampler['type'] = 'cubic'

    # Generate header
    # trait_header = image.get_header()
    # trait_header['bands']= 2
    # trait_header['data type'] = 4
    # trait_header['wavelength']= []
    # trait_header['fwhm']= []
    # trait_header['band names'] = ["%s_mean" % trait_name,
    #                          "%s_std" % trait_name]

    # writer = iIO.write_spectrum(trait_file_name)
    # iterator = image.iterate(by = 'chunk',
    #                     chunk_size = (500,500),resample=True)

    # while not iterator.complete:
    #     chunk = np.copy(iterator.read_next())
    #     line_start = iterator.current_line
    #     col_start = iterator.current_column

    # if  transform == "vnorm":
    #     norm = np.linalg.norm(chunk,axis=2)
    #     chunk = chunk/norm[:,:,np.newaxis]

    # if transform == "log(1/R)":
    #     chunk = np.log(1/chunk)

    # if transform == "mean":
    #     mean = chunk.mean(axis=2)
    #     chunk = chunk/mean[:,:,np.newaxis]
    # chunk[np.isnan(chunk)] =0

    trait_pred = np.einsum('jkl,ml->jkm', imagem_resampled, coeffs, optimize='optimal')
    trait_pred = trait_pred + intercept
    trait_mean = trait_pred.mean(axis=2)
    trait_std = trait_pred.std(ddof=1, axis=2)
    tunk = np.concatenate([trait_mean[:, :, np.newaxis], trait_std[:, :, np.newaxis]], axis=2)

    # writer.write_chunk(tunk,line_start,col_start)

    print(tunk.shape)
    tunk2 = tunk.reshape((-1, tunk.shape[2]), order='F')
    print(tunk2.shape)
    np.savetxt(trait_file_name, tunk2, delimiter=",", header='mean,sd', comments='')

#  Mineral algorithm from David Thompson
#  Least-Squares Fitting of a library spectrum to a target
#  Apply continuum removal on both spectra, based on the
#  interval "rng" which is a tuple of (low, high) wavelengths
#  old inputs: wl, x, lib, rng


def do_mineral_algorithm(outdir2: pathlib.Path,
                         algorithm_file: pathlib.Path,
                         algorithm_type: pathlib.Path,
                         est_refl_file: pathlib.Path):

    # Load reflectance
    image = sp.open_image(str(est_refl_file) + ".hdr")
    image_open = image.open_memmap(writable=True)
    x = np.asarray(image_open)
    # wavelengths #old wl
    wavelengths = image.metadata['wavelength']
    sw = np.asarray(wavelengths)
    swy = sw.astype(np.float)

    fwhm = np.ones(sw.shape) * 10

    # reference
    # Load a library spectrum
    ldata = np.loadtxt(algorithm_file, delimiter=' ', skiprows=1)  # not sure if we want the algorithm file here or what
    lrfl = resample_spectrum(ldata[:, 1], ldata[:, 0] * 1000.0, swy, fwhm)
    print(lrfl)

    lib = lrfl

    # range #we will probably want to define these in configs instead. just not sure because these are mineral algo specific.
    ranges = [[2100, 2300], [2200, 2430],
              [750, 1250], [750, 1250]]

    if algorithm_type == 'hematite':
        rng = ranges[2]
    elif algorithm_type == 'goethite':
        rng = ranges[3]
    elif algorithm_type == 'calcite':
        rng = ranges[1]
    elif algorithm_type == 'kaolinite':
        rng = ranges[0]
    else:
        raise ValueError(f"Invalid algorithm type {algorithm_type}")

    print(algorithm_type, rng)

    #
    # start fitting
    #

    # x[np.logical_not(np.isfinite(x))] = 0;
    swy[np.logical_not(np.isfinite(swy))] = 0
    lib[np.logical_not(np.isfinite(lib))] = 0

    # subset to our range of interest
    i1 = np.argmin(abs(swy - rng[0]))
    i2 = np.argmin(abs(swy - rng[-1]))

    x, wlf, lib = x[:, :, i1:i2], swy[i1:i2], lib[i1:i2];

    # Continuum level
    ends = np.array([0, -1], dtype=np.int32)

    # Continuum of library
    p = np.polyfit(wlf[ends], lib[ends], 1)
    lctm = np.polyval(p, wlf)
    lctmr = lib / lctm - 1.0

    print(lib)
    print('LIB')

    libfit = np.zeros(shape=(x.shape[0], x.shape[1], wlf.shape[0]))
    scale_save = np.zeros(shape=(x.shape[0], x.shape[1]))

    for pxlsx in range(0, x.shape[0]):
        for pxlsy in range(0, x.shape[1]):

            p = np.polyfit(wlf[ends], x[pxlsx, pxlsy, ends], 1)
            xctm = np.polyval(p, wlf)
            xctmr = x[pxlsx, pxlsy, :] / xctm - 1.0;

            # Fit a scaling term
            def err(scale):
                return sum(pow(scale * lctmr - xctmr, 2))

            scale = minimize_scalar(err, bracket=[0, 1])

            libfit[pxlsx, pxlsy, :] = (1.0 + scale.x * lctmr) * xctm;

            scale_save[pxlsx, pxlsy] = scale.fun

    # return wlf, libfit, scale.x

    # print(scale_save)

    print(scale_save)

    # libfit2 = libfit.reshape((-1,libfit.shape[2]),order = 'F')

    mineral_file_name = os.path.join(outdir2, "mineral.csv")
    colnames = list(range(0, x.shape[1]))
    np.savetxt(mineral_file_name, scale_save, header=str(colnames), delimiter=",", comments='')


def do_aquatic_algorithm(outdir2: pathlib.Path,
                         algorithm_file: pathlib.Path,
                         est_refl_file: pathlib.Path):

    # Load reflectance
    image = sp.open_image(str(est_refl_file) + ".hdr")
    image_open = image.open_memmap(writable=True)
    x = image_open
    # x = np.asarray(image_open)

    # wavelengths #old wl
    wavelengths = image.metadata['wavelength']
    sw = np.asarray(wavelengths)
    wl = sw.astype(np.float)

    lut = LUT(wl, algorithm_file)
    output = np.zeros(shape=(x.shape[0], x.shape[1], 5))  # 5 for 5 retrivals

    # loop over lines and samples
    for pxlsx in range(0, image_open.shape[0]):
        for pxlsy in range(0, image_open.shape[1]):

            solzen = 45  # solzens[col] #hard coded fix later
            x0 = [0 for q in lut.band_names]
            bounds = [(0, 790) for q in lut.band_names]
            res = minimize(err, x0, args=(x[pxlsx, pxlsy, :], lut, solzen),
                           method='tnc', tol=1e-9, bounds=bounds)
            xbest = res.x
            model = forward(xbest, lut, solzen)

            # calculate RMSE and construct the output vector
            residual = model - x[pxlsx, pxlsy, :]
            rmse = np.sqrt(pow(residual, 2).mean())
            output[pxlsx, pxlsy, :] = np.r_[xbest, rmse]

    output2 = output.reshape((-1, output.shape[2]), order='F')
    aqua_file_name = os.path.join(outdir2, "aquatic.csv")
    np.savetxt(aqua_file_name, output2, delimiter=",", header='vwind,chla,cdom,mineral,rmse', comments='')


def do_snow_algorithm(outdir2: pathlib.Path,
                      algorithm_file: pathlib.Path,
                      est_refl_file: pathlib.Path):

    # Load reflectance
    image = sp.open_image(str(est_refl_file) + ".hdr")
    image_open = image.open_memmap(writable=True)
    x = np.asarray(image_open)
    # wavelengths #old wl
    wavelengths = image.metadata['wavelength']
    sw = np.asarray(wavelengths)
    swy = sw.astype(np.float)

    fwhm = np.ones(sw.shape) * 10

    # reference
    # Load a library spectrum
    ldata = pandas.read_csv(algorithm_file)  # not sure if we want the algorithm file here or what
    print('wavelengths')
    ldata1 = np.asarray(ldata)
    print(ldata1[0, :])
    print('wavelengths')

    lrfl = resample_spectrum(ldata[:, 1], ldata[:, 0] * 1000.0, swy, fwhm)
    print(lrfl)

    lib1 = np.asarray(ldata)
    lib = np.transpose(lib1)

    # range #we will probably want to define these in configs instead. just not sure because these are mineral algo specific.
    rng = [890, 1710]

    #
    # start fitting
    #

    # x[np.logical_not(np.isfinite(x))] = 0;
    swy[np.logical_not(np.isfinite(swy))] = 0
    lib[np.logical_not(np.isfinite(lib))] = 0

    # subset to our range of interest
    i1 = np.argmin(abs(swy - rng[0]))
    i2 = np.argmin(abs(swy - rng[-1]))
    print(i1)
    print(i2)

    x, wlf, lib = x[:, :, i1:i2], swy[i1:i2], lib[:, i1:i2];

    print(lib.shape)

    # Continuum level
    ends = np.array([0, -1], dtype=np.int32)
    print(ends)

    # Continuum of library
    p = np.polyfit(wlf[ends], lib[ends], 1)
    lctm = np.polyval(p, wlf)
    lctmr = lib / lctm - 1.0

    libfit = np.zeros(shape=(x.shape[0], x.shape[1], lib.shape[0], wlf.shape[0]))
    scale_save = np.zeros(shape=(x.shape[0], x.shape[1]))
    scale_vec = np.zeros(shape=(lib.shape[0]))

    for pxlsx in range(0, x.shape[0]):
        for pxlsy in range(0, x.shape[1]):
            for libx in range(0, lib.shape[0]):

                p = np.polyfit(wlf[ends], x[pxlsx, pxlsy, ends], 1)
                xctm = np.polyval(p, wlf)
                xctmr = x[pxlsx, pxlsy, :] / xctm - 1.0;

            # Fit a scaling term
                def err(scale):
                    return sum(pow(scale * lctmr[libx, :] - xctmr, 2))

                scale = minimize_scalar(err, bracket=[0, 1])

                libfit[pxlsx, pxlsy, libx, :] = (1.0 + scale.x * lctmr[libx, :]) * xctm;

                print(libfit)

                scale_vec[libx] = mean(libfit)

            print(scale_vec)
            scale_save[pxlsx, pxlsy] = np.argmin(scale_vec)

    # return wlf, libfit, scale.x

    # print(scale_save)

    print(scale_save)

    # libfit2 = libfit.reshape((-1,libfit.shape[2]),order = 'F')

    snow_file_name = os.path.join(outdir2, "snow.csv")
    colnames = list(range(0, x.shape[1]))
    np.savetxt(snow_file_name, scale_save, header=str(colnames), delimiter=",", comments='')


class LUT:

    def __init__(self, wl, matfile=None):
        self.wl = wl
        D = loadmat(matfile)
        data = D['data']
        self.lut_grid = [grid[0] for grid in D['grids'][0]]
        self.interp_types = np.array(['n' for n in self.lut_grid])
        wl_LUT = D['wl'][0]
        self.grids = D['grids']
        self.band_names = [str(g).strip() for g in self.lut_grid[1:]]

        # resample all LUT spectra
        old_shape = data.shape
        new_shape = list(old_shape[:])
        new_shape[-1] = len(self.wl)
        nb = int(data.shape[-1])
        data = data.reshape((int(data.size / nb), nb))
        data_resamp = []
        for x in data:
            x_resamp = interp1d(wl_LUT, x, bounds_error=False,
                                fill_value='extrapolate')(wl)
            data_resamp.append(x_resamp)
        self.data = np.array(data_resamp).reshape(new_shape)
        self.data = VectorInterpolator(self.lut_grid, self.data,
                                       self.interp_types)
        self.use = np.logical_and(self.wl > wl_LUT[0], self.wl < wl_LUT[-1])

def forward(x, lut, solzen=45):

    return lut.data(np.r_[solzen, x])


def err(x, rfl, lut, solzen=45):
    mdl = forward(x, lut, solzen)
    er = np.sqrt(pow(mdl - rfl, 2)[lut.use].mean())
    return er


# TEST
# do_algorithm("/Users/araiho/isofit/examples/py-hypertrace/output-dirsig/example-dirsig/-2249895404685772980",
#     "/Users/araiho/isofit/examples/py-hypertrace/hypertrace-data/algorithm/chlorophylls_area_10nm.json",
#     "/Users/araiho/isofit/examples/py-hypertrace/output-dirsig/example-dirsig/-2249895404685772980/estimated-reflectance")
