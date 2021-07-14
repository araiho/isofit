#!/usr/bin/env python3
#
# Authors: Ann Raiho with inspiration from Adam Chlus

import copy
import pathlib
import json
import shutil
import logging
import os.path

import numpy as np
import spectral as sp
from scipy.io import loadmat
from scipy.interpolate import interp1d

from isofit.core.isofit import Isofit
from isofit.core.fileio import IO
from isofit.utils import empirical_line, segment, extractions
from isofit.utils.apply_oe import write_modtran_template

logger = logging.getLogger(__name__)


##################################################
def do_algorithm(outdir2: pathlib.Path,
               algorithm_file: pathlib.Path,
               est_refl_file: pathlib.Path):
    logger.info("INSIDE algorithm function")
    with open(algorithm_file) as json_file: trait_model = json.load(json_file)

    print(outdir2)
    print(algorithm_file)
    print(est_refl_file)

    coeffs = np.array(trait_model['model']['coefficients'])
    intercept = np.array(trait_model['model']['intercepts'])
    model_waves = np.array(trait_model['wavelengths'])
    transform = trait_model['model']['transform'][0]
    trait_name= trait_model['name']
    trait_file_name= os.path.join(outdir2, "trait.csv")
    
    # Load reflectance
    image = sp.open_image(str(est_refl_file) + ".hdr")

    wavelengths = image.metadata['wavelength']
    #print(wavelengths)
    #print(model_waves)
    imagem = image.open_memmap(writable=True)

    idx = range(0,36) #need to match wavelengths to model_waves here
    print(idx)

    #find_nearest()

    #rolling window over 10nm or nearest

    # Assign wavelength resampling settings
    #image.resampler['out_waves'] = model_waves
    #image.resampler['type'] = 'cubic'

    # Generate header
    #trait_header = image.get_header()
    #trait_header['bands']= 2
    #trait_header['data type'] = 4
    #trait_header['wavelength']= []
    #trait_header['fwhm']= []
    #trait_header['band names'] = ["%s_mean" % trait_name,
    #                          "%s_std" % trait_name]

    #writer = iIO.write_spectrum(trait_file_name)
    #iterator = image.iterate(by = 'chunk',
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
    trait_pred = np.einsum('jkl,ml->jkm', imagem[:,:,idx], coeffs, optimize='optimal')
    trait_pred = trait_pred + intercept
    trait_mean= trait_pred.mean(axis=2)
    trait_std = trait_pred.std(ddof=1,axis=2)
    tunk = np.concatenate([trait_mean[:,:,np.newaxis],trait_std[:,:,np.newaxis]],axis=2)
    
    
    # writer.write_chunk(tunk,line_start,col_start)  

    print(tunk.shape)
    tunk2 = tunk.reshape((-1,tunk.shape[2]),order = 'F')
    print(tunk2.shape)
    np.savetxt(trait_file_name, tunk2, delimiter=",", header = 'mean,sd', comments='')

import numpy as np
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

 
### TEST
# do_algorithm("/Users/araiho/isofit/examples/py-hypertrace/output-dirsig/example-dirsig/-2249895404685772980",
#     "/Users/araiho/isofit/examples/py-hypertrace/hypertrace-data/algorithm/chlorophylls_area_10nm.json",
#     "/Users/araiho/isofit/examples/py-hypertrace/output-dirsig/example-dirsig/-2249895404685772980/estimated-reflectance")



