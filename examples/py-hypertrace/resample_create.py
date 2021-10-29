#!/usr/bin/env python3
#
# Authors: Ann Raiho
# loop script to match data to wavelength for spectral resolution experiment
# provide default hypertrace script with reflectance file in the hypertrace section
# only one reflectance in the config the script will make the new config to use to run the experiments
import json
from isofit.core.common import resample_spectrum
import spectral as sp
import numpy as np
import os.path
import spectral.io.envi as envi

def spectral_resample_data(hypertrace_configs):

    specRes = [10, 15, 20, 25, 30]
    # read in hypertrace configs
    with open(hypertrace_configs) as hypertrace_json_file:
        ht_configs_orig = json.load(hypertrace_json_file)

    print(ht_configs_orig['hypertrace']['reflectance_file'])

    # write back in original
    ht_configs = ht_configs_orig

    # read in reflectance from configs
    # note: getting wavelengths from image itself not the wavelength file in config
    origRefl = ht_configs['hypertrace']['reflectance_file']

    print(origRefl)

    image_hdr = sp.open_image(str(origRefl) + ".hdr")
    strOrigWavelengths = image_hdr.metadata['wavelength']
    arrOrigWavelengths = np.asarray(strOrigWavelengths)
    origWavelengths = arrOrigWavelengths.astype(np.float) / 1000

    imagem = image_hdr.open_memmap(writable=True)

    for ss in specRes:
        print(ss)

        # read new wavelength file
        strNewWaves = ['./hypertrace-data/instruments/', str(ss), 'nm/wavelengths.txt']
        newWaves = np.loadtxt("".join(strNewWaves))

        # write empty array for resampled image
        imagem_resampled = np.zeros(shape=(imagem.shape[0], imagem.shape[1], len(newWaves)))

        # resample to new wavelengths across image dimensions
        for pxlsx in range(0, imagem.shape[0]):
            for pxlsy in range(0, imagem.shape[1]):
                imagem_resampled[pxlsx, pxlsy, :] = resample_spectrum(imagem[pxlsx, pxlsy, :], origWavelengths[:], newWaves[:, 1], newWaves[:, 2])

        size = len(origRefl)
        # Slice string to remove last 5 characters from string
        mod_string = origRefl[:size - 5]

        # write out new reflectance
        strHdr = [mod_string, str(ss), '.envi.hdr']
        hdr_file = "".join(strHdr)
        strEnvi = [mod_string, str(ss), '.envi']
        envi_file = "".join(strEnvi)

        #TO DO: add overwrite arg so that you don't have to resample if you are just adjusting the configs

        envi.save_image(str(hdr_file), imagem_resampled, force=True, ext='.envi',
                        metadata={'wavelengths': newWaves[:, 1].astype(np.float32)})

        # edit configs to match new resolution
        ht_configs['wavelength_file'] = "".join(strNewWaves)
        ht_configs['hypertrace']['reflectance_file'] = [str(envi_file)]

        instrument_output_file = './hypertrace-data/instruments/output'
        noise_out = [instrument_output_file, '/', 'sbg_cbe_chromaD', '-specRes', str(ss), 'noisefile.txt']
        output_noisefile = "".join(noise_out)

        ht_configs['hypertrace']['noisefile'] = [output_noisefile]

        # write out configs for later
        config_out = ['./configs/', 'run_specRes', str(ss), '.json']
        config_outfile = "".join(config_out)
        json.dump(ht_configs, open(config_outfile, "w"), indent=1)
