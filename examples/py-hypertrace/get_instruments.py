#!/usr/bin/env python3
#
# Authors: Ann Raiho

import os
import json
import shutil
from calculate_instrument_model import calculate_instrument_model

# space_res = [20, 30, 40, 50, 60, 75, 100]
instType = ['sbg_cbe_chromaA', 'sbg_cbe_chromaD']
integrationT = [.0030, .0044, .0059, .0074, .0089, .0111, .0148]
specRes = [10, 15, 20, 25, 30]
for ii in instType:
    # Spatial
    for tt in integrationT:

        # read original instrument config
        strConfig = ['./configs/', ii, '.json']
        joinConfig = "".join(strConfig)

        # assumes wavelength of 10nm
        configs = open(joinConfig)

        instrument_configs = json.load(configs)
        instrument_configs["integration_seconds"] = tt

        # alter integration time
        config_out = ['./configs/', 'intT', str(tt), ii, '.json']
        config_outfile = "".join(config_out)
        json.dump(instrument_configs, open(config_outfile, "w"))

        mainfile = './hypertrace-data/instruments/10nm'

        output_file = './hypertrace-data/instruments/output'
        noise_out = [output_file, '/', ii, '-intT', str(tt), 'noisefile.txt']
        output_noisefile = "".join(noise_out)

        snr_out = [output_file, '/', ii, '-intT', str(tt), 'snr.txt']
        output_snrfile = "".join(snr_out)

        calculate_instrument_model(ii, config_outfile, mainfile, output_noisefile, output_snrfile)

    # Spectral
    for ss in specRes:

        # read original instrument config
        strConfig = ['./configs/', ii, '.json']
        joinConfig = "".join(strConfig)

        # assumes wavelength of 10nm
        configs = open(joinConfig)
        instrument_configs = json.load(configs)
        strWave = ['./hypertrace-data/instruments/', str(ss), 'nm/wavelengths.txt']
        instrument_configs["wavelength_file"] = "".join(strWave)

        # alter integration time
        config_out = ['./configs/', 'specRes', str(ss), ii, '.json']
        config_outfile = "".join(config_out)
        json.dump(instrument_configs, open(config_outfile, "w"))

        strMainfile = ['./hypertrace-data/instruments/', str(ss), 'nm']
        mainfile = "".join(strMainfile)

        output_file = './hypertrace-data/instruments/output'
        noise_out = [output_file, '/', ii, '-specRes', str(ss), 'noisefile.txt']
        output_noisefile = "".join(noise_out)

        snr_out = [output_file, '/', ii, '-specRes', str(ss), 'snr.txt']
        output_snrfile = "".join(snr_out)

        calculate_instrument_model(ii, config_outfile, mainfile, output_noisefile, output_snrfile)
