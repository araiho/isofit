from resample_create import spectral_resample_data

path = ['./configs/example-cuprite.json']
indir = "".join(path)

print(indir)

spectral_resample_data(indir)
