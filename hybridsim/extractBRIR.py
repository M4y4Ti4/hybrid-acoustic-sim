import numpy as np
from scipy.io import wavfile



def geotoBRIR(geo_rir):
    fs = 44100
    brir_l = geo_rir["brir_l"]
    brir_r = geo_rir["brir_r"]
    if brir_l.shape != brir_r.shape:
        raise ValueError("Both channels must have the same number of samples")
    stereo_brir = np.stack((brir_l, brir_r), axis=1)
    wavfile.write('geo_brir', fs, stereo_brir)
    print("done")
    return stereo_brir

geo_brir = geotoBRIR()