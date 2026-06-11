import numpy as np
import sys
import os
from scipy.io import wavfile
from rayroom.core.auralisation import render_brir, plot_brir, load_hrtf, get_hrir

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

def geotoBRIR(geo_rir):
    fs = 44100
    brir_l = geo_rir["brir_l"].squeeze()
    brir_r = geo_rir["brir_r"].squeeze()
    print(brir_r.shape)
    if brir_l.shape != brir_r.shape:
        raise ValueError("Both channels must have the same number of samples")
    stereo_brir = np.column_stack([brir_l, brir_r])
    wavfile.write(r'C:\Masters\Hybrid\hybridsim\shoebox\results\pos1\geo_brir_newhrtf.wav', fs, stereo_brir)
    print("done")
    return stereo_brir

geo_data = np.load(r"C:\Masters\Hybrid\hybridsim\shoebox\results\pos1\rir_shoebox_pos1_200000_newhrtf.npz")

hybrid_data = np.load(r"C:\Masters\Hybrid\hybridsim\shoebox\results\pos1\hybrid_rir_newcarpet_withphase_ismspec.npz")

rir = geo_data["rir_total"]

hybrid_left = np.load(r"C:\Masters\Hybrid\hybridsim\shoebox\results\pos1\hybrid_BRIR.npz")["hybrid_rir_left"]

hybrid_right = np.load(r"C:\Masters\Hybrid\hybridsim\shoebox\results\pos1\hybrid_BRIR.npz")["hybrid_rir_right"]

def create_BRIR(left_signal, right_signal, output_path, fs = 44100):
    if left_signal.shape != right_signal.shape: 
        raise ValueError("Both channels must be same length")
    stereo_brir = np.column_stack([left_signal, right_signal])
    wavfile.write(output_path, fs, stereo_brir)
    print("done")
    return stereo_brir

output_path = r"C:\Masters\Hybrid\hybridsim\shoebox\results\pos1\hybrid_BRIR.wav"

create_BRIR(hybrid_left, hybrid_right, output_path)