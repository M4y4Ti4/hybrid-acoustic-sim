"""
a pipeline script to run both the wave and geometrical simulations
and combine them into a hybrid model 
"""

import os
import sys 
import numpy as np

HYBRID_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#adding both submodules to the path
sys.path.append(os.path.join(HYBRID_DIR, "DGsim", "edg-acoustics"))
sys.path.append(os.path.join(HYBRID_DIR, "RayroomProject", "rayroom"))


from hybridsim.shoebox.geo_wrapper import run_geometric 

from hybridsim.shoebox.wave_wrapper import run_wave 

from calibration import create_hybrid

def main():

    source_pos = [3.04, 2.59, 1.62]
    recx = 4.26
    recy = 1.76
    recz = 1.62
    rec_pos = np.array([recx, recy, recz])

    room_dim = [5,4,3]

    rir_duration = 0.1

    max_freq = 100

    mesh_input = os.path.join(HYBRID_DIR, "DGsim", "examples", "shoebox", "shoebox_lc1")

    geo_res = run_geometric(rec_pos=rec_pos, source_pos=source_pos, room_dim=room_dim, n_rays = 2000, max_hops = 150, rir_duration=rir_duration, ism_order = 5)

    wave_res = run_wave(mesh_input=mesh_input, max_freq=max_freq, recx=recx, recy=recy, recz=recz, source_pos=source_pos, rir_duration=rir_duration)

    hybrid_res = create_hybrid(geo_res = geo_res, wave_res = wave_res, crossover_hz = 100, fs = 44100)

    return wave_res, geo_res, hybrid_res

if __name__ == "__main__":
    main()